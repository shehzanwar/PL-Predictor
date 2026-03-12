"""Match prediction interface — soft-voting ensemble.

Loads trained models from ``models/`` and produces calibrated
probability distributions over {Home Win, Draw, Away Win} for
any given match.

This module is the primary interface for the Streamlit UI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from model.registry import (
    load_feature_columns,
    load_model,
    load_scaler,
)

logger = logging.getLogger(__name__)

LABEL_NAMES: dict[int, str] = {0: "Home Win", 1: "Draw", 2: "Away Win"}

# Ensemble weights (optimized during training; XGBoost gets highest weight)
DEFAULT_WEIGHTS: dict[str, float] = {
    "xgboost_primary": 0.50,
    "random_forest": 0.35,
    "logistic_baseline": 0.15,
}


@dataclass
class MatchPrediction:
    """Prediction output for a single match.

    Attributes:
        home_team: Name of the home team.
        away_team: Name of the away team.
        probabilities: Dict mapping outcome to probability.
        predicted_outcome: Most likely outcome string.
        confidence: Probability of the predicted outcome.
        model_predictions: Per-model probability breakdowns.
    """

    home_team: str
    away_team: str
    probabilities: dict[str, float]
    predicted_outcome: str
    confidence: float
    model_predictions: dict[str, dict[str, float]]


class PLPredictor:
    """Ensemble match outcome predictor.

    Loads trained XGBoost, Random Forest, and Logistic Regression
    models and produces soft-voting ensemble predictions.

    The ensemble blends each model's probability distribution using
    configurable weights (default: 50% XGBoost, 35% RF, 15% LR).

    Example::

        predictor = PLPredictor()
        prediction = predictor.predict_match(feature_row)
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.weights = weights or DEFAULT_WEIGHTS

        # Load artifacts
        try:
            self.feature_columns = load_feature_columns()
            self.scaler = load_scaler()
            self.models: dict[str, Any] = {}
            for model_name in self.weights:
                self.models[model_name] = load_model(model_name)
            logger.info(
                "Loaded %d models and %d feature columns",
                len(self.models),
                len(self.feature_columns),
            )
        except FileNotFoundError as e:
            logger.error("Failed to load model artifacts: %s", e)
            raise

    def predict_match(
        self,
        features: pd.Series | pd.DataFrame | dict[str, Any],
        home_team: str = "",
        away_team: str = "",
    ) -> MatchPrediction:
        """Predict the outcome of a single match.

        Args:
            features: Feature values for the match. Can be a dict,
                Series, or single-row DataFrame.
            home_team: Home team name (for display).
            away_team: Away team name (for display).

        Returns:
            MatchPrediction with probabilities and outcome.
        """
        X = self._prepare_features(features)

        # Get per-model predictions
        model_preds: dict[str, dict[str, float]] = {}
        ensemble_proba = np.zeros(3)

        for model_name, model in self.models.items():
            proba = model.predict_proba(X)[0]
            weight = self.weights.get(model_name, 0.0)
            ensemble_proba += weight * proba

            model_preds[model_name] = {
                LABEL_NAMES[i]: round(float(proba[i]), 4)
                for i in range(3)
            }

        # Normalize ensemble probabilities
        total = ensemble_proba.sum()
        if total > 0:
            ensemble_proba = ensemble_proba / total

        # Build result
        probabilities = {
            LABEL_NAMES[i]: round(float(ensemble_proba[i]), 4)
            for i in range(3)
        }
        predicted_idx = int(np.argmax(ensemble_proba))
        predicted_outcome = LABEL_NAMES[predicted_idx]
        confidence = float(ensemble_proba[predicted_idx])

        return MatchPrediction(
            home_team=home_team,
            away_team=away_team,
            probabilities=probabilities,
            predicted_outcome=predicted_outcome,
            confidence=round(confidence, 4),
            model_predictions=model_preds,
        )

    def predict_matchweek(
        self,
        feature_df: pd.DataFrame,
    ) -> list[MatchPrediction]:
        """Predict outcomes for an entire matchweek.

        Args:
            feature_df: DataFrame with one row per match,
                containing feature columns plus ``home_team``
                and ``away_team``.

        Returns:
            List of MatchPrediction objects.
        """
        predictions: list[MatchPrediction] = []

        for _, row in feature_df.iterrows():
            pred = self.predict_match(
                features=row,
                home_team=str(row.get("home_team", "")),
                away_team=str(row.get("away_team", "")),
            )
            predictions.append(pred)

        return predictions

    def _prepare_features(
        self,
        features: pd.Series | pd.DataFrame | dict[str, Any],
    ) -> np.ndarray:
        """Convert input features to a scaled numpy array.

        Ensures feature ordering matches the training columns
        and applies the fitted scaler.

        Args:
            features: Raw feature values.

        Returns:
            Scaled feature array of shape (1, n_features).
        """
        if isinstance(features, dict):
            features = pd.Series(features)
        if isinstance(features, pd.DataFrame):
            features = features.iloc[0]

        # Extract only the columns used during training, in order
        values: list[float] = []
        for col in self.feature_columns:
            val = features.get(col, 0.0)
            if pd.isna(val):
                val = 0.0
            values.append(float(val))

        X = np.array(values).reshape(1, -1)
        X_scaled: np.ndarray = self.scaler.transform(X)
        return X_scaled
