"""Model trainer — XGBoost, Random Forest, Logistic Regression.

Trains all three models using walk-forward time-series validation
(K-Fold is banned). Produces calibrated probabilities for the
soft-voting ensemble used by the predictor.

Usage::

    python -m model.trainer
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from features.validation import TimeSeriesSplitter
from model.evaluator import (
    EvaluationResult,
    evaluate_walk_forward,
    format_evaluation_report,
)
from model.registry import (
    save_feature_columns,
    save_model,
    save_scaler,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"

# Columns that are identifiers, not model features
ID_COLUMNS = [
    "date", "home_team", "away_team", "season", "matchweek",
    "venue", "referee", "attendance", "home_goals", "away_goals",
    "home_xg", "away_xg", "understat_match_id", "is_result",
    "fixture_id", "round", "status", "fetch_timestamp",
]

# Walk-forward validation config
N_SPLITS: int = 10
MIN_TRAIN_MATCHES: int = 150  # ~1 full season before first validation fold


class PLModelTrainer:
    """Train and evaluate the PL Predictor model ensemble.

    Trains three models with walk-forward validation:
        1. **XGBoost** (primary) — Gradient-boosted trees.
        2. **Random Forest** (secondary) — Bagged trees for calibration.
        3. **Logistic Regression** (baseline) — Linear benchmark.

    After evaluation, the best-performing models are retrained on the
    full dataset and serialized to ``models/`` for Streamlit.

    Example::

        trainer = PLModelTrainer()
        results = trainer.run()
    """

    def __init__(
        self,
        feature_matrix_path: str | Path | None = None,
    ) -> None:
        if feature_matrix_path is None:
            # Prefer pruned matrix; fall back to unpruned
            pruned = FEATURES_DIR / "pl_feature_matrix_pruned.csv"
            unpruned = FEATURES_DIR / "pl_feature_matrix.csv"
            feature_matrix_path = pruned if pruned.exists() else unpruned

        self.feature_matrix_path = Path(feature_matrix_path)
        self.splitter = TimeSeriesSplitter(
            n_splits=N_SPLITS,
            min_train_matches=MIN_TRAIN_MATCHES,
        )

        # Model definitions
        self.models: dict[str, Any] = {
            "xgboost_primary": XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                reg_alpha=0.1,
                reg_lambda=1.0,
                objective="multi:softprob",
                num_class=3,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
                use_label_encoder=False,
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features="sqrt",
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            "logistic_baseline": LogisticRegression(
                penalty="l2",
                C=1.0,
                solver="lbfgs",
                max_iter=3000,
                multi_class="multinomial",
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        }

        self.scaler = StandardScaler()
        self.feature_columns: list[str] = []

    def run(self) -> list[EvaluationResult]:
        """Execute the full training and evaluation pipeline.

        Steps:
            1. Load and prepare the feature matrix.
            2. Evaluate each model via walk-forward validation.
            3. Retrain all models on full data.
            4. Serialize artifacts to ``models/``.
            5. Generate evaluation report.

        Returns:
            List of EvaluationResult objects (one per model).
        """
        logger.info("=" * 60)
        logger.info("PL-Predictor Model Training Pipeline")
        logger.info("=" * 60)

        # Step 1: Load data
        df = pd.read_csv(self.feature_matrix_path, parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)
        logger.info("Loaded feature matrix: %d rows from %s", len(df), self.feature_matrix_path)

        # Identify feature columns
        self.feature_columns = [
            c for c in df.columns
            if c not in ID_COLUMNS
            and c != "target"
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        logger.info("Using %d feature columns", len(self.feature_columns))

        X = df[self.feature_columns].copy()
        y = df["target"].values.astype(int)

        # Handle NaN/inf
        X = X.fillna(X.median())
        X = X.replace([np.inf, -np.inf], 0).fillna(0)

        # Step 2: Walk-forward evaluation for each model
        all_results: list[EvaluationResult] = []

        for model_name, model in self.models.items():
            logger.info("\n--- Evaluating: %s ---", model_name)
            result = self._walk_forward_evaluate(
                X.values, y, df, model, model_name
            )
            all_results.append(result)

        # Step 3: Retrain on full data
        logger.info("\n--- Retraining on full dataset ---")
        X_scaled = self.scaler.fit_transform(X.values)

        for model_name, model in self.models.items():
            logger.info("Training %s on %d samples...", model_name, len(X_scaled))
            model.fit(X_scaled, y)

        # Step 4: Serialize artifacts
        logger.info("\n--- Saving artifacts ---")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        for model_name, model in self.models.items():
            result = next(r for r in all_results if r.model_name == model_name)
            save_model(model, model_name, metadata={
                "accuracy": result.accuracy,
                "log_loss": result.log_loss_score,
                "brier_score": result.brier_score,
                "n_features": len(self.feature_columns),
                "n_training_samples": len(X_scaled),
                "n_cv_folds": N_SPLITS,
            })

        save_scaler(self.scaler)
        save_feature_columns(self.feature_columns)

        # Step 5: Generate report
        report = format_evaluation_report(all_results)

        report_path = MODELS_DIR / "model_evaluation.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# PL Predictor — Model Evaluation Report\n\n")
            f.write(f"**Training Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write(f"**Dataset:** {len(df)} matches, {len(self.feature_columns)} features\n\n")
            f.write(f"**Validation:** {N_SPLITS}-fold walk-forward time-series split\n\n")
            f.write(report)
        logger.info("Saved evaluation report: %s", report_path)

        logger.info("=" * 60)
        logger.info("Training Pipeline Complete")
        logger.info("=" * 60)

        return all_results

    def _walk_forward_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        df: pd.DataFrame,
        model: Any,
        model_name: str,
    ) -> EvaluationResult:
        """Evaluate a single model using walk-forward validation.

        Each fold:
            1. Fit scaler on training data only.
            2. Scale train and validation sets.
            3. Train model on training fold.
            4. Predict probabilities on validation fold.

        Args:
            X: Unscaled feature matrix.
            y: Target vector.
            df: Full DataFrame (for date-based splitting).
            model: sklearn-compatible model.
            model_name: Model identifier.

        Returns:
            EvaluationResult with per-fold and aggregate metrics.
        """
        fold_predictions: list[dict[str, Any]] = []

        for train_idx, val_idx, split_info in self.splitter.split(df, date_col="date"):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Fit scaler on training data only (prevents leakage)
            fold_scaler = StandardScaler()
            X_train_scaled = fold_scaler.fit_transform(X_train)
            X_val_scaled = fold_scaler.transform(X_val)

            # Clone model for fresh state each fold
            from sklearn.base import clone
            fold_model = clone(model)
            fold_model.fit(X_train_scaled, y_train)

            # Predict
            y_pred = fold_model.predict(X_val_scaled)
            y_proba = fold_model.predict_proba(X_val_scaled)

            # Ensure y_proba has 3 columns (some folds may miss a class)
            if y_proba.shape[1] < 3:
                full_proba = np.zeros((len(y_val), 3))
                for i, cls in enumerate(fold_model.classes_):
                    full_proba[:, int(cls)] = y_proba[:, i]
                y_proba = full_proba

            fold_predictions.append({
                "fold": split_info.fold,
                "y_true": y_val,
                "y_pred": y_pred,
                "y_proba": y_proba,
            })

        return evaluate_walk_forward(fold_predictions, model_name)


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def main() -> None:
    """CLI entry point for model training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    trainer = PLModelTrainer()
    trainer.run()


if __name__ == "__main__":
    main()
