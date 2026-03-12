"""Model registry — serialize and load trained artifacts.

Saves and loads all artifacts that Streamlit needs:
  - Trained model objects (.joblib)
  - Fitted StandardScaler (.joblib)
  - Feature column list (.joblib)
  - Training metadata (.json)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def save_model(
    model: Any,
    model_name: str,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save a trained model to the models directory.

    Args:
        model: Trained sklearn/xgboost model object.
        model_name: Filename stem (e.g. ``"xgboost_primary"``).
        metadata: Optional metadata dict (params, scores, etc.).

    Returns:
        Path to the saved .joblib file.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / f"{model_name}.joblib"
    joblib.dump(model, model_path)
    logger.info("Saved model: %s", model_path)

    if metadata:
        meta_path = MODELS_DIR / f"{model_name}_metadata.json"
        metadata["saved_at"] = datetime.utcnow().isoformat()
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info("Saved metadata: %s", meta_path)

    return model_path


def load_model(model_name: str) -> Any:
    """Load a trained model from the models directory.

    Args:
        model_name: Filename stem (e.g. ``"xgboost_primary"``).

    Returns:
        Loaded model object.

    Raises:
        FileNotFoundError: If the model file doesn't exist.
    """
    model_path = MODELS_DIR / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)
    logger.info("Loaded model: %s", model_path)
    return model


def save_scaler(scaler: Any, name: str = "feature_scaler") -> Path:
    """Save a fitted StandardScaler.

    Args:
        scaler: Fitted sklearn StandardScaler.
        name: Filename stem.

    Returns:
        Path to the saved file.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(scaler, path)
    logger.info("Saved scaler: %s", path)
    return path


def load_scaler(name: str = "feature_scaler") -> Any:
    """Load a fitted StandardScaler.

    Args:
        name: Filename stem.

    Returns:
        Loaded scaler object.
    """
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Scaler not found: {path}")
    return joblib.load(path)


def save_feature_columns(
    columns: list[str],
    name: str = "feature_columns",
) -> Path:
    """Save the ordered list of feature columns used for training.

    This ensures Streamlit uses the exact same features in the same
    order when making predictions.

    Args:
        columns: Ordered list of feature column names.
        name: Filename stem.

    Returns:
        Path to the saved file.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(columns, path)
    logger.info("Saved %d feature columns: %s", len(columns), path)
    return path


def load_feature_columns(name: str = "feature_columns") -> list[str]:
    """Load the ordered list of feature columns.

    Args:
        name: Filename stem.

    Returns:
        List of feature column names.
    """
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Feature columns not found: {path}")
    columns: list[str] = joblib.load(path)
    return columns


def list_saved_models() -> list[dict[str, Any]]:
    """List all saved models with their metadata.

    Returns:
        List of dicts with model info (name, path, metadata).
    """
    if not MODELS_DIR.exists():
        return []

    models: list[dict[str, Any]] = []
    for model_file in MODELS_DIR.glob("*.joblib"):
        if model_file.stem in ("feature_scaler", "feature_columns"):
            continue

        info: dict[str, Any] = {
            "name": model_file.stem,
            "path": str(model_file),
            "size_mb": round(model_file.stat().st_size / (1024 * 1024), 2),
        }

        # Check for metadata file
        meta_file = model_file.with_name(f"{model_file.stem}_metadata.json")
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                info["metadata"] = json.load(f)

        models.append(info)

    return models
