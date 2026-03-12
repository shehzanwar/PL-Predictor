"""PL-Predictor features package — feature engineering modules.

Public API:
    - ``run_feature_pipeline()``: Full feature engineering pipeline.
    - ``run_feature_selection()``: Automated SHAP/Lasso pruning.
    - ``TimeSeriesSplitter``: Walk-forward validation (K-Fold banned).
"""

from features.feature_selection import run_feature_selection
from features.pipeline import run_feature_pipeline
from features.validation import TimeSeriesSplitter

__all__ = [
    "run_feature_pipeline",
    "run_feature_selection",
    "TimeSeriesSplitter",
]
