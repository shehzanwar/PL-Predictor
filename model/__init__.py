"""PL-Predictor model package — training, prediction, and evaluation.

Public API:
    - ``PLModelTrainer``: Train all models with walk-forward validation.
    - ``PLPredictor``: Ensemble prediction interface for Streamlit.
    - ``EvaluationResult``: Container for evaluation metrics.
"""

from model.evaluator import EvaluationResult
from model.predictor import PLPredictor
from model.trainer import PLModelTrainer

__all__ = [
    "PLModelTrainer",
    "PLPredictor",
    "EvaluationResult",
]
