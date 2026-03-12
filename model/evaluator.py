"""Model evaluator — log-loss, Brier score, and per-class metrics.

Evaluates probability calibration and prediction quality beyond
raw accuracy. For a 3-class problem (Home/Draw/Away), we compute:
  - Multi-class log-loss (measures probability quality)
  - Multi-class Brier score (measures probability calibration)
  - Per-class F1, precision, recall
  - Accuracy (for reference only — not the primary metric)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    log_loss,
)

logger = logging.getLogger(__name__)

LABEL_NAMES: dict[int, str] = {0: "Home Win", 1: "Draw", 2: "Away Win"}


@dataclass
class EvaluationResult:
    """Container for a single model's evaluation metrics.

    Attributes:
        model_name: Identifier for the model.
        accuracy: Raw classification accuracy.
        log_loss_score: Multi-class logarithmic loss (lower is better).
        brier_score: Multi-class Brier score (lower is better).
        per_class_report: Dict from sklearn classification_report.
        fold_results: Per-fold metrics for walk-forward validation.
    """

    model_name: str
    accuracy: float
    log_loss_score: float
    brier_score: float
    per_class_report: dict[str, Any] = field(default_factory=dict)
    fold_results: list[dict[str, float]] = field(default_factory=list)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "model",
) -> EvaluationResult:
    """Evaluate model predictions with comprehensive metrics.

    Args:
        y_true: True labels (0=Home, 1=Draw, 2=Away).
        y_pred: Predicted labels.
        y_proba: Predicted probabilities, shape (n_samples, 3).
        model_name: Identifier for logging.

    Returns:
        EvaluationResult with all computed metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    ll = _compute_log_loss(y_true, y_proba)
    brier = _compute_multiclass_brier(y_true, y_proba)
    report = classification_report(
        y_true, y_pred,
        target_names=list(LABEL_NAMES.values()),
        output_dict=True,
        zero_division=0,
    )

    result = EvaluationResult(
        model_name=model_name,
        accuracy=round(acc, 4),
        log_loss_score=round(ll, 4),
        brier_score=round(brier, 4),
        per_class_report=report,
    )

    logger.info("--- %s Evaluation ---", model_name)
    logger.info("  Accuracy:    %.4f", acc)
    logger.info("  Log-loss:    %.4f", ll)
    logger.info("  Brier Score: %.4f", brier)

    return result


def evaluate_walk_forward(
    fold_predictions: list[dict[str, Any]],
    model_name: str = "model",
) -> EvaluationResult:
    """Aggregate evaluation across all walk-forward folds.

    Args:
        fold_predictions: List of dicts, each with keys:
            ``y_true``, ``y_pred``, ``y_proba``, ``fold``.
        model_name: Model identifier.

    Returns:
        EvaluationResult with both aggregate and per-fold metrics.
    """
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_proba: list[np.ndarray] = []
    fold_results: list[dict[str, float]] = []

    for fold_data in fold_predictions:
        y_true = np.asarray(fold_data["y_true"])
        y_pred = np.asarray(fold_data["y_pred"])
        y_proba = np.asarray(fold_data["y_proba"])

        fold_acc = accuracy_score(y_true, y_pred)
        fold_ll = _compute_log_loss(y_true, y_proba)
        fold_brier = _compute_multiclass_brier(y_true, y_proba)

        fold_results.append({
            "fold": fold_data.get("fold", -1),
            "accuracy": round(fold_acc, 4),
            "log_loss": round(fold_ll, 4),
            "brier_score": round(fold_brier, 4),
            "n_samples": len(y_true),
        })

        all_true.append(y_true)
        all_pred.append(y_pred)
        all_proba.append(y_proba)

    # Aggregate
    y_true_all = np.concatenate(all_true)
    y_pred_all = np.concatenate(all_pred)
    y_proba_all = np.concatenate(all_proba)

    result = evaluate_predictions(y_true_all, y_pred_all, y_proba_all, model_name)
    result.fold_results = fold_results

    # Log per-fold
    for fr in fold_results:
        logger.info(
            "  Fold %d: acc=%.4f  ll=%.4f  brier=%.4f  (n=%d)",
            fr["fold"], fr["accuracy"], fr["log_loss"], fr["brier_score"], fr["n_samples"],
        )

    return result


def format_evaluation_report(results: list[EvaluationResult]) -> str:
    """Format evaluation results as a markdown table for reporting.

    Args:
        results: List of evaluation results from different models.

    Returns:
        Markdown-formatted comparison table.
    """
    lines: list[str] = []
    lines.append("## Model Comparison\n")
    lines.append("| Model | Accuracy | Log-Loss | Brier Score |")
    lines.append("|---|---|---|---|")

    for r in results:
        lines.append(f"| {r.model_name} | {r.accuracy:.4f} | {r.log_loss_score:.4f} | {r.brier_score:.4f} |")

    lines.append("\n## Per-Class Metrics\n")

    for r in results:
        lines.append(f"### {r.model_name}\n")
        lines.append("| Class | Precision | Recall | F1-Score | Support |")
        lines.append("|---|---|---|---|---|")

        for label_id, label_name in LABEL_NAMES.items():
            if label_name in r.per_class_report:
                cls = r.per_class_report[label_name]
                lines.append(
                    f"| {label_name} | {cls.get('precision', 0):.3f} | "
                    f"{cls.get('recall', 0):.3f} | {cls.get('f1-score', 0):.3f} | "
                    f"{cls.get('support', 0):.0f} |"
                )
        lines.append("")

    if results and results[0].fold_results:
        lines.append("## Walk-Forward Fold Breakdown\n")
        lines.append("| Fold | Accuracy | Log-Loss | Brier Score | Samples |")
        lines.append("|---|---|---|---|---|")
        for fr in results[0].fold_results:
            lines.append(
                f"| {fr['fold']} | {fr['accuracy']:.4f} | "
                f"{fr['log_loss']:.4f} | {fr['brier_score']:.4f} | "
                f"{fr['n_samples']:.0f} |"
            )
        lines.append("")

    return "\n".join(lines)


# ------------------------------------------------------------------
# Internal scoring functions
# ------------------------------------------------------------------

def _compute_log_loss(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Compute multi-class log-loss with label safety."""
    try:
        return float(log_loss(y_true, y_proba, labels=[0, 1, 2]))
    except ValueError as e:
        logger.warning("Log-loss computation failed: %s", e)
        return 999.0


def _compute_multiclass_brier(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Compute multi-class Brier score.

    Brier score for K classes:
        BS = (1/N) * Σ Σ (p_ik - y_ik)²

    where p_ik is the predicted probability for class k and y_ik
    is 1 if sample i belongs to class k, else 0.

    Lower is better. Random baseline ≈ 0.667 for 3 classes.

    Args:
        y_true: True labels (0, 1, 2).
        y_proba: Predicted probabilities, shape (n, 3).

    Returns:
        Brier score as a float.
    """
    n_classes = y_proba.shape[1] if y_proba.ndim > 1 else 3

    # One-hot encode true labels
    y_onehot = np.zeros((len(y_true), n_classes))
    for i, label in enumerate(y_true):
        if 0 <= int(label) < n_classes:
            y_onehot[i, int(label)] = 1.0

    brier = float(np.mean(np.sum((y_proba - y_onehot) ** 2, axis=1)))
    return round(brier, 4)
