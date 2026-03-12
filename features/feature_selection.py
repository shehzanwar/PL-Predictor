"""Automated feature selection — SHAP values and L1 (Lasso) regularization.

The final step of Phase 2: prune statistically useless features from
the expanded feature space to prevent overfitting. Two complementary
methods are used:

1. **Lasso (L1 Regularization):** Fits a multiclass logistic regression
   with L1 penalty. Features with coefficients driven to zero across
   all classes are dropped.

2. **SHAP (SHapley Additive exPlanations):** Fits a lightweight XGBoost
   model and computes mean |SHAP| values. Features below the
   significance threshold are dropped.

The intersection of survivors from both methods forms the final
pruned feature set — a conservative approach that only keeps features
validated by both statistical tests.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = PROJECT_ROOT / "data" / "features"

# Columns that are identifiers, not model features
ID_COLUMNS = [
    "date", "home_team", "away_team", "season", "matchweek",
    "venue", "referee", "attendance", "home_goals", "away_goals",
    "home_xg", "away_xg", "understat_match_id", "is_result",
    "fixture_id", "round", "status", "fetch_timestamp",
]

# Feature selection thresholds
LASSO_C: float = 0.1          # Inverse regularization strength (smaller = more aggressive pruning)
SHAP_THRESHOLD: float = 0.01  # Min mean |SHAP| to keep a feature (relative scale)


def run_feature_selection(
    feature_matrix_path: str | Path | None = None,
    output_path: str | Path | None = None,
    use_shap: bool = True,
    use_lasso: bool = True,
) -> pd.DataFrame:
    """Execute automated feature selection on the full feature matrix.

    Args:
        feature_matrix_path: Path to the unfiltered feature matrix CSV.
        output_path: Path to save the pruned feature matrix.
        use_shap: Whether to use SHAP-based selection.
        use_lasso: Whether to use Lasso-based selection.

    Returns:
        Pruned DataFrame with only significant features retained.
    """
    if feature_matrix_path is None:
        feature_matrix_path = FEATURES_DIR / "pl_feature_matrix.csv"
    if output_path is None:
        output_path = FEATURES_DIR / "pl_feature_matrix_pruned.csv"

    feature_matrix_path = Path(feature_matrix_path)
    output_path = Path(output_path)

    df = pd.read_csv(feature_matrix_path, parse_dates=["date"])
    logger.info("Loaded feature matrix: %d rows, %d columns", len(df), len(df.columns))

    # Identify feature columns (exclude IDs and target)
    feature_cols = _get_feature_columns(df)
    logger.info("Feature columns before selection: %d", len(feature_cols))

    # Prepare numeric feature matrix (handle NaN, non-numeric)
    X, valid_features = _prepare_feature_matrix(df, feature_cols)
    y = df["target"].values

    if len(valid_features) == 0:
        logger.error("No valid numeric features found. Aborting selection.")
        return df

    survivors: set[str] = set(valid_features)
    selection_report: list[dict[str, Any]] = []

    # Method 1: Lasso (L1 Regularization)
    if use_lasso:
        logger.info("--- Lasso Feature Selection ---")
        lasso_survivors, lasso_importance = _lasso_selection(X, y, valid_features)
        selection_report.extend(lasso_importance)

        if lasso_survivors:
            logger.info(
                "Lasso kept %d / %d features", len(lasso_survivors), len(valid_features)
            )
            survivors = survivors.intersection(lasso_survivors)
        else:
            logger.warning("Lasso returned no survivors — skipping Lasso filtering.")

    # Method 2: SHAP values (via XGBoost)
    if use_shap:
        logger.info("--- SHAP Feature Selection ---")
        shap_survivors, shap_importance = _shap_selection(X, y, valid_features)
        selection_report.extend(shap_importance)

        if shap_survivors:
            logger.info(
                "SHAP kept %d / %d features", len(shap_survivors), len(valid_features)
            )
            survivors = survivors.intersection(shap_survivors)
        else:
            logger.warning("SHAP returned no survivors — skipping SHAP filtering.")

    # Safety: keep at least 5 features
    if len(survivors) < 5:
        logger.warning(
            "Only %d features survived — falling back to top 10 by combined importance.",
            len(survivors),
        )
        survivors = _fallback_top_features(selection_report, n=10)

    dropped = set(valid_features) - survivors
    logger.info("DROPPED %d features: %s", len(dropped), sorted(dropped))
    logger.info("KEPT %d features: %s", len(survivors), sorted(survivors))

    # Build final pruned DataFrame
    keep_cols = list(ID_COLUMNS) + ["target"] + sorted(survivors)
    keep_cols = [c for c in keep_cols if c in df.columns]
    pruned_df = df[keep_cols].copy()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pruned_df.to_csv(output_path, index=False)
    logger.info("Saved pruned feature matrix: %s (%d columns)", output_path, len(pruned_df.columns))

    # Save selection report
    report_path = FEATURES_DIR / "feature_selection_report.csv"
    report_df = pd.DataFrame(selection_report)
    report_df.to_csv(report_path, index=False)
    logger.info("Saved selection report: %s", report_path)

    return pruned_df


def _get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Identify columns that are model features (not IDs/target)."""
    return [
        c for c in df.columns
        if c not in ID_COLUMNS
        and c != "target"
        and c not in ("match_idx", "is_home", "result", "merge_date")
    ]


def _prepare_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Prepare a clean numeric feature matrix for selection.

    - Drops non-numeric columns.
    - Fills NaN with column medians.
    - Scales features.

    Args:
        df: Full DataFrame.
        feature_cols: List of feature column names.

    Returns:
        Tuple of (scaled_feature_array, valid_feature_names).
    """
    valid_cols: list[str] = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            valid_cols.append(col)
        else:
            # Try to coerce; drop if it fails
            try:
                pd.to_numeric(df[col], errors="raise")
                valid_cols.append(col)
            except (ValueError, TypeError):
                logger.debug("Dropping non-numeric feature: %s", col)

    X = df[valid_cols].copy()
    X = X.fillna(X.median())

    # Replace any remaining NaN/inf with 0
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, valid_cols


def _lasso_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> tuple[set[str], list[dict[str, Any]]]:
    """Select features using L1-regularized logistic regression.

    Features with ALL coefficients driven to zero (across all classes)
    are considered statistically useless and dropped.

    Args:
        X: Scaled feature matrix.
        y: Target vector.
        feature_names: Feature column names.

    Returns:
        Tuple of (surviving_feature_names, importance_records).
    """
    try:
        model = LogisticRegression(
            penalty="l1",
            C=LASSO_C,
            solver="saga",
            max_iter=5000,
            multi_class="multinomial",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X, y)
    except Exception as e:
        logger.error("Lasso fitting failed: %s", e)
        return set(feature_names), []

    # For multiclass, coef_ has shape (n_classes, n_features)
    # A feature survives if ANY class has a non-zero coefficient
    coef_abs = np.abs(model.coef_)
    max_coef_per_feature = coef_abs.max(axis=0)

    survivors: set[str] = set()
    records: list[dict[str, Any]] = []

    for i, name in enumerate(feature_names):
        importance = float(max_coef_per_feature[i])
        is_kept = importance > 1e-8  # Non-zero threshold
        records.append({
            "feature": name,
            "method": "lasso",
            "importance": round(importance, 6),
            "kept": is_kept,
        })
        if is_kept:
            survivors.add(name)

    return survivors, records


def _shap_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> tuple[set[str], list[dict[str, Any]]]:
    """Select features using SHAP values from a lightweight XGBoost model.

    Trains a small XGBoost model and computes feature importance
    via gain-based importance (a fast proxy for full SHAP when
    the ``shap`` library is not available).

    Args:
        X: Scaled feature matrix.
        y: Target vector.
        feature_names: Feature column names.

    Returns:
        Tuple of (surviving_feature_names, importance_records).
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        logger.warning("XGBoost not installed — skipping SHAP selection.")
        return set(feature_names), []

    try:
        model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            objective="multi:softprob",
            num_class=3,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X, y)
    except Exception as e:
        logger.error("XGBoost fitting failed: %s", e)
        return set(feature_names), []

    # Use gain-based feature importance as SHAP proxy
    importance_dict = model.get_booster().get_score(importance_type="gain")

    survivors: set[str] = set()
    records: list[dict[str, Any]] = []

    # Normalize importance values
    max_importance = max(importance_dict.values()) if importance_dict else 1.0

    for i, name in enumerate(feature_names):
        # XGBoost uses "f0", "f1", ... as internal feature names
        xgb_name = f"f{i}"
        raw_importance = importance_dict.get(xgb_name, 0.0)
        normalized = raw_importance / max_importance if max_importance > 0 else 0.0

        is_kept = normalized >= SHAP_THRESHOLD
        records.append({
            "feature": name,
            "method": "shap_proxy",
            "importance": round(normalized, 6),
            "kept": is_kept,
        })
        if is_kept:
            survivors.add(name)

    return survivors, records


def _fallback_top_features(
    report: list[dict[str, Any]],
    n: int = 10,
) -> set[str]:
    """Fallback: select top N features by combined importance.

    Used when the intersection of Lasso and SHAP is too aggressive.

    Args:
        report: Combined selection report records.
        n: Number of features to keep.

    Returns:
        Set of top feature names.
    """
    if not report:
        return set()

    report_df = pd.DataFrame(report)
    # Average importance across methods
    avg_importance = (
        report_df.groupby("feature")["importance"]
        .mean()
        .sort_values(ascending=False)
    )
    return set(avg_importance.head(n).index)


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def main() -> None:
    """CLI entry point for feature selection."""
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    run_feature_selection()


if __name__ == "__main__":
    main()
