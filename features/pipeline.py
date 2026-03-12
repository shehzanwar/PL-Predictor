"""Feature pipeline — end-to-end orchestration of all feature modules.

Reads merged match data from ``data/processed/``, runs every feature
module in dependency order, constructs the target variable, and writes
the final model-ready dataset to ``data/features/``.

All features maintain strict temporal integrity — no future data leakage.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from features.form_features import compute_form_features
from features.h2h_features import compute_h2h_features
from features.xg_features import compute_xg_features
from features.schedule_features import compute_schedule_features
from features.squad_features import compute_squad_features

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"

# Output filename
OUTPUT_FILENAME = "pl_feature_matrix.csv"

# Columns that are identifiers, not model features
ID_COLUMNS = [
    "date", "home_team", "away_team", "season", "matchweek",
    "venue", "referee", "attendance", "home_goals", "away_goals",
    "home_xg", "away_xg", "understat_match_id", "is_result",
    "fixture_id", "round", "status", "fetch_timestamp",
]

# Target variable encoding
TARGET_MAP = {"H": 0, "D": 1, "A": 2}


def run_feature_pipeline(
    matches_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Execute the full feature engineering pipeline.

    Steps:
        1. Load merged match data.
        2. Compute form features (EMA-based).
        3. Compute head-to-head features.
        4. Compute xG features.
        5. Compute schedule/congestion features.
        6. Compute squad impact features.
        7. Construct target variable.
        8. Clean and save the model-ready dataset.

    Args:
        matches_path: Path to merged matches CSV. Defaults to
            ``data/processed/pl_matches_merged.csv``.
        output_path: Path to save the feature matrix. Defaults to
            ``data/features/pl_feature_matrix.csv``.

    Returns:
        The final model-ready DataFrame.
    """
    if matches_path is None:
        matches_path = PROCESSED_DIR / "pl_matches_merged.csv"
    if output_path is None:
        FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        output_path = FEATURES_DIR / OUTPUT_FILENAME

    matches_path = Path(matches_path)
    output_path = Path(output_path)

    # Step 1: Load data
    logger.info("=" * 60)
    logger.info("Feature Pipeline — Starting")
    logger.info("=" * 60)

    df = pd.read_csv(matches_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    logger.info("Loaded %d matches from %s", len(df), matches_path)

    # Ensure required columns exist
    for col in ["date", "home_team", "away_team", "home_goals", "away_goals"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}'")

    # Step 2: Form features (EMA)
    logger.info("--- Step 2/7: Form Features (EMA) ---")
    df = compute_form_features(df)
    logger.info("Columns after form: %d", len(df.columns))

    # Step 3: Head-to-head features
    logger.info("--- Step 3/7: Head-to-Head Features ---")
    df = compute_h2h_features(df)
    logger.info("Columns after H2H: %d", len(df.columns))

    # Step 4: xG features
    logger.info("--- Step 4/7: xG Features ---")
    df = compute_xg_features(df)
    logger.info("Columns after xG: %d", len(df.columns))

    # Step 5: Schedule/congestion features
    logger.info("--- Step 5/7: Schedule Features ---")
    df = compute_schedule_features(df)
    logger.info("Columns after schedule: %d", len(df.columns))

    # Step 6: Squad impact features
    logger.info("--- Step 6/7: Squad Features ---")
    df = compute_squad_features(df)
    logger.info("Columns after squad: %d", len(df.columns))

    # Step 7: Target variable
    logger.info("--- Step 7/7: Target Variable ---")
    df = _add_target_variable(df)

    # Clean up
    df = _clean_feature_matrix(df)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # Summary
    feature_cols = [c for c in df.columns if c not in ID_COLUMNS and c != "target"]
    logger.info("=" * 60)
    logger.info("Feature Pipeline — Complete")
    logger.info("Total rows: %d", len(df))
    logger.info("Total feature columns: %d", len(feature_cols))
    logger.info("Target distribution:")
    if "target" in df.columns:
        for label, count in df["target"].value_counts().sort_index().items():
            pct = count / len(df) * 100
            name = {0: "Home Win", 1: "Draw", 2: "Away Win"}.get(label, str(label))
            logger.info("  %s: %d (%.1f%%)", name, count, pct)
    logger.info("Saved to: %s", output_path)
    logger.info("=" * 60)

    return df


def _add_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """Construct the match outcome target variable.

    Encoding: Home Win = 0, Draw = 1, Away Win = 2.

    Args:
        df: Feature DataFrame with ``home_goals`` and ``away_goals``.

    Returns:
        DataFrame with ``target`` column added.
    """
    conditions = [
        df["home_goals"] > df["away_goals"],   # Home Win
        df["home_goals"] == df["away_goals"],   # Draw
        df["home_goals"] < df["away_goals"],    # Away Win
    ]
    choices = [0, 1, 2]
    df["target"] = np.select(conditions, choices, default=np.nan)
    df["target"] = pd.to_numeric(df["target"], errors="coerce")

    # Drop rows without a valid target (unplayed matches)
    pre_drop = len(df)
    df = df.dropna(subset=["target"]).copy()
    df["target"] = df["target"].astype(int)
    logger.info("Target variable: %d valid matches (dropped %d)", len(df), pre_drop - len(df))

    return df


def _clean_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Final cleanup: remove duplicate columns, constant columns, etc.

    Args:
        df: Raw feature DataFrame.

    Returns:
        Cleaned DataFrame ready for feature selection.
    """
    # Remove exact duplicate columns (can happen from multiple merges)
    df = df.loc[:, ~df.columns.duplicated()]

    # Drop columns that are entirely NaN
    all_nan_cols = df.columns[df.isna().all()].tolist()
    if all_nan_cols:
        logger.info("Dropping %d all-NaN columns: %s", len(all_nan_cols), all_nan_cols)
        df = df.drop(columns=all_nan_cols)

    # Drop constant columns (zero variance — useless for ML)
    feature_cols = [c for c in df.columns if c not in ID_COLUMNS and c != "target"]
    constant_cols = [
        c for c in feature_cols
        if df[c].nunique(dropna=True) <= 1
    ]
    if constant_cols:
        logger.info("Dropping %d constant columns: %s", len(constant_cols), constant_cols)
        df = df.drop(columns=constant_cols)

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    return df


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def main() -> None:
    """CLI entry point for the feature pipeline."""
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    run_feature_pipeline()


if __name__ == "__main__":
    main()
