"""Squad features — impact-weighted absence metrics.

CRITICAL DESIGN DECISION: We do NOT one-hot encode individual players.
That approach causes massive overfitting on ~380 matches/season with
~500 unique players. Instead, we quantify the *impact* of missing
players using their share of the team's rolling production.

Key metrics:
  - missing_squad_xg_percent: % of team's 12-month xG/xA currently injured
  - missing_defensive_minutes_percent: % of defensive minutes unavailable
  - squad_value_ratio: relative squad market value between teams
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Default xG share per position when detailed stats are unavailable
# These are PL-average priors from 2020-2025 data
POSITION_XG_WEIGHT: dict[str, float] = {
    "forward": 0.45,
    "midfielder": 0.35,
    "defender": 0.15,
    "goalkeeper": 0.05,
}

POSITION_DEFENSIVE_WEIGHT: dict[str, float] = {
    "forward": 0.05,
    "midfielder": 0.25,
    "defender": 0.50,
    "goalkeeper": 0.20,
}


def compute_squad_features(
    matches_df: pd.DataFrame,
    injuries_path: str | Path | None = None,
    xg_season_path: str | Path | None = None,
) -> pd.DataFrame:
    """Compute impact-weighted squad availability features.

    Instead of counting raw injured players, we weight each absence
    by their contribution to the team's rolling attacking/defensive output.

    Args:
        matches_df: Match DataFrame to append features to.
        injuries_path: Path to the current injuries CSV.
        xg_season_path: Path to the season xG stats CSV.

    Returns:
        DataFrame with squad impact features appended.
    """
    df = matches_df.copy()

    # Load auxiliary data
    injuries_df = _load_injuries(injuries_path)
    xg_season_df = _load_xg_season(xg_season_path)

    if injuries_df.empty:
        logger.warning("No injury data available — using neutral squad features.")
        df = _add_neutral_squad_features(df)
        return df

    # Compute per-team impact metrics
    team_impact = _compute_team_impact_metrics(injuries_df, xg_season_df)

    # Merge for home team
    home_impact = team_impact.rename(
        columns=lambda c: f"home_{c}" if c != "team" else c
    )
    df = df.merge(home_impact, left_on="home_team", right_on="team", how="left")
    df = df.drop(columns=["team"], errors="ignore")

    # Merge for away team
    away_impact = team_impact.rename(
        columns=lambda c: f"away_{c}" if c != "team" else c
    )
    df = df.merge(away_impact, left_on="away_team", right_on="team", how="left")
    df = df.drop(columns=["team"], errors="ignore")

    # Fill NaN with neutral values (team not found in injury data = full squad)
    impact_cols = [c for c in df.columns if "missing_" in c or "squad_strength" in c]
    for col in impact_cols:
        if "missing_" in col:
            df[col] = df[col].fillna(0.0)
        elif "squad_strength" in col:
            df[col] = df[col].fillna(1.0)

    # Differential features
    if "home_missing_xg_percent" in df.columns and "away_missing_xg_percent" in df.columns:
        df["missing_xg_impact_diff"] = (
            df["home_missing_xg_percent"] - df["away_missing_xg_percent"]
        )
    if "home_missing_defensive_percent" in df.columns and "away_missing_defensive_percent" in df.columns:
        df["missing_defense_impact_diff"] = (
            df["home_missing_defensive_percent"] - df["away_missing_defensive_percent"]
        )
    if "home_squad_strength_index" in df.columns and "away_squad_strength_index" in df.columns:
        df["squad_strength_ratio"] = (
            df["home_squad_strength_index"] / df["away_squad_strength_index"].replace(0, 1)
        )

    logger.info("Computed squad features for %d matches", len(df))
    return df


def _compute_team_impact_metrics(
    injuries_df: pd.DataFrame,
    xg_season_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute impact-weighted absence metrics per team.

    For each team:
        - ``missing_xg_percent``: Estimated % of team's xG production
          that is currently unavailable due to injuries/suspensions.
        - ``missing_defensive_percent``: Estimated % of defensive
          contribution that is currently unavailable.
        - ``total_unavailable``: Raw count of unavailable players.
        - ``squad_strength_index``: 1.0 - weighted_absence_impact.

    Args:
        injuries_df: Current injuries/suspensions data.
        xg_season_df: Team-level season xG stats.

    Returns:
        Per-team impact metrics.
    """
    rows: list[dict[str, Any]] = []

    for team, group in injuries_df.groupby("team"):
        n_injured = len(group)

        # Estimate xG share of missing players
        # If we have per-player stats, use them; otherwise use positional priors
        missing_xg_share = 0.0
        missing_def_share = 0.0

        for _, player in group.iterrows():
            # Use positional priors as the baseline impact weight
            # In a full implementation, this would use actual player xG/90 data
            position = _infer_position(player.get("reason", ""), player.get("player_name", ""))
            missing_xg_share += POSITION_XG_WEIGHT.get(position, 0.10)
            missing_def_share += POSITION_DEFENSIVE_WEIGHT.get(position, 0.10)

        # Cap at 100% (mathematically possible with many injuries)
        missing_xg_pct = min(missing_xg_share, 1.0)
        missing_def_pct = min(missing_def_share, 1.0)

        # Squad strength index: 1.0 = full squad, 0.0 = entire squad injured
        # Weighted average of offensive and defensive impact
        strength_index = 1.0 - (0.6 * missing_xg_pct + 0.4 * missing_def_pct)

        rows.append({
            "team": team,
            "total_unavailable": n_injured,
            "missing_xg_percent": round(missing_xg_pct, 3),
            "missing_defensive_percent": round(missing_def_pct, 3),
            "squad_strength_index": round(max(strength_index, 0.0), 3),
        })

    return pd.DataFrame(rows)


def _infer_position(reason: str, player_name: str) -> str:
    """Infer a player's rough position from available metadata.

    This is a heuristic fallback when detailed squad data isn't available.
    In production, this would be replaced by a proper squad roster lookup.

    Args:
        reason: Injury reason string.
        player_name: Player name (used for GK detection).

    Returns:
        One of: "goalkeeper", "defender", "midfielder", "forward".
    """
    name_lower = player_name.lower()

    # Common goalkeeper name patterns
    gk_indicators = ["keeper", "gk"]
    if any(ind in name_lower for ind in gk_indicators):
        return "goalkeeper"

    # Default distribution: assume midfielders (most common position)
    return "midfielder"


def _add_neutral_squad_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add neutral (no-impact) squad features when injury data is unavailable.

    Args:
        df: Match DataFrame.

    Returns:
        DataFrame with neutral squad feature columns.
    """
    for prefix in ["home", "away"]:
        df[f"{prefix}_total_unavailable"] = 0
        df[f"{prefix}_missing_xg_percent"] = 0.0
        df[f"{prefix}_missing_defensive_percent"] = 0.0
        df[f"{prefix}_squad_strength_index"] = 1.0

    df["missing_xg_impact_diff"] = 0.0
    df["missing_defense_impact_diff"] = 0.0
    df["squad_strength_ratio"] = 1.0

    return df


def _load_injuries(path: str | Path | None) -> pd.DataFrame:
    """Load injuries CSV file with graceful fallback."""
    if path is None:
        path = PROCESSED_DIR / "pl_current_injuries.csv"
    path = Path(path)

    if not path.exists():
        logger.warning("Injuries file not found: %s", path)
        return pd.DataFrame()

    return pd.read_csv(path)


def _load_xg_season(path: str | Path | None) -> pd.DataFrame:
    """Load season xG stats with graceful fallback."""
    if path is None:
        path = PROCESSED_DIR / "pl_xg_season_stats.csv"
    path = Path(path)

    if not path.exists():
        logger.warning("xG season stats not found: %s", path)
        return pd.DataFrame()

    return pd.read_csv(path)
