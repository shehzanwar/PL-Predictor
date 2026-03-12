"""xG features — expected goals differentials and overperformance.

xGA (Expected Goals Against) is central to defensive form calculation
as requested. Combines offensive xG and defensive xGA into composite
threat/vulnerability metrics using EMA smoothing.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EMA_SPAN: int = 5


def compute_xg_features(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Compute xG-based features for every match.

    Features center on the xG *differential* (attacking quality minus
    defensive vulnerability) and *overperformance* (actual vs expected).

    All features use EMA smoothing and are shift(1)-ed to prevent leakage.

    Args:
        matches_df: Match DataFrame with ``home_xg``, ``away_xg`` columns.

    Returns:
        DataFrame with xG feature columns appended.
    """
    df = matches_df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # Skip if no xG data at all
    if "home_xg" not in df.columns or df["home_xg"].isna().all():
        logger.warning("No xG data available — skipping xG features.")
        return df

    # Build per-team xG records
    team_xg = _build_team_xg_records(df)

    # Compute EMA features per team
    team_xg_features = _compute_team_xg_ema(team_xg)

    # Merge home team xG features
    home_xg = team_xg_features.rename(
        columns=lambda c: f"home_{c}" if c not in ("team", "date") else c
    )
    df = df.merge(
        home_xg,
        left_on=["date", "home_team"],
        right_on=["date", "team"],
        how="left",
        suffixes=("", "_xgh"),
    ).drop(columns=["team"], errors="ignore")

    # Merge away team xG features
    away_xg = team_xg_features.rename(
        columns=lambda c: f"away_{c}" if c not in ("team", "date") else c
    )
    df = df.merge(
        away_xg,
        left_on=["date", "away_team"],
        right_on=["date", "team"],
        how="left",
        suffixes=("", "_xga"),
    ).drop(columns=["team"], errors="ignore")

    # Cross-team differentials
    if "home_xg_differential" in df.columns and "away_xg_differential" in df.columns:
        df["xg_diff_advantage"] = df["home_xg_differential"] - df["away_xg_differential"]

    if "home_defensive_ema_xga" in df.columns and "away_defensive_ema_xga" in df.columns:
        # Positive = home defense leaks more xGA (away advantage)
        df["defensive_vulnerability_diff"] = (
            df["home_defensive_ema_xga"] - df["away_defensive_ema_xga"]
        )

    if "home_xg_overperformance" in df.columns and "away_xg_overperformance" in df.columns:
        df["overperformance_diff"] = (
            df["home_xg_overperformance"] - df["away_xg_overperformance"]
        )

    logger.info("Computed xG features: %d columns added", len(df.columns) - len(matches_df.columns))
    return df


def _build_team_xg_records(df: pd.DataFrame) -> pd.DataFrame:
    """Create per-team xG records from match data.

    Args:
        df: Sorted match DataFrame.

    Returns:
        Per-team records with xG/xGA from each team's perspective.
    """
    records: list[dict[str, object]] = []

    for _, row in df.iterrows():
        h_xg = row.get("home_xg", np.nan)
        a_xg = row.get("away_xg", np.nan)
        hg = row.get("home_goals", np.nan)
        ag = row.get("away_goals", np.nan)

        if pd.isna(h_xg) and pd.isna(a_xg):
            continue

        # Home team record
        records.append({
            "team": row["home_team"],
            "date": row["date"],
            "xg_for": float(h_xg) if pd.notna(h_xg) else np.nan,
            "xg_against": float(a_xg) if pd.notna(a_xg) else np.nan,
            "goals_for": float(hg) if pd.notna(hg) else np.nan,
            "goals_against": float(ag) if pd.notna(ag) else np.nan,
        })

        # Away team record
        records.append({
            "team": row["away_team"],
            "date": row["date"],
            "xg_for": float(a_xg) if pd.notna(a_xg) else np.nan,
            "xg_against": float(h_xg) if pd.notna(h_xg) else np.nan,
            "goals_for": float(ag) if pd.notna(ag) else np.nan,
            "goals_against": float(hg) if pd.notna(hg) else np.nan,
        })

    return pd.DataFrame(records).sort_values(["team", "date"]).reset_index(drop=True)


def _compute_team_xg_ema(team_xg: pd.DataFrame) -> pd.DataFrame:
    """Compute EMA-smoothed xG features per team.

    All EMAs are shift(1)-ed to prevent leakage.

    Features:
        - ``ema_xg_for``: Smoothed attacking xG.
        - ``defensive_ema_xga``: Smoothed xGA (central to defensive form).
        - ``xg_differential``: ``ema_xg_for - defensive_ema_xga``.
        - ``xg_overperformance``: Actual goals - expected goals (EMA).
        - ``defensive_overperformance``: Actual goals conceded vs xGA.

    Args:
        team_xg: Per-team xG records.

    Returns:
        DataFrame with EMA features per (team, date).
    """
    result_frames: list[pd.DataFrame] = []

    for team, group in team_xg.groupby("team"):
        g = group.copy().reset_index(drop=True)

        # Offensive xG EMA
        g["ema_xg_for"] = (
            g["xg_for"].ewm(span=EMA_SPAN, min_periods=1).mean().shift(1)
        )

        # Defensive xGA EMA — CENTRAL to defensive form per user requirement
        g["defensive_ema_xga"] = (
            g["xg_against"].ewm(span=EMA_SPAN, min_periods=1).mean().shift(1)
        )

        # xG differential (positive = better overall quality)
        g["xg_differential"] = g["ema_xg_for"] - g["defensive_ema_xga"]

        # Offensive overperformance (actual goals scored vs xG)
        ema_goals = g["goals_for"].ewm(span=EMA_SPAN, min_periods=1).mean().shift(1)
        g["xg_overperformance"] = ema_goals - g["ema_xg_for"]

        # Defensive overperformance (positive = conceding fewer than expected)
        ema_conceded = (
            g["goals_against"].ewm(span=EMA_SPAN, min_periods=1).mean().shift(1)
        )
        g["defensive_overperformance"] = g["defensive_ema_xga"] - ema_conceded

        result_frames.append(g[["team", "date", "ema_xg_for", "defensive_ema_xga",
                                "xg_differential", "xg_overperformance",
                                "defensive_overperformance"]])

    return pd.concat(result_frames, ignore_index=True)
