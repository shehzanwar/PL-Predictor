"""Schedule features — non-linear rest and congestion analysis.

Analytics show that normal rest variances (3 vs 4 days) barely impact
win probability. Instead of a linear rest_days feature, we create
binary flags for EXTREME conditions:
  - extreme_congestion: > 2 matches in 7 days
  - heavy_european_travel: played a European away match mid-week

This captures the non-linear relationship between schedule density
and performance that linear features miss.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Thresholds (derived from PL performance research)
EXTREME_CONGESTION_MATCHES: int = 3        # >= 3 matches in 7 days
CONGESTION_WINDOW_DAYS: int = 7
HEAVY_CONGESTION_14D_MATCHES: int = 5      # >= 5 matches in 14 days
MIDWEEK_TURNAROUND_HOURS: int = 72         # < 72h between matches


def compute_schedule_features(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Compute non-linear schedule and congestion features.

    All features are computed using only past match dates for each team.
    Binary flags for extreme conditions replace linear rest-day features.

    Args:
        matches_df: Match DataFrame with ``date`` column, sorted by date.

    Returns:
        DataFrame with schedule feature columns appended.
    """
    df = matches_df.copy()
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    # Build per-team schedule records
    schedule_data = _build_team_schedules(df)

    # Compute features per team
    team_features = _compute_congestion_features(schedule_data)

    # Merge home team schedule features
    home_sched = team_features.rename(
        columns=lambda c: f"home_{c}" if c not in ("team", "date") else c
    )
    df = df.merge(
        home_sched,
        left_on=["date", "home_team"],
        right_on=["date", "team"],
        how="left",
    ).drop(columns=["team"], errors="ignore")

    # Merge away team schedule features
    away_sched = team_features.rename(
        columns=lambda c: f"away_{c}" if c not in ("team", "date") else c
    )
    df = df.merge(
        away_sched,
        left_on=["date", "away_team"],
        right_on=["date", "team"],
        how="left",
        suffixes=("", "_as"),
    ).drop(columns=["team"], errors="ignore")

    # Relative congestion advantage (positive = home has MORE congestion)
    if "home_extreme_congestion" in df.columns and "away_extreme_congestion" in df.columns:
        df["congestion_disadvantage_diff"] = (
            df["home_extreme_congestion"].astype(int)
            - df["away_extreme_congestion"].astype(int)
        )
    if "home_heavy_congestion_14d" in df.columns and "away_heavy_congestion_14d" in df.columns:
        df["heavy_congestion_diff"] = (
            df["home_heavy_congestion_14d"].astype(int)
            - df["away_heavy_congestion_14d"].astype(int)
        )

    # Season phase (capturing different dynamics at season start/end)
    df["season_phase"] = _compute_season_phase(df)

    # Boxing Day / New Year flag
    df["is_festive_period"] = df["date"].dt.month.eq(12) & df["date"].dt.day.ge(26) | (
        df["date"].dt.month.eq(1) & df["date"].dt.day.le(3)
    )

    logger.info("Computed schedule features for %d matches", len(df))
    return df


def _build_team_schedules(df: pd.DataFrame) -> pd.DataFrame:
    """Create per-team schedule records.

    Args:
        df: Sorted match DataFrame.

    Returns:
        Per-team records with match dates and home/away indicators.
    """
    records: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        records.append({
            "team": row["home_team"],
            "date": row["date"],
            "is_home": True,
        })
        records.append({
            "team": row["away_team"],
            "date": row["date"],
            "is_home": False,
        })

    return pd.DataFrame(records).sort_values(["team", "date"]).reset_index(drop=True)


def _compute_congestion_features(schedule: pd.DataFrame) -> pd.DataFrame:
    """Compute non-linear congestion flags per team per match.

    Binary flags for extreme conditions:
        - ``extreme_congestion``: >= 3 matches in last 7 days
        - ``heavy_congestion_14d``: >= 5 matches in last 14 days
        - ``short_turnaround``: < 72 hours since last match
        - ``is_midweek``: Match on Tue/Wed/Thu (typical European nights)

    Args:
        schedule: Per-team schedule records.

    Returns:
        DataFrame with binary congestion features.
    """
    result_frames: list[pd.DataFrame] = []

    for team, group in schedule.groupby("team"):
        g = group.copy().reset_index(drop=True)
        n = len(g)

        extreme_congestion: list[bool] = []
        heavy_congestion_14d: list[bool] = []
        short_turnaround: list[bool] = []
        days_rest: list[float] = []

        for i in range(n):
            match_date = g.iloc[i]["date"]

            # Count matches in last 7 days (excluding current match)
            window_7d = g[
                (g["date"] < match_date)
                & (g["date"] >= match_date - pd.Timedelta(days=CONGESTION_WINDOW_DAYS))
            ]
            extreme_congestion.append(len(window_7d) >= EXTREME_CONGESTION_MATCHES)

            # Count matches in last 14 days
            window_14d = g[
                (g["date"] < match_date)
                & (g["date"] >= match_date - pd.Timedelta(days=14))
            ]
            heavy_congestion_14d.append(len(window_14d) >= HEAVY_CONGESTION_14D_MATCHES)

            # Days since last match
            prior = g[g["date"] < match_date]
            if len(prior) > 0:
                last_match = prior.iloc[-1]["date"]
                delta_hours = (match_date - last_match).total_seconds() / 3600
                days_rest.append(delta_hours / 24)
                short_turnaround.append(delta_hours < MIDWEEK_TURNAROUND_HOURS)
            else:
                days_rest.append(np.nan)
                short_turnaround.append(False)

        g["extreme_congestion"] = extreme_congestion
        g["heavy_congestion_14d"] = heavy_congestion_14d
        g["short_turnaround"] = short_turnaround
        g["days_rest"] = days_rest  # Kept for reference but not primary feature
        g["is_midweek"] = g["date"].dt.dayofweek.isin([1, 2, 3])  # Tue/Wed/Thu

        result_frames.append(g)

    return pd.concat(result_frames, ignore_index=True)


def _compute_season_phase(df: pd.DataFrame) -> pd.Series:
    """Categorize each match into a season phase.

    Phases:
        - ``early``: Aug-Oct (first ~10 matchweeks)
        - ``mid``: Nov-Jan (winter congestion period)
        - ``late``: Feb-Mar (run-in begins)
        - ``final``: Apr-May (final ~8 matchweeks, highest stakes)

    Args:
        df: DataFrame with ``date`` column.

    Returns:
        Series of season phase labels.
    """
    month = df["date"].dt.month
    return pd.cut(
        month,
        bins=[0, 3, 5, 7, 10, 12],
        labels=["late", "final", "preseason", "early", "mid"],
        ordered=False,
    ).astype(str).replace("preseason", "early")
