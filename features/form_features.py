"""Form features — EMA-based team form and rolling attacking threat.

Uses Exponential Moving Averages instead of raw rolling windows
to smooth out noise while still being responsive to recent results.
EMA gives more weight to recent matches, naturally decaying the
influence of older games — ideal for detecting form swings.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# EMA span parameters (in matches, not days)
# span=5 → ~63% weight on last 5 matches
# span=10 → ~63% weight on last 10 matches
EMA_SHORT_SPAN: int = 5
EMA_LONG_SPAN: int = 10


def compute_form_features(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Compute EMA-based form features for every match in the dataset.

    For each match row, features are computed from that team's perspective
    using only data available *before* kickoff (strict temporal ordering).

    Args:
        matches_df: Merged match dataset with columns ``date``,
            ``home_team``, ``away_team``, ``home_goals``, ``away_goals``,
            ``home_xg``, ``away_xg``.

    Returns:
        DataFrame with original columns plus form features for both
        the home and away team.
    """
    df = matches_df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # Build per-team match history (one row per team per match)
    team_records = _build_team_match_records(df)

    # Compute EMA features per team
    team_form = _compute_team_ema_features(team_records)

    # Merge back: home team features
    home_features = team_form.rename(
        columns=lambda c: f"home_{c}" if c not in ("team", "date", "match_idx") else c
    )
    df = df.merge(
        home_features,
        left_on=["date", "home_team"],
        right_on=["date", "team"],
        how="left",
        suffixes=("", "_hf"),
    ).drop(columns=["team"], errors="ignore")

    # Merge back: away team features
    away_features = team_form.rename(
        columns=lambda c: f"away_{c}" if c not in ("team", "date", "match_idx") else c
    )
    df = df.merge(
        away_features,
        left_on=["date", "away_team"],
        right_on=["date", "team"],
        how="left",
        suffixes=("", "_af"),
    ).drop(columns=["team"], errors="ignore")

    # Differential features
    for stat in ["ema_points_short", "ema_points_long", "ema_goals_scored",
                 "ema_goals_conceded", "ema_xg", "ema_xga",
                 "rolling_attacking_threat"]:
        home_col = f"home_{stat}"
        away_col = f"away_{stat}"
        if home_col in df.columns and away_col in df.columns:
            df[f"diff_{stat}"] = df[home_col] - df[away_col]

    logger.info("Computed form features: %d rows, %d columns", len(df), len(df.columns))
    return df


def _build_team_match_records(df: pd.DataFrame) -> pd.DataFrame:
    """Unstack matches into per-team records (one row per team per match).

    Each match produces two rows: one from the home team's perspective
    and one from the away team's perspective.

    Args:
        df: Sorted match DataFrame.

    Returns:
        DataFrame with columns: ``team``, ``date``, ``match_idx``,
        ``goals_scored``, ``goals_conceded``, ``xg``, ``xga``,
        ``points``, ``is_home``, ``result``.
    """
    records: list[dict[str, Any]] = []

    for idx, row in df.iterrows():
        hg = row.get("home_goals", np.nan)
        ag = row.get("away_goals", np.nan)
        h_xg = row.get("home_xg", np.nan)
        a_xg = row.get("away_xg", np.nan)
        date = row["date"]

        # Skip matches with no score
        if pd.isna(hg) or pd.isna(ag):
            continue

        hg, ag = int(hg), int(ag)

        # Home team record
        h_points = 3 if hg > ag else (1 if hg == ag else 0)
        h_result = "W" if hg > ag else ("D" if hg == ag else "L")
        records.append({
            "team": row["home_team"],
            "date": date,
            "match_idx": idx,
            "goals_scored": hg,
            "goals_conceded": ag,
            "xg": h_xg if not pd.isna(h_xg) else np.nan,
            "xga": a_xg if not pd.isna(a_xg) else np.nan,
            "points": h_points,
            "is_home": 1,
            "result": h_result,
        })

        # Away team record
        a_points = 3 if ag > hg else (1 if ag == hg else 0)
        a_result = "W" if ag > hg else ("D" if ag == hg else "L")
        records.append({
            "team": row["away_team"],
            "date": date,
            "match_idx": idx,
            "goals_scored": ag,
            "goals_conceded": hg,
            "xg": a_xg if not pd.isna(a_xg) else np.nan,
            "xga": h_xg if not pd.isna(h_xg) else np.nan,
            "points": a_points,
            "is_home": 0,
            "result": a_result,
        })

    return pd.DataFrame(records).sort_values(["team", "date"]).reset_index(drop=True)


def _compute_team_ema_features(team_records: pd.DataFrame) -> pd.DataFrame:
    """Compute EMA-smoothed features per team using only past data.

    All EMA values are **shifted by 1** within each team group so that
    the feature for match N uses only matches 1..N-1 (no leakage).

    Args:
        team_records: Per-team match records from :func:`_build_team_match_records`.

    Returns:
        DataFrame with EMA features aligned to each (team, date) pair.
    """
    result_frames: list[pd.DataFrame] = []

    for team, group in team_records.groupby("team"):
        g = group.copy().reset_index(drop=True)

        # EMA of points (form indicator)
        g["ema_points_short"] = (
            g["points"].ewm(span=EMA_SHORT_SPAN, min_periods=1).mean().shift(1)
        )
        g["ema_points_long"] = (
            g["points"].ewm(span=EMA_LONG_SPAN, min_periods=1).mean().shift(1)
        )

        # EMA of goals scored / conceded
        g["ema_goals_scored"] = (
            g["goals_scored"].ewm(span=EMA_SHORT_SPAN, min_periods=1).mean().shift(1)
        )
        g["ema_goals_conceded"] = (
            g["goals_conceded"].ewm(span=EMA_SHORT_SPAN, min_periods=1).mean().shift(1)
        )

        # EMA of xG and xGA (where available)
        if g["xg"].notna().sum() > 0:
            g["ema_xg"] = (
                g["xg"].ewm(span=EMA_SHORT_SPAN, min_periods=1).mean().shift(1)
            )
            g["ema_xga"] = (
                g["xga"].ewm(span=EMA_SHORT_SPAN, min_periods=1).mean().shift(1)
            )
        else:
            g["ema_xg"] = np.nan
            g["ema_xga"] = np.nan

        # Rolling attacking threat: composite of goals + xG
        # Captures both actual output and underlying chance creation
        g["rolling_attacking_threat"] = (
            (g["goals_scored"] * 0.4 + g["xg"].fillna(g["goals_scored"]) * 0.6)
            .ewm(span=EMA_SHORT_SPAN, min_periods=1)
            .mean()
            .shift(1)
        )

        # Streak features
        g["current_streak_length"] = _compute_streak_length(g["result"].tolist())
        g["is_on_winning_streak"] = (
            (g["result"].shift(1) == "W").astype(int)
            * g["current_streak_length"]
        ).clip(upper=0).where(g["result"].shift(1) == "W", 0)

        # Unbeaten run length
        g["unbeaten_run"] = _compute_unbeaten_run(g["result"].tolist())

        result_frames.append(g)

    return pd.concat(result_frames, ignore_index=True)


def _compute_streak_length(results: list[str]) -> list[int]:
    """Compute the current streak length at each match (using only past data).

    The streak at match N reflects the consecutive same-result count
    from matches 1..N-1. Match 0 has streak 0.

    Args:
        results: Ordered list of result strings ("W", "D", "L").

    Returns:
        List of streak lengths, same length as input.
    """
    streaks: list[int] = [0]  # No streak before first match
    for i in range(1, len(results)):
        count = 0
        prev_result = results[i - 1]
        for j in range(i - 1, -1, -1):
            if results[j] == prev_result:
                count += 1
            else:
                break
        streaks.append(count)
    return streaks


def _compute_unbeaten_run(results: list[str]) -> list[int]:
    """Compute unbeaten run length (wins + draws) at each match.

    Args:
        results: Ordered list of result strings.

    Returns:
        List of unbeaten run lengths, same length as input.
    """
    runs: list[int] = [0]
    for i in range(1, len(results)):
        count = 0
        for j in range(i - 1, -1, -1):
            if results[j] in ("W", "D"):
                count += 1
            else:
                break
        runs.append(count)
    return runs
