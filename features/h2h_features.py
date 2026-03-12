"""Head-to-head features — historical matchup analysis.

Computes historical performance between two specific teams,
using only matches that occurred *before* the match being predicted.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

H2H_LOOKBACK: int = 10  # Max prior H2H meetings to consider


def compute_h2h_features(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Add head-to-head features to every match row.

    For each match (home vs away), looks back at the last N meetings
    between these two specific teams (regardless of who was home/away)
    and computes win rates, goal averages, and recency.

    Args:
        matches_df: Match DataFrame sorted by date.

    Returns:
        DataFrame with H2H feature columns appended.
    """
    df = matches_df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    h2h_features: list[dict[str, Any]] = []

    for idx, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        match_date = row["date"]

        # Get all prior meetings between these two teams
        prior = df[
            (df["date"] < match_date)
            & (
                ((df["home_team"] == home) & (df["away_team"] == away))
                | ((df["home_team"] == away) & (df["away_team"] == home))
            )
        ].tail(H2H_LOOKBACK)

        features = _compute_h2h_stats(prior, home, away)
        features["_idx"] = idx
        h2h_features.append(features)

    h2h_df = pd.DataFrame(h2h_features).set_index("_idx")
    result = pd.concat([df, h2h_df], axis=1)

    logger.info("Computed H2H features for %d matches", len(result))
    return result


def _compute_h2h_stats(
    prior_meetings: pd.DataFrame,
    home_team: str,
    away_team: str,
) -> dict[str, Any]:
    """Compute H2H statistics from prior meetings.

    Args:
        prior_meetings: Past matches between these two teams.
        home_team: The team playing at home in the *current* match.
        away_team: The team playing away in the *current* match.

    Returns:
        Dict of H2H feature values.
    """
    n_meetings = len(prior_meetings)

    if n_meetings == 0:
        return {
            "h2h_matches_played": 0,
            "h2h_home_team_win_pct": 0.5,   # Prior: assume even
            "h2h_away_team_win_pct": 0.5,
            "h2h_draw_pct": 0.25,
            "h2h_avg_total_goals": 2.5,      # League average prior
            "h2h_home_team_avg_goals": 1.25,
            "h2h_away_team_avg_goals": 1.25,
            "h2h_days_since_last": np.nan,
        }

    home_wins = 0
    away_wins = 0
    draws = 0
    home_team_goals: list[int] = []
    away_team_goals: list[int] = []

    for _, match in prior_meetings.iterrows():
        hg = int(match["home_goals"])
        ag = int(match["away_goals"])

        # Determine result from the perspective of `home_team` in the current match
        if match["home_team"] == home_team:
            home_team_goals.append(hg)
            away_team_goals.append(ag)
            if hg > ag:
                home_wins += 1
            elif hg < ag:
                away_wins += 1
            else:
                draws += 1
        else:
            # home_team was the away side in this historical match
            home_team_goals.append(ag)
            away_team_goals.append(hg)
            if ag > hg:
                home_wins += 1
            elif ag < hg:
                away_wins += 1
            else:
                draws += 1

    last_meeting_date = prior_meetings["date"].max()
    current_date = prior_meetings["date"].max()  # Placeholder — overridden by caller context

    return {
        "h2h_matches_played": n_meetings,
        "h2h_home_team_win_pct": round(home_wins / n_meetings, 3),
        "h2h_away_team_win_pct": round(away_wins / n_meetings, 3),
        "h2h_draw_pct": round(draws / n_meetings, 3),
        "h2h_avg_total_goals": round(
            (sum(home_team_goals) + sum(away_team_goals)) / n_meetings, 2
        ),
        "h2h_home_team_avg_goals": round(np.mean(home_team_goals), 2),
        "h2h_away_team_avg_goals": round(np.mean(away_team_goals), 2),
        "h2h_days_since_last": (
            pd.Timestamp.now() - pd.Timestamp(last_meeting_date)
        ).days if pd.notna(last_meeting_date) else np.nan,
    }
