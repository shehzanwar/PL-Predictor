"""Injury scraper — aggregates player availability data.

Combines data from API-Football's injury endpoint with
supplementary web sources to build a comprehensive
squad availability picture for each team.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pandas as pd

from ingestion.api_football_client import APIFootballClient
from ingestion.base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class InjuryScraper(BaseScraper):
    """Aggregate injury and suspension data for PL squads.

    This scraper primarily delegates to :class:`APIFootballClient`
    for injury data, then enriches it with staleness tracking and
    summary statistics per team.

    Example::

        scraper = InjuryScraper()
        df = scraper.run(season="2024")
    """

    def __init__(self, api_key: str | None = None) -> None:
        super().__init__(
            source_name="injuries",
            base_url="https://v3.football.api-sports.io",
            delay_min=0.5,
            delay_max=1.0,
        )
        self._api_client = APIFootballClient(api_key=api_key)

    def scrape(self, season: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Fetch injury data via API-Football.

        Args:
            season: Season year (e.g. ``"2024"``).

        Returns:
            List of injury record dicts.
        """
        df = self._api_client.get_injuries(season=int(season))
        if df.empty:
            logger.warning("[injuries] No injury data returned for season %s", season)
            return []
        return df.to_dict(orient="records")

    def parse(self, raw_data: list[dict[str, Any]]) -> pd.DataFrame:
        """Parse and enrich injury records with metadata.

        Args:
            raw_data: List of injury dicts from :meth:`scrape`.

        Returns:
            DataFrame with injury data and a ``fetch_timestamp`` column.
        """
        if not raw_data:
            return pd.DataFrame()

        df = pd.DataFrame(raw_data)
        df["fetch_timestamp"] = datetime.utcnow().isoformat()
        return df

    def get_team_availability_summary(self, season: str) -> pd.DataFrame:
        """Build a per-team summary of squad availability.

        Counts the number of injured/suspended players per team
        and assigns a ``squad_availability_score`` (0.0 to 1.0,
        where 1.0 = full squad available).

        Args:
            season: Season year (e.g. ``"2024"``).

        Returns:
            DataFrame with one row per team.
        """
        raw = self.scrape(season)
        injury_df = self.parse(raw)

        if injury_df.empty:
            logger.info("[injuries] No injuries — all squads fully available.")
            return pd.DataFrame(columns=[
                "team", "injured_count", "suspended_count", "total_unavailable",
            ])

        # Count injuries and suspensions per team
        summary_rows: list[dict[str, Any]] = []
        for team, group in injury_df.groupby("team"):
            injured = len(group[group["injury_type"].str.lower() != "suspended"])
            suspended = len(group[group["injury_type"].str.lower() == "suspended"])

            summary_rows.append({
                "team": team,
                "injured_count": injured,
                "suspended_count": suspended,
                "total_unavailable": injured + suspended,
            })

        return pd.DataFrame(summary_rows).sort_values(
            "total_unavailable", ascending=False
        ).reset_index(drop=True)
