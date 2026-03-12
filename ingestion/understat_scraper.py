"""Understat scraper — xG, xGA, and advanced shooting statistics.

Understat serves data as JSON embedded in ``<script>`` tags, making it
more reliable to parse than raw HTML tables. This is our primary source
for expected goals data after FBRef lost Opta access in January 2026.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import pandas as pd

from ingestion.base_scraper import BaseScraper

logger = logging.getLogger(__name__)

# Understat uses a different season format: the start year only
# e.g. https://understat.com/league/EPL/2024
UNDERSTAT_LEAGUE = "EPL"


class UnderstatScraper(BaseScraper):
    """Scrape xG data from Understat for Premier League matches.

    Understat embeds match data as JSON inside ``<script>`` tags
    using the ``datesData`` variable. This scraper extracts and
    parses that JSON to get per-match xG statistics.

    Example::

        scraper = UnderstatScraper()
        df = scraper.run(season="2024")
    """

    def __init__(
        self,
        delay_min: float = 2.0,
        delay_max: float = 5.0,
        cache_ttl: int = 86400,
    ) -> None:
        super().__init__(
            source_name="understat",
            base_url="https://understat.com",
            delay_min=delay_min,
            delay_max=delay_max,
            cache_ttl=cache_ttl,
        )

    def scrape(self, season: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Fetch match-level xG data from Understat.

        Args:
            season: Start year of the season (e.g. ``"2024"`` for 2024-25).

        Returns:
            List of match dicts with xG data from Understat's embedded JSON.
        """
        url = f"{self.base_url}/league/{UNDERSTAT_LEAGUE}/{season}"
        html = self.fetch_page(url)

        # Understat stores match data in a JS variable called "datesData"
        # Format: var datesData = JSON.parse('<escaped_json>')
        match = re.search(
            r"var\s+datesData\s*=\s*JSON\.parse\('(.+?)'\)",
            html,
        )

        if match is None:
            logger.error(
                "[understat] Could not find datesData JSON for season %s",
                season,
            )
            return []

        # The JSON string is escaped — unescape it
        json_str = match.group(1)
        json_str = json_str.encode("utf-8").decode("unicode_escape")

        try:
            dates_data: list[Any] = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error("[understat] Failed to parse JSON: %s", e)
            return []

        # Flatten the nested structure: each date contains multiple matches
        records: list[dict[str, Any]] = []
        for date_group in dates_data:
            if isinstance(date_group, list):
                for match_entry in date_group:
                    if isinstance(match_entry, dict):
                        records.append(match_entry)
            elif isinstance(date_group, dict):
                records.append(date_group)

        logger.info(
            "[understat] Scraped %d match records for season %s",
            len(records),
            season,
        )
        return records

    def parse(self, raw_data: list[dict[str, Any]]) -> pd.DataFrame:
        """Parse raw Understat JSON records into a standardized DataFrame.

        Args:
            raw_data: List of match dicts from :meth:`scrape`.

        Returns:
            DataFrame with columns: ``date``, ``home_team``, ``away_team``,
            ``home_xg``, ``away_xg``, ``home_goals``, ``away_goals``,
            ``is_result`` (whether the match has been played).
        """
        if not raw_data:
            return pd.DataFrame()

        parsed_rows: list[dict[str, Any]] = []

        for record in raw_data:
            # Understat nests team info under "h" (home) and "a" (away)
            home_info = record.get("h", {})
            away_info = record.get("a", {})

            # Extract team names
            home_name = (
                home_info.get("title", "")
                if isinstance(home_info, dict)
                else ""
            )
            away_name = (
                away_info.get("title", "")
                if isinstance(away_info, dict)
                else ""
            )

            # Extract goals
            home_goals = self._safe_int(
                record.get("goals", {}).get("h")
                if isinstance(record.get("goals"), dict)
                else home_info.get("goals")
                if isinstance(home_info, dict)
                else None
            )
            away_goals = self._safe_int(
                record.get("goals", {}).get("a")
                if isinstance(record.get("goals"), dict)
                else away_info.get("goals")
                if isinstance(away_info, dict)
                else None
            )

            # Extract xG
            home_xg = self._safe_float(
                record.get("xG", {}).get("h")
                if isinstance(record.get("xG"), dict)
                else home_info.get("xG")
                if isinstance(home_info, dict)
                else None
            )
            away_xg = self._safe_float(
                record.get("xG", {}).get("a")
                if isinstance(record.get("xG"), dict)
                else away_info.get("xG")
                if isinstance(away_info, dict)
                else None
            )

            is_result = record.get("isResult", False)

            parsed_rows.append({
                "date": record.get("datetime", ""),
                "home_team": self.normalize_team_name(home_name),
                "away_team": self.normalize_team_name(away_name),
                "home_goals": home_goals,
                "away_goals": away_goals,
                "home_xg": home_xg,
                "away_xg": away_xg,
                "is_result": is_result,
                "understat_match_id": record.get("id", ""),
            })

        result = pd.DataFrame(parsed_rows)
        result["date"] = pd.to_datetime(result["date"], errors="coerce")

        # Only keep completed matches
        result = result[result["is_result"] == True].copy()  # noqa: E712
        result = result.dropna(subset=["date"])

        logger.info("[understat] Parsed %d completed match records", len(result))
        return result

    # ------------------------------------------------------------------
    # Team-level aggregation (for season-level stats)
    # ------------------------------------------------------------------

    def scrape_team_season_stats(self, season: str) -> pd.DataFrame:
        """Scrape team-level aggregated xG stats for the full season.

        This provides season totals for xG, xGA, xPts which are used
        for the ``xpts_luck_factor`` feature.

        Args:
            season: Start year (e.g. ``"2024"``).

        Returns:
            DataFrame with per-team season aggregates.
        """
        url = f"{self.base_url}/league/{UNDERSTAT_LEAGUE}/{season}"
        html = self.fetch_page(url, use_cache=True)

        # Team-level data is in "teamsData"
        match = re.search(
            r"var\s+teamsData\s*=\s*JSON\.parse\('(.+?)'\)",
            html,
        )

        if match is None:
            logger.warning(
                "[understat] Could not find teamsData for season %s", season
            )
            return pd.DataFrame()

        json_str = match.group(1)
        json_str = json_str.encode("utf-8").decode("unicode_escape")

        try:
            teams_data: dict[str, Any] = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error("[understat] Failed to parse teamsData: %s", e)
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for team_id, team_info in teams_data.items():
            if not isinstance(team_info, dict):
                continue
            title = team_info.get("title", f"Team_{team_id}")

            # Aggregate match-level data
            history = team_info.get("history", [])
            if not history:
                continue

            total_xg = sum(self._safe_float(m.get("xG", 0)) for m in history)
            total_xga = sum(self._safe_float(m.get("xGA", 0)) for m in history)
            total_pts = sum(self._safe_int(m.get("pts", 0)) for m in history)
            total_xpts = sum(self._safe_float(m.get("xpts", 0)) for m in history)
            matches_played = len(history)

            rows.append({
                "team": self.normalize_team_name(title),
                "matches_played": matches_played,
                "xg_total": round(total_xg, 2),
                "xga_total": round(total_xga, 2),
                "pts_total": total_pts,
                "xpts_total": round(total_xpts, 2),
                "xpts_luck_factor": round(total_pts - total_xpts, 2),
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_float(value: Any) -> float:
        """Safely convert a value to float, defaulting to 0.0."""
        if value is None:
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _safe_int(value: Any) -> int:
        """Safely convert a value to int, defaulting to 0."""
        if value is None:
            return 0
        try:
            return int(value)
        except (ValueError, TypeError):
            return 0
