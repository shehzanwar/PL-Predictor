"""Understat / xG data scraper — pivoted to API-Football.

IMPORTANT: As of early 2026, Understat renders data entirely via
client-side JavaScript (league.min.js). The inline ``datesData``
JSON variable that older scrapers relied on no longer exists in the
page source — standard HTTP requests return only an 18KB HTML shell.

This module now uses **API-Football** as the primary xG data source.
The API-Football ``/fixtures`` endpoint with the ``statistics``
parameter provides per-match xG data for all PL matches.

If API-Football is unavailable, the module degrades gracefully
by returning match data without xG columns.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
import pandas as pd
from diskcache import Cache

from ingestion.base_scraper import BaseScraper, PROJECT_ROOT

logger = logging.getLogger(__name__)

PREMIER_LEAGUE_ID = 39


class UnderstatScraper(BaseScraper):
    """Fetch per-match xG data from API-Football.

    Despite the class name (preserved for backward compatibility with
    the orchestrator), this now wraps the API-Football ``/fixtures``
    endpoint to retrieve xG statistics per match.

    The API key is loaded from the ``API_FOOTBALL_KEY`` environment
    variable (supports ``.env`` files via ``python-dotenv``).

    Example::

        scraper = UnderstatScraper()
        df = scraper.run(season="2024")
    """

    def __init__(
        self,
        api_key: str | None = None,
        delay_min: float = 0.5,
        delay_max: float = 1.0,
        cache_ttl: int = 86400,
    ) -> None:
        super().__init__(
            source_name="understat",
            base_url="https://v3.football.api-sports.io",
            delay_min=delay_min,
            delay_max=delay_max,
            cache_ttl=cache_ttl,
        )
        self.api_key = api_key or os.environ.get("API_FOOTBALL_KEY", "")
        if not self.api_key:
            logger.warning(
                "[understat/api-football] No API key found. "
                "Set API_FOOTBALL_KEY in .env file. xG data will be unavailable."
            )

        self._http_client = httpx.Client(
            base_url=self.base_url,
            headers={
                "x-apisports-key": self.api_key,
                "Accept": "application/json",
            },
            timeout=30.0,
        )
        self._api_cache: Cache = Cache(
            str(PROJECT_ROOT / "data" / "cache" / "understat_apifb")
        )

    def scrape(self, season: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Fetch fixtures with statistics from API-Football.

        Args:
            season: Season start year (e.g. ``"2024"`` for 2024-25).

        Returns:
            List of fixture dicts including xG from statistics.
        """
        if not self.api_key:
            logger.warning("[understat] No API key — returning empty data.")
            return []

        cache_key = f"fixtures_xg:{season}"
        cached = self._api_cache.get(cache_key)
        if cached is not None and isinstance(cached, list):
            logger.info("[understat] Cache hit for season %s (%d fixtures)", season, len(cached))
            return cached

        self._rate_limit()

        try:
            response = self._http_client.get(
                "/fixtures",
                params={
                    "league": PREMIER_LEAGUE_ID,
                    "season": int(season),
                    "status": "FT",  # Finished matches only
                },
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error("[understat] API request failed: %s", e)
            return []

        errors = data.get("errors", {})
        if errors:
            logger.error("[understat] API errors: %s", errors)
            return []

        results: list[dict[str, Any]] = data.get("response", [])

        # Now fetch statistics for each fixture to get xG
        # API-Football includes xG in fixture statistics
        enriched: list[dict[str, Any]] = []
        for fixture in results:
            fixture_id = fixture.get("fixture", {}).get("id")
            if fixture_id:
                stats = self._fetch_fixture_stats(fixture_id)
                fixture["_statistics"] = stats
            enriched.append(fixture)

        self._api_cache.set(cache_key, enriched, expire=self.cache_ttl)
        logger.info("[understat] Fetched %d fixtures for season %s", len(enriched), season)
        return enriched

    def parse(self, raw_data: list[dict[str, Any]]) -> pd.DataFrame:
        """Parse API-Football fixture data into a DataFrame with xG columns.

        Args:
            raw_data: List of fixture dicts from :meth:`scrape`.

        Returns:
            DataFrame with ``date``, ``home_team``, ``away_team``,
            ``home_goals``, ``away_goals``, ``home_xg``, ``away_xg``.
        """
        if not raw_data:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for fixture in raw_data:
            fixture_info = fixture.get("fixture", {})
            teams_info = fixture.get("teams", {})
            goals_info = fixture.get("goals", {})
            stats = fixture.get("_statistics", [])

            home_name = teams_info.get("home", {}).get("name", "")
            away_name = teams_info.get("away", {}).get("name", "")

            # Extract xG from statistics
            home_xg, away_xg = self._extract_xg(stats, teams_info)

            rows.append({
                "date": fixture_info.get("date", ""),
                "home_team": self.normalize_team_name(home_name),
                "away_team": self.normalize_team_name(away_name),
                "home_goals": goals_info.get("home"),
                "away_goals": goals_info.get("away"),
                "home_xg": home_xg,
                "away_xg": away_xg,
                "is_result": True,
                "understat_match_id": fixture_info.get("id", ""),
            })

        result = pd.DataFrame(rows)
        result["date"] = pd.to_datetime(result["date"], errors="coerce", utc=True)
        # Normalize to timezone-naive for merging
        result["date"] = result["date"].dt.tz_localize(None)

        logger.info("[understat] Parsed %d match records with xG", len(result))
        return result

    def scrape_team_season_stats(self, season: str) -> pd.DataFrame:
        """Compute team-level season xG aggregates from fixture data.

        Args:
            season: Season start year.

        Returns:
            DataFrame with per-team xG totals for the season.
        """
        raw = self.scrape(season)
        df = self.parse(raw)

        if df.empty:
            return pd.DataFrame()

        # Aggregate per team from both home and away perspective
        home_stats = df.groupby("home_team").agg(
            xg_for=("home_xg", "sum"),
            xg_against=("away_xg", "sum"),
            matches_home=("home_team", "count"),
        ).reset_index().rename(columns={"home_team": "team"})

        away_stats = df.groupby("away_team").agg(
            xg_for=("away_xg", "sum"),
            xg_against=("home_xg", "sum"),
            matches_away=("away_team", "count"),
        ).reset_index().rename(columns={"away_team": "team"})

        merged = pd.merge(home_stats, away_stats, on="team", how="outer", suffixes=("_h", "_a"))
        merged = merged.fillna(0)

        merged["xg_total"] = merged["xg_for_h"] + merged["xg_for_a"]
        merged["xga_total"] = merged["xg_against_h"] + merged["xg_against_a"]
        merged["matches_played"] = merged["matches_home"] + merged["matches_away"]

        return merged[["team", "matches_played", "xg_total", "xga_total"]].round(2)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_fixture_stats(self, fixture_id: int) -> list[dict[str, Any]]:
        """Fetch statistics for a single fixture.

        Args:
            fixture_id: API-Football fixture ID.

        Returns:
            List of team statistics dicts.
        """
        cache_key = f"fixture_stats:{fixture_id}"
        cached = self._api_cache.get(cache_key)
        if cached is not None:
            return cached if isinstance(cached, list) else []

        self._rate_limit()

        try:
            response = self._http_client.get(
                "/fixtures/statistics",
                params={"fixture": fixture_id},
            )
            response.raise_for_status()
            data = response.json()
            result: list[dict[str, Any]] = data.get("response", [])
            self._api_cache.set(cache_key, result, expire=self.cache_ttl * 7)
            return result
        except Exception as e:
            logger.debug("[understat] Stats fetch failed for fixture %d: %s", fixture_id, e)
            return []

    @staticmethod
    def _extract_xg(
        stats: list[dict[str, Any]],
        teams_info: dict[str, Any],
    ) -> tuple[float | None, float | None]:
        """Extract xG values from fixture statistics.

        Args:
            stats: Statistics response from API-Football.
            teams_info: Teams info to match home/away.

        Returns:
            Tuple of (home_xg, away_xg).
        """
        home_xg: float | None = None
        away_xg: float | None = None

        home_id = teams_info.get("home", {}).get("id")
        away_id = teams_info.get("away", {}).get("id")

        for team_stats in stats:
            team_id = team_stats.get("team", {}).get("id")
            statistics = team_stats.get("statistics", [])

            for stat in statistics:
                if stat.get("type", "").lower() == "expected_goals":
                    val = stat.get("value")
                    xg_val = float(val) if val is not None else None

                    if team_id == home_id:
                        home_xg = xg_val
                    elif team_id == away_id:
                        away_xg = xg_val

        return home_xg, away_xg

    def __del__(self) -> None:
        """Clean up HTTP client."""
        try:
            self._http_client.close()
        except Exception:
            pass
