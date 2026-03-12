"""API-Football REST client — injuries, fixtures, standings.

Uses the free tier of API-Football (api-sports.io) which provides
100 requests/day across all endpoints. All responses are cached
for 24 hours to minimize API usage.
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

# Premier League ID in API-Football
PREMIER_LEAGUE_ID = 39


class APIFootballClient(BaseScraper):
    """REST client for API-Football (api-sports.io).

    Handles authentication, rate-limit tracking, response caching,
    and team name normalization for injury, fixture, and standings data.

    The API key is read from the ``API_FOOTBALL_KEY`` environment variable.

    Example::

        client = APIFootballClient()
        injuries_df = client.get_injuries(season=2024)
        fixtures_df = client.get_fixtures(season=2024)
    """

    def __init__(
        self,
        api_key: str | None = None,
        cache_ttl: int = 86400,
    ) -> None:
        super().__init__(
            source_name="api_football",
            base_url="https://v3.football.api-sports.io",
            delay_min=0.5,
            delay_max=1.0,
            cache_ttl=cache_ttl,
        )
        self.api_key = api_key or os.environ.get("API_FOOTBALL_KEY", "")
        if not self.api_key:
            logger.warning(
                "[api_football] No API key found. Set API_FOOTBALL_KEY env var. "
                "Requests will fail with 401."
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
            str(PROJECT_ROOT / "data" / "cache" / "api_football")
        )

    def scrape(self, season: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Fetch fixture results from API-Football.

        Args:
            season: Season year (e.g. ``"2024"`` for 2024-25).

        Returns:
            List of fixture dicts from the API response.
        """
        return self._api_request(
            endpoint="/fixtures",
            params={"league": PREMIER_LEAGUE_ID, "season": int(season)},
        )

    def parse(self, raw_data: list[dict[str, Any]]) -> pd.DataFrame:
        """Parse fixture data into a standardized DataFrame.

        Args:
            raw_data: List of fixture dicts from :meth:`scrape`.

        Returns:
            DataFrame with match details.
        """
        if not raw_data:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for fixture in raw_data:
            fixture_info = fixture.get("fixture", {})
            teams_info = fixture.get("teams", {})
            goals_info = fixture.get("goals", {})

            home_name = teams_info.get("home", {}).get("name", "")
            away_name = teams_info.get("away", {}).get("name", "")

            rows.append({
                "date": fixture_info.get("date", ""),
                "home_team": self.normalize_team_name(home_name),
                "away_team": self.normalize_team_name(away_name),
                "home_goals": goals_info.get("home"),
                "away_goals": goals_info.get("away"),
                "status": fixture_info.get("status", {}).get("short", ""),
                "venue": fixture_info.get("venue", {}).get("name", ""),
                "fixture_id": fixture_info.get("id"),
                "round": fixture.get("league", {}).get("round", ""),
            })

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
        return df

    # ------------------------------------------------------------------
    # Specialized endpoints
    # ------------------------------------------------------------------

    def get_injuries(self, season: int) -> pd.DataFrame:
        """Fetch current injury data for all PL teams.

        Args:
            season: Season year (e.g. ``2024``).

        Returns:
            DataFrame with columns: ``team``, ``player_name``,
            ``player_id``, ``type`` (injury/suspension), ``reason``.
        """
        raw = self._api_request(
            endpoint="/injuries",
            params={"league": PREMIER_LEAGUE_ID, "season": season},
        )

        if not raw:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for entry in raw:
            team_info = entry.get("team", {})
            player_info = entry.get("player", {})

            rows.append({
                "team": self.normalize_team_name(team_info.get("name", "")),
                "player_name": player_info.get("name", ""),
                "player_id": player_info.get("id"),
                "injury_type": player_info.get("type", ""),
                "reason": player_info.get("reason", ""),
            })

        return pd.DataFrame(rows)

    def get_standings(self, season: int) -> pd.DataFrame:
        """Fetch current league standings.

        Args:
            season: Season year (e.g. ``2024``).

        Returns:
            DataFrame with full standings table.
        """
        raw = self._api_request(
            endpoint="/standings",
            params={"league": PREMIER_LEAGUE_ID, "season": season},
        )

        if not raw:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for league_entry in raw:
            standings_list = league_entry.get("league", {}).get("standings", [[]])
            for group in standings_list:
                for team_entry in group:
                    team_info = team_entry.get("team", {})
                    all_stats = team_entry.get("all", {})

                    rows.append({
                        "rank": team_entry.get("rank"),
                        "team": self.normalize_team_name(team_info.get("name", "")),
                        "points": team_entry.get("points"),
                        "played": all_stats.get("played"),
                        "wins": all_stats.get("win"),
                        "draws": all_stats.get("draw"),
                        "losses": all_stats.get("lose"),
                        "goals_for": all_stats.get("goals", {}).get("for"),
                        "goals_against": all_stats.get("goals", {}).get("against"),
                        "goal_diff": team_entry.get("goalsDiff"),
                        "form": team_entry.get("form", ""),
                    })

        return pd.DataFrame(rows)

    def get_upcoming_fixtures(self, next_n: int = 10) -> pd.DataFrame:
        """Fetch the next N upcoming PL fixtures.

        Args:
            next_n: Number of upcoming fixtures to fetch.

        Returns:
            DataFrame of upcoming match details.
        """
        raw = self._api_request(
            endpoint="/fixtures",
            params={
                "league": PREMIER_LEAGUE_ID,
                "next": next_n,
            },
        )
        return self.parse(raw)

    # ------------------------------------------------------------------
    # Internal API request handler
    # ------------------------------------------------------------------

    def _api_request(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Make a cached API request.

        Args:
            endpoint: API endpoint path (e.g. ``"/fixtures"``).
            params: Query parameters.

        Returns:
            ``response`` list from the API JSON payload.
        """
        params = params or {}
        cache_key = f"{endpoint}:{sorted(params.items())}"

        # Check cache
        cached = self._api_cache.get(cache_key)
        if cached is not None:
            logger.debug("[api_football] Cache hit: %s", cache_key)
            if isinstance(cached, list):
                return cached
            return []

        if not self.api_key:
            logger.error("[api_football] Cannot make request — no API key configured.")
            return []

        self._rate_limit()

        try:
            response = self._http_client.get(endpoint, params=params)
            response.raise_for_status()
            data: dict[str, Any] = response.json()
        except (httpx.HTTPError, Exception) as e:
            logger.error("[api_football] Request failed for %s: %s", endpoint, e)
            return []

        # Check for API errors
        errors = data.get("errors", {})
        if errors:
            logger.error("[api_football] API errors: %s", errors)
            return []

        result: list[dict[str, Any]] = data.get("response", [])

        # Track remaining rate limit
        remaining = data.get("results", 0)
        logger.info(
            "[api_football] %s returned %d results", endpoint, remaining
        )

        # Cache result
        self._api_cache.set(cache_key, result, expire=self.cache_ttl)

        return result

    def __del__(self) -> None:
        """Clean up HTTP client on garbage collection."""
        try:
            self._http_client.close()
        except Exception:
            pass
