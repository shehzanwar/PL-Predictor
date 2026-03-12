"""Abstract base class for all data scrapers.

Provides shared infrastructure: rate limiting, caching, retry logic,
and team name normalization. All concrete scrapers inherit from this.
"""

from __future__ import annotations

import json
import logging
import random
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from diskcache import Cache

logger = logging.getLogger(__name__)

# Project root (two levels up from ingestion/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


class BaseScraper(ABC):
    """Abstract base class enforcing a consistent scraper interface.

    Every scraper must implement ``scrape()`` to fetch raw data and
    ``parse()`` to convert it into a clean DataFrame.

    Attributes:
        source_name: Identifier for the data source (e.g. ``"fbref"``).
        base_url: Root URL for the scraper target.
        delay_min: Minimum seconds between HTTP requests.
        delay_max: Maximum seconds between HTTP requests.
        cache: Disk-based response cache to avoid redundant requests.
        team_mappings: Canonical team name lookup table.
    """

    def __init__(
        self,
        source_name: str,
        base_url: str,
        delay_min: float = 3.0,
        delay_max: float = 6.0,
        cache_dir: str | None = None,
        cache_ttl: int = 86400,
    ) -> None:
        self.source_name = source_name
        self.base_url = base_url.rstrip("/")
        self.delay_min = delay_min
        self.delay_max = delay_max
        self.cache_ttl = cache_ttl

        # Disk cache
        resolved_cache_dir = cache_dir or str(PROJECT_ROOT / "data" / "cache" / source_name)
        self.cache: Cache = Cache(resolved_cache_dir)

        # HTTP session with rotating User-Agent
        self.session = requests.Session()
        self._user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3) AppleWebKit/605.1.15 "
            "(KHTML, like Gecko) Version/17.2 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0",
        ]

        # Team name mappings
        self.team_mappings: dict[str, dict[str, str]] = self._load_team_mappings()

        # Reverse lookup: source-specific name → canonical name
        self._reverse_map: dict[str, str] = {}
        for canonical, aliases in self.team_mappings.items():
            source_alias = aliases.get(self.source_name, "")
            if source_alias:
                self._reverse_map[source_alias] = canonical

    # ------------------------------------------------------------------
    # Abstract interface — every scraper must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def scrape(self, season: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Fetch raw data for a given season.

        Args:
            season: Season identifier (e.g. ``"2024"`` for 2024-25).
            **kwargs: Source-specific parameters.

        Returns:
            List of raw data records (dicts).
        """
        ...

    @abstractmethod
    def parse(self, raw_data: list[dict[str, Any]]) -> pd.DataFrame:
        """Transform raw records into a structured DataFrame.

        Args:
            raw_data: Output from :meth:`scrape`.

        Returns:
            Cleaned and standardized DataFrame.
        """
        ...

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    def fetch_page(self, url: str, use_cache: bool = True) -> str:
        """Fetch an HTML page with caching and rate limiting.

        Args:
            url: Full URL to fetch.
            use_cache: If True, check and store in disk cache.

        Returns:
            Page HTML as a string.

        Raises:
            requests.HTTPError: On non-2xx responses after retries.
        """
        # Check cache first
        if use_cache and url in self.cache:
            logger.debug("Cache hit: %s", url)
            cached_value = self.cache.get(url)
            if isinstance(cached_value, str):
                return cached_value

        # Rate limiting
        self._rate_limit()

        # Rotate User-Agent
        self.session.headers.update({
            "User-Agent": random.choice(self._user_agents),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        })

        logger.info("Fetching: %s", url)
        response = self.session.get(url, timeout=30)
        response.raise_for_status()

        html: str = response.text

        # Cache the response
        if use_cache:
            self.cache.set(url, html, expire=self.cache_ttl)

        return html

    def normalize_team_name(self, raw_name: str) -> str:
        """Convert a source-specific team name to the canonical name.

        Args:
            raw_name: Team name as it appears on the source website.

        Returns:
            Canonical team name. Falls back to ``raw_name`` if no mapping exists.
        """
        canonical = self._reverse_map.get(raw_name.strip())
        if canonical is None:
            logger.warning(
                "[%s] No mapping for team '%s' — using raw name.",
                self.source_name,
                raw_name,
            )
            return raw_name.strip()
        return canonical

    def save_raw(self, df: pd.DataFrame, filename: str) -> Path:
        """Save a DataFrame to the raw data directory.

        Args:
            df: Data to save.
            filename: Output filename (e.g. ``"fbref_matches_2024.csv"``).

        Returns:
            Path to the saved file.
        """
        raw_dir = PROJECT_ROOT / "data" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        filepath = raw_dir / filename
        df.to_csv(filepath, index=False)
        logger.info("Saved raw data: %s (%d rows)", filepath, len(df))
        return filepath

    def run(self, season: str, **kwargs: Any) -> pd.DataFrame:
        """Execute the full scrape → parse → save pipeline.

        Args:
            season: Season identifier.
            **kwargs: Passed to :meth:`scrape`.

        Returns:
            Parsed DataFrame.
        """
        logger.info("[%s] Starting pipeline for season %s", self.source_name, season)
        raw_data = self.scrape(season, **kwargs)
        df = self.parse(raw_data)
        filename = f"{self.source_name}_season_{season}.csv"
        self.save_raw(df, filename)
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _rate_limit(self) -> None:
        """Sleep for a random duration between configured min/max delays."""
        delay = random.uniform(self.delay_min, self.delay_max)
        logger.debug("Rate limiting: sleeping %.1fs", delay)
        time.sleep(delay)

    @staticmethod
    def _load_team_mappings() -> dict[str, dict[str, str]]:
        """Load team name mappings from the config JSON file."""
        mappings_path = CONFIG_DIR / "team_mappings.json"
        if not mappings_path.exists():
            logger.warning("Team mappings file not found at %s", mappings_path)
            return {}
        with open(mappings_path, "r", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
        teams: dict[str, dict[str, str]] = data.get("teams", {})
        return teams
