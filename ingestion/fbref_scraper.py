"""FBRef scraper — basic match results and schedules.

Since FBRef lost access to advanced Opta data in January 2026,
this scraper is limited to: match dates, scores, venues, and lineups.
xG data is sourced from Understat instead.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup

from ingestion.base_scraper import BaseScraper

logger = logging.getLogger(__name__)

# FBRef season URL pattern: /en/comps/9/{year}-{year+1}/schedule/
# e.g. /en/comps/9/2024-2025/schedule/2024-2025-Premier-League-Scores-and-Fixtures
PREMIER_LEAGUE_COMP_ID = "9"


class FBRefScraper(BaseScraper):
    """Scrape Premier League match results from FBRef.

    Extracts the Scores & Fixtures table for each season including:
    match date, home/away teams, scores, venue, and attendance.

    Example::

        scraper = FBRefScraper()
        df = scraper.run(season="2024")
    """

    def __init__(
        self,
        delay_min: float = 4.0,
        delay_max: float = 8.0,
        cache_ttl: int = 86400,
    ) -> None:
        super().__init__(
            source_name="fbref",
            base_url="https://fbref.com/en",
            delay_min=delay_min,
            delay_max=delay_max,
            cache_ttl=cache_ttl,
        )

    def scrape(self, season: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Fetch match results from the FBRef Scores & Fixtures page.

        Args:
            season: Start year of the season (e.g. ``"2024"`` for 2024-25).

        Returns:
            List of match record dicts with raw FBRef column names.
        """
        start_year = int(season)
        end_year = start_year + 1
        season_slug = f"{start_year}-{end_year}"

        url = (
            f"{self.base_url}/comps/{PREMIER_LEAGUE_COMP_ID}/"
            f"{season_slug}/schedule/{season_slug}-Premier-League-"
            f"Scores-and-Fixtures"
        )

        html = self.fetch_page(url)
        soup = BeautifulSoup(html, "lxml")

        # The main fixtures table has id "sched_{season_slug}_1"
        table = soup.find("table", id=re.compile(r"sched_\d{4}-\d{4}"))
        if table is None:
            # Fallback: try to find any table with "Scores & Fixtures" caption
            table = soup.find("table", class_="stats_table")

        if table is None:
            logger.error("Could not find fixtures table for season %s", season_slug)
            return []

        records: list[dict[str, Any]] = []
        header_row = table.find("thead")
        if header_row is None:
            logger.error("No table header found for season %s", season_slug)
            return []

        headers: list[str] = []
        for th in header_row.find_all("th"):
            text = th.get_text(strip=True)
            headers.append(text)

        tbody = table.find("tbody")
        if tbody is None:
            return []

        for row in tbody.find_all("tr"):
            # Skip spacer rows (rows with class "spacer" or "thead")
            row_classes = row.get("class", [])
            if any(cls in row_classes for cls in ["spacer", "thead", "partial_table"]):
                continue

            cells = row.find_all(["th", "td"])
            if len(cells) < 5:
                continue

            record: dict[str, Any] = {}
            for i, cell in enumerate(cells):
                if i < len(headers):
                    record[headers[i]] = cell.get_text(strip=True)

            # Only include rows that have a valid score (match has been played)
            score = record.get("Score", "")
            if score and "–" in score:
                records.append(record)

        logger.info(
            "[fbref] Scraped %d match results for season %s",
            len(records),
            season_slug,
        )
        return records

    def parse(self, raw_data: list[dict[str, Any]]) -> pd.DataFrame:
        """Parse raw FBRef records into a standardized DataFrame.

        Args:
            raw_data: List of dicts from :meth:`scrape`.

        Returns:
            DataFrame with columns: ``date``, ``home_team``, ``away_team``,
            ``home_goals``, ``away_goals``, ``venue``, ``attendance``,
            ``matchweek``, ``season``.
        """
        if not raw_data:
            return pd.DataFrame()

        df = pd.DataFrame(raw_data)

        # Standardize column names (FBRef uses varying headers)
        column_map = self._detect_columns(df.columns.tolist())

        parsed_rows: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            score_raw = str(row.get(column_map.get("score", "Score"), ""))
            home_goals, away_goals = self._parse_score(score_raw)

            home_raw = str(row.get(column_map.get("home", "Home"), ""))
            away_raw = str(row.get(column_map.get("away", "Away"), ""))

            parsed_rows.append({
                "date": row.get(column_map.get("date", "Date"), ""),
                "home_team": self.normalize_team_name(home_raw),
                "away_team": self.normalize_team_name(away_raw),
                "home_goals": home_goals,
                "away_goals": away_goals,
                "venue": row.get(column_map.get("venue", "Venue"), ""),
                "attendance": self._parse_attendance(
                    str(row.get(column_map.get("attendance", "Attendance"), ""))
                ),
                "matchweek": row.get(column_map.get("matchweek", "Wk"), ""),
                "referee": row.get(column_map.get("referee", "Referee"), ""),
            })

        result = pd.DataFrame(parsed_rows)

        # Convert types
        result["date"] = pd.to_datetime(result["date"], errors="coerce")
        result["home_goals"] = pd.to_numeric(result["home_goals"], errors="coerce")
        result["away_goals"] = pd.to_numeric(result["away_goals"], errors="coerce")
        result["attendance"] = pd.to_numeric(result["attendance"], errors="coerce")

        # Drop rows with no valid date (header repeats, etc.)
        result = result.dropna(subset=["date"])

        logger.info("[fbref] Parsed %d valid match records", len(result))
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_score(score_str: str) -> tuple[int | None, int | None]:
        """Split a score string like ``'2–1'`` into (home_goals, away_goals)."""
        # FBRef uses en-dash (–) not hyphen (-)
        for sep in ["–", "-", "—"]:
            if sep in score_str:
                parts = score_str.split(sep)
                if len(parts) == 2:
                    try:
                        return int(parts[0].strip()), int(parts[1].strip())
                    except ValueError:
                        return None, None
        return None, None

    @staticmethod
    def _parse_attendance(att_str: str) -> int | None:
        """Parse attendance strings like ``'60,704'`` into integers."""
        cleaned = att_str.replace(",", "").strip()
        if cleaned.isdigit():
            return int(cleaned)
        return None

    @staticmethod
    def _detect_columns(columns: list[str]) -> dict[str, str]:
        """Map expected fields to actual FBRef column names.

        FBRef column headers vary slightly between seasons.
        This method provides fuzzy matching to handle that.
        """
        mapping: dict[str, str] = {}
        lower_cols = {c.lower(): c for c in columns}

        field_patterns: dict[str, list[str]] = {
            "date": ["date"],
            "home": ["home", "squad_home"],
            "away": ["away", "squad_away"],
            "score": ["score"],
            "venue": ["venue"],
            "attendance": ["attendance", "attend"],
            "matchweek": ["wk", "week", "matchweek"],
            "referee": ["referee", "ref"],
        }

        for field, patterns in field_patterns.items():
            for pattern in patterns:
                if pattern in lower_cols:
                    mapping[field] = lower_cols[pattern]
                    break

        return mapping
