"""Pipeline orchestrator — coordinates all scrapers into a single workflow.

This module is the main entry point for the data ingestion pipeline.
It is called by the GitHub Actions cron job and can also be run manually.

Usage::

    python -m ingestion.orchestrator          # Full pipeline
    python -m ingestion.orchestrator --season 2024  # Single season
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ingestion.base_scraper import PROJECT_ROOT
from ingestion.fbref_scraper import FBRefScraper
from ingestion.understat_scraper import UnderstatScraper
from ingestion.api_football_client import APIFootballClient
from ingestion.injury_scraper import InjuryScraper

logger = logging.getLogger(__name__)

# Directories
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Default seasons to scrape (post-pandemic, as per project plan)
DEFAULT_SEASONS = ["2020", "2021", "2022", "2023", "2024", "2025"]

# Output filenames in data/processed/
OUTPUT_MATCHES = "pl_matches_merged.csv"
OUTPUT_XG_SEASON = "pl_xg_season_stats.csv"
OUTPUT_INJURIES = "pl_current_injuries.csv"
OUTPUT_STANDINGS = "pl_current_standings.csv"
OUTPUT_UPCOMING = "pl_upcoming_fixtures.csv"


class PipelineOrchestrator:
    """Coordinate all data scrapers into a unified ingestion pipeline.

    The orchestrator runs each scraper, merges match-level data from
    FBRef and Understat, and saves processed outputs to ``data/processed/``.

    Attributes:
        seasons: List of season start years to process.
        fbref: FBRef scraper instance.
        understat: Understat scraper instance.
        api_football: API-Football client instance.
        injury_scraper: Injury scraper instance.
    """

    def __init__(
        self,
        seasons: list[str] | None = None,
        api_key: str | None = None,
    ) -> None:
        self.seasons = seasons or DEFAULT_SEASONS
        self.fbref = FBRefScraper()
        self.understat = UnderstatScraper()
        self.api_football = APIFootballClient(api_key=api_key)
        self.injury_scraper = InjuryScraper(api_key=api_key)

        # Ensure output directories exist
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    def run_full_pipeline(self) -> dict[str, Path]:
        """Execute the complete data ingestion pipeline.

        Steps:
            1. Scrape match results from FBRef (all seasons).
            2. Scrape xG data from Understat (all seasons).
            3. Merge FBRef + Understat on (date, home_team, away_team).
            4. Fetch current injuries from API-Football.
            5. Fetch current standings from API-Football.
            6. Fetch upcoming fixtures from API-Football.
            7. Save all outputs to ``data/processed/``.

        Returns:
            Dict mapping output names to their file paths.
        """
        logger.info("=" * 60)
        logger.info("PL-Predictor Data Pipeline — Starting at %s", datetime.utcnow())
        logger.info("=" * 60)

        output_paths: dict[str, Path] = {}

        # ------- Step 1 & 2: Historical match data (FBRef + Understat) -------
        all_fbref: list[pd.DataFrame] = []
        all_understat: list[pd.DataFrame] = []
        all_xg_season: list[pd.DataFrame] = []

        for season in self.seasons:
            logger.info("--- Processing season %s ---", season)

            # FBRef: match results
            try:
                fbref_df = self.fbref.run(season)
                fbref_df["season"] = season
                all_fbref.append(fbref_df)
                logger.info("[fbref] Season %s: %d matches", season, len(fbref_df))
            except Exception as e:
                logger.error("[fbref] Failed for season %s: %s", season, e)

            # Understat: xG data
            try:
                understat_df = self.understat.run(season)
                understat_df["season"] = season
                all_understat.append(understat_df)
                logger.info(
                    "[understat] Season %s: %d matches", season, len(understat_df)
                )
            except Exception as e:
                logger.error("[understat] Failed for season %s: %s", season, e)

            # Understat: season-level team xG stats
            try:
                xg_season = self.understat.scrape_team_season_stats(season)
                xg_season["season"] = season
                all_xg_season.append(xg_season)
            except Exception as e:
                logger.error("[understat] Team stats failed for %s: %s", season, e)

        # ------- Step 3: Merge FBRef + Understat -------
        merged_df = self._merge_match_data(all_fbref, all_understat)
        merged_path = PROCESSED_DIR / OUTPUT_MATCHES
        merged_df.to_csv(merged_path, index=False)
        output_paths["matches"] = merged_path
        logger.info("Saved merged matches: %s (%d rows)", merged_path, len(merged_df))

        # Save season-level xG stats
        if all_xg_season:
            xg_season_df = pd.concat(all_xg_season, ignore_index=True)
            xg_path = PROCESSED_DIR / OUTPUT_XG_SEASON
            xg_season_df.to_csv(xg_path, index=False)
            output_paths["xg_season"] = xg_path
            logger.info("Saved xG season stats: %s", xg_path)

        # ------- Step 4: Current injuries -------
        current_season = self.seasons[-1]
        try:
            injuries_df = self.injury_scraper.get_team_availability_summary(
                current_season
            )
            injuries_path = PROCESSED_DIR / OUTPUT_INJURIES
            injuries_df.to_csv(injuries_path, index=False)
            output_paths["injuries"] = injuries_path
            logger.info("Saved injuries: %s", injuries_path)
        except Exception as e:
            logger.error("[injuries] Failed: %s", e)

        # ------- Step 5: Current standings -------
        try:
            standings_df = self.api_football.get_standings(season=int(current_season))
            standings_path = PROCESSED_DIR / OUTPUT_STANDINGS
            standings_df.to_csv(standings_path, index=False)
            output_paths["standings"] = standings_path
            logger.info("Saved standings: %s", standings_path)
        except Exception as e:
            logger.error("[standings] Failed: %s", e)

        # ------- Step 6: Upcoming fixtures -------
        try:
            upcoming_df = self.api_football.get_upcoming_fixtures(next_n=10)
            upcoming_path = PROCESSED_DIR / OUTPUT_UPCOMING
            upcoming_df.to_csv(upcoming_path, index=False)
            output_paths["upcoming"] = upcoming_path
            logger.info("Saved upcoming fixtures: %s", upcoming_path)
        except Exception as e:
            logger.error("[upcoming] Failed: %s", e)

        logger.info("=" * 60)
        logger.info("Pipeline complete. %d outputs saved.", len(output_paths))
        logger.info("=" * 60)

        return output_paths

    def run_daily_update(self) -> dict[str, Path]:
        """Run a lightweight daily update (current season only).

        This is the default mode for the GitHub Actions cron job.
        It only scrapes the current season and refreshes injury/standings data.

        Returns:
            Dict mapping output names to their file paths.
        """
        current_season = self.seasons[-1]
        logger.info("Running daily update for season %s", current_season)

        output_paths: dict[str, Path] = {}

        # Refresh current season match data
        try:
            fbref_df = self.fbref.run(current_season)
            fbref_df["season"] = current_season
        except Exception as e:
            logger.error("[fbref] Daily update failed: %s", e)
            fbref_df = pd.DataFrame()

        try:
            understat_df = self.understat.run(current_season)
            understat_df["season"] = current_season
        except Exception as e:
            logger.error("[understat] Daily update failed: %s", e)
            understat_df = pd.DataFrame()

        # Check if we have existing historical data to merge with
        existing_path = PROCESSED_DIR / OUTPUT_MATCHES
        if existing_path.exists():
            existing_df = pd.read_csv(existing_path, parse_dates=["date"])
            # Remove old current-season data and replace with fresh scrape
            historical_df = existing_df[
                existing_df["season"] != current_season
            ].copy()
        else:
            historical_df = pd.DataFrame()

        # Merge the fresh data
        merged_df = self._merge_match_data(
            [historical_df, fbref_df] if not historical_df.empty else [fbref_df],
            [understat_df],
        )
        merged_path = PROCESSED_DIR / OUTPUT_MATCHES
        merged_df.to_csv(merged_path, index=False)
        output_paths["matches"] = merged_path
        logger.info("Updated matches: %d rows", len(merged_df))

        # Refresh season xG stats
        try:
            xg_season = self.understat.scrape_team_season_stats(current_season)
            xg_season["season"] = current_season
            xg_path = PROCESSED_DIR / OUTPUT_XG_SEASON

            # Merge with existing historical xG stats
            if xg_path.exists():
                existing_xg = pd.read_csv(xg_path)
                existing_xg = existing_xg[existing_xg["season"] != current_season]
                xg_season = pd.concat([existing_xg, xg_season], ignore_index=True)

            xg_season.to_csv(xg_path, index=False)
            output_paths["xg_season"] = xg_path
        except Exception as e:
            logger.error("[understat] xG season update failed: %s", e)

        # Refresh injuries, standings, upcoming
        try:
            injuries_df = self.injury_scraper.get_team_availability_summary(
                current_season
            )
            injuries_path = PROCESSED_DIR / OUTPUT_INJURIES
            injuries_df.to_csv(injuries_path, index=False)
            output_paths["injuries"] = injuries_path
        except Exception as e:
            logger.error("[injuries] Daily update failed: %s", e)

        try:
            standings_df = self.api_football.get_standings(season=int(current_season))
            standings_path = PROCESSED_DIR / OUTPUT_STANDINGS
            standings_df.to_csv(standings_path, index=False)
            output_paths["standings"] = standings_path
        except Exception as e:
            logger.error("[standings] Daily update failed: %s", e)

        try:
            upcoming_df = self.api_football.get_upcoming_fixtures(next_n=10)
            upcoming_path = PROCESSED_DIR / OUTPUT_UPCOMING
            upcoming_df.to_csv(upcoming_path, index=False)
            output_paths["upcoming"] = upcoming_path
        except Exception as e:
            logger.error("[upcoming] Daily update failed: %s", e)

        return output_paths

    # ------------------------------------------------------------------
    # Merge logic
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_match_data(
        fbref_dfs: list[pd.DataFrame],
        understat_dfs: list[pd.DataFrame],
    ) -> pd.DataFrame:
        """Merge FBRef and Understat data on (date, home_team, away_team).

        FBRef provides: venue, attendance, matchweek, referee.
        Understat provides: xG, xGA.
        Both provide: date, teams, goals (used for validation).

        Args:
            fbref_dfs: List of FBRef DataFrames.
            understat_dfs: List of Understat DataFrames.

        Returns:
            Merged DataFrame with all columns.
        """
        # Concatenate all season DataFrames
        fbref_all = pd.concat(
            [df for df in fbref_dfs if not df.empty], ignore_index=True
        ) if fbref_dfs else pd.DataFrame()

        understat_all = pd.concat(
            [df for df in understat_dfs if not df.empty], ignore_index=True
        ) if understat_dfs else pd.DataFrame()

        if fbref_all.empty and understat_all.empty:
            logger.warning("Both FBRef and Understat data are empty.")
            return pd.DataFrame()

        if fbref_all.empty:
            logger.warning("FBRef data is empty — returning Understat only.")
            return understat_all

        if understat_all.empty:
            logger.warning("Understat data is empty — returning FBRef only.")
            return fbref_all

        # Normalize date columns to date-only (no time) for merging
        fbref_all["merge_date"] = pd.to_datetime(
            fbref_all["date"], errors="coerce"
        ).dt.date
        understat_all["merge_date"] = pd.to_datetime(
            understat_all["date"], errors="coerce"
        ).dt.date

        # Select non-overlapping columns from Understat for the merge
        understat_cols_to_add = ["merge_date", "home_team", "away_team",
                                 "home_xg", "away_xg", "understat_match_id"]
        understat_subset = understat_all[
            [c for c in understat_cols_to_add if c in understat_all.columns]
        ].copy()

        # Left join: keep all FBRef matches, add xG where available
        merged = pd.merge(
            fbref_all,
            understat_subset,
            on=["merge_date", "home_team", "away_team"],
            how="left",
        )

        # Clean up
        merged = merged.drop(columns=["merge_date"], errors="ignore")

        # Sort by date
        merged = merged.sort_values("date").reset_index(drop=True)

        logger.info(
            "Merged dataset: %d rows (%d with xG data)",
            len(merged),
            merged["home_xg"].notna().sum() if "home_xg" in merged.columns else 0,
        )

        return merged


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def _setup_logging() -> None:
    """Configure structured logging for the pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                PROJECT_ROOT / "data" / "pipeline.log",
                encoding="utf-8",
            ),
        ],
    )


def main() -> None:
    """CLI entry point for the ingestion pipeline."""
    parser = argparse.ArgumentParser(
        description="PL-Predictor Data Ingestion Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "daily"],
        default="daily",
        help="Pipeline mode: 'full' for all seasons, 'daily' for current season only.",
    )
    parser.add_argument(
        "--season",
        type=str,
        default=None,
        help="Single season to scrape (overrides default list).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API-Football key (overrides API_FOOTBALL_KEY env var).",
    )

    args = parser.parse_args()

    _setup_logging()

    seasons: list[str] | None = [args.season] if args.season else None
    orchestrator = PipelineOrchestrator(seasons=seasons, api_key=args.api_key)

    if args.mode == "full":
        results = orchestrator.run_full_pipeline()
    else:
        results = orchestrator.run_daily_update()

    logger.info("Pipeline finished. Outputs:")
    for name, path in results.items():
        logger.info("  %s -> %s", name, path)


if __name__ == "__main__":
    main()
