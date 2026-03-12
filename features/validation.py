"""Time-series walk-forward validation — K-Fold CV is banned.

Standard K-Fold cross-validation allows future data to leak into
training folds, which inflates accuracy for time-dependent sports data.
This module implements strict walk-forward (expanding window) splits
where training data always precedes validation data chronologically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Generator

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitInfo:
    """Metadata for a single train/validation split.

    Attributes:
        fold: Fold number (0-indexed).
        train_start: Earliest date in the training set.
        train_end: Latest date in the training set.
        val_start: Earliest date in the validation set.
        val_end: Latest date in the validation set.
        train_size: Number of training samples.
        val_size: Number of validation samples.
    """

    fold: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    train_size: int
    val_size: int


class TimeSeriesSplitter:
    """Walk-forward validation splitter for Premier League match data.

    Generates expanding-window splits where each fold uses all prior
    matches for training and the next ``val_size`` matches for validation.
    This guarantees no future data leakage.

    Args:
        n_splits: Number of validation folds to generate.
        min_train_matches: Minimum matches required in the training set
            before the first validation fold begins.
        val_matches: Number of matches per validation fold. If ``None``,
            splits are computed based on ``n_splits`` dividing the
            available data after ``min_train_matches``.
        gap_matches: Number of matches to skip between training and
            validation sets (prevents overlap/bleeding effects).

    Example::

        splitter = TimeSeriesSplitter(n_splits=5, min_train_matches=380)
        for train_idx, val_idx, info in splitter.split(df, date_col="date"):
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]
    """

    def __init__(
        self,
        n_splits: int = 5,
        min_train_matches: int = 380,
        val_matches: int | None = None,
        gap_matches: int = 0,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2 for meaningful validation.")
        self.n_splits = n_splits
        self.min_train_matches = min_train_matches
        self.val_matches = val_matches
        self.gap_matches = gap_matches

    def split(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
    ) -> Generator[tuple[np.ndarray, np.ndarray, SplitInfo], None, None]:
        """Generate train/validation index splits.

        The DataFrame must be sorted by ``date_col`` before calling.
        If unsorted, this method will sort a copy internally.

        Args:
            df: Feature matrix with a date column.
            date_col: Name of the datetime column.

        Yields:
            Tuple of (train_indices, val_indices, split_info).

        Raises:
            ValueError: If the dataset is too small for the requested splits.
        """
        # Ensure sorted by date
        if not df[date_col].is_monotonic_increasing:
            df = df.sort_values(date_col).reset_index(drop=True)
            logger.warning("DataFrame was not sorted by '%s' — sorted internally.", date_col)

        n_total = len(df)

        if n_total <= self.min_train_matches:
            raise ValueError(
                f"Dataset has {n_total} rows but min_train_matches={self.min_train_matches}. "
                "Not enough data for even one validation fold."
            )

        # Compute validation fold size
        available = n_total - self.min_train_matches
        if self.val_matches is not None:
            val_size = self.val_matches
        else:
            val_size = available // self.n_splits

        if val_size < 10:
            logger.warning(
                "Validation fold size is only %d matches — results may be noisy.",
                val_size,
            )

        fold = 0
        for i in range(self.n_splits):
            val_start_idx = self.min_train_matches + i * val_size + self.gap_matches
            val_end_idx = val_start_idx + val_size

            if val_end_idx > n_total:
                logger.info(
                    "Fold %d truncated at dataset boundary (%d/%d).",
                    i, val_end_idx, n_total,
                )
                val_end_idx = n_total

            if val_start_idx >= n_total:
                break

            train_end_idx = val_start_idx - self.gap_matches
            train_indices = np.arange(0, train_end_idx)
            val_indices = np.arange(val_start_idx, val_end_idx)

            info = SplitInfo(
                fold=fold,
                train_start=str(df.iloc[0][date_col]),
                train_end=str(df.iloc[train_end_idx - 1][date_col]),
                val_start=str(df.iloc[val_start_idx][date_col]),
                val_end=str(df.iloc[val_end_idx - 1][date_col]),
                train_size=len(train_indices),
                val_size=len(val_indices),
            )

            logger.info(
                "Fold %d: train[%s → %s] (%d) | val[%s → %s] (%d)",
                fold, info.train_start, info.train_end, info.train_size,
                info.val_start, info.val_end, info.val_size,
            )

            yield train_indices, val_indices, info
            fold += 1

    def get_split_summary(
        self, df: pd.DataFrame, date_col: str = "date"
    ) -> pd.DataFrame:
        """Generate a summary table of all splits for inspection.

        Args:
            df: Feature matrix.
            date_col: Date column name.

        Returns:
            DataFrame with one row per fold showing date ranges and sizes.
        """
        summaries: list[dict[str, object]] = []
        for _, _, info in self.split(df, date_col):
            summaries.append({
                "fold": info.fold,
                "train_start": info.train_start,
                "train_end": info.train_end,
                "val_start": info.val_start,
                "val_end": info.val_end,
                "train_size": info.train_size,
                "val_size": info.val_size,
            })
        return pd.DataFrame(summaries)
