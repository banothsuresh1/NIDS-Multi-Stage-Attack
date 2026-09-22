"""Stage 1: Data Acquisition & Chronological Partition.

Loads the seven CIC-IDS2017 per-day CSV files, fixes the dataset's known
data-quality issues, and produces a strict chronological
train (Days 1-4) / validation (Day 5) / test (Days 6-7) split.

No random shuffling occurs anywhere in this module: rows are sorted by
flow start time within each day and days are concatenated in calendar
order, which is what prevents temporal leakage downstream.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

# Candidate names for the flow-start timestamp column across CIC-IDS2017
# CSV variants (column names are stripped before this lookup happens).
_TIMESTAMP_CANDIDATES = ["Flow Start Time", "Timestamp", "timestamp"]


def _find_timestamp_col(df: pd.DataFrame) -> str:
    for cand in _TIMESTAMP_CANDIDATES:
        if cand in df.columns:
            return cand
    raise KeyError(
        f"No timestamp column found. Looked for {_TIMESTAMP_CANDIDATES}, "
        f"got columns: {list(df.columns)[:10]}..."
    )


def load_day(filepath: str | Path, day_num: int) -> pd.DataFrame:
    """Load a single CIC-IDS2017 day CSV and clean it.

    Handles: leading-space column names, the " Label" -> "Label" rename,
    duplicate embedded header rows, inf values, and unparsable timestamps.
    """
    filepath = Path(filepath)
    df = pd.read_csv(filepath, low_memory=False, encoding="latin1")

    # Known issue: column names carry leading/trailing spaces.
    df.columns = df.columns.str.strip()

    if "Label" not in df.columns:
        raise KeyError(f"'Label' column missing after strip in {filepath.name}")

    # Known issue: some CSVs re-embed the header row as a data row
    # (Label column literally equal to the string "Label").
    before = len(df)
    df = df[df["Label"] != "Label"].copy()
    dropped = before - len(df)
    if dropped:
        logger.info("Day %d: dropped %d duplicate header rows", day_num, dropped)

    ts_col = _find_timestamp_col(df)
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    if ts_col != "Flow Start Time":
        df = df.rename(columns={ts_col: "Flow Start Time"})

    n_unparsable = df["Flow Start Time"].isna().sum()
    if n_unparsable:
        logger.warning(
            "Day %d: %d rows had unparsable timestamps (kept, sorted last)",
            day_num,
            n_unparsable,
        )

    # Known issue: inf values from Flow Bytes/s, Flow Packets/s divisions.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].astype("float64")
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    df["Day"] = day_num

    df = df.sort_values("Flow Start Time", kind="mergesort").reset_index(drop=True)

    logger.info(
        "Day %d (%s): loaded %d rows, %d columns", day_num, filepath.name, *df.shape
    )
    return df


def load_all_days(dataset_dir: str | Path | None = None):
    """Load all 7 days and build the chronological train/val/test split.

    Returns
    -------
    (df_train, df_val, df_test) : tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Train = Days 1-4, Validation = Day 5, Test = Days 6-7.
        CRITICAL: no shuffling occurs anywhere in this function.
    """
    dataset_dir = Path(dataset_dir) if dataset_dir is not None else config.DATASET_DIR

    frames = {}
    for day_num, filename in config.DAY_FILES.items():
        filepath = dataset_dir / filename
        frames[day_num] = load_day(filepath, day_num)

    def _concat(days: list[int]) -> pd.DataFrame:
        parts = [frames[d] for d in days]
        out = pd.concat(parts, axis=0, ignore_index=True)
        return out

    df_train = _concat(config.TRAIN_DAYS)
    df_val = _concat(config.VAL_DAYS)
    df_test = _concat(config.TEST_DAYS)

    for name, split in [("train", df_train), ("val", df_val), ("test", df_test)]:
        dist = split["Label"].value_counts()
        logger.info("Split=%s shape=%s\n%s", name, split.shape, dist.to_string())

    return df_train, df_val, df_test
