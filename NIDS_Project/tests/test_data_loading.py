"""Tests for Stage 1 data loading: raw CIC-IDS2017 quirk handling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nids.data_loading import load_day


def _write_raw_csv(tmp_path, rows, columns):
    path = tmp_path / "Monday-WorkingHours.pcap_ISCX.csv"
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(path, index=False)
    return path


def test_load_day_strips_column_names_and_renames_label(tmp_path):
    columns = [" Flow Duration", " Destination Port", " Timestamp", " Label"]
    rows = [
        [100, 80, "03/07/2017 09:00:00", "BENIGN"],
        [200, 443, "03/07/2017 09:00:05", "DDoS"],
    ]
    path = _write_raw_csv(tmp_path, rows, columns)

    df = load_day(path, day_num=1)

    assert "Label" in df.columns
    assert " Label" not in df.columns
    assert "Flow Start Time" in df.columns
    assert "Day" in df.columns and (df["Day"] == 1).all()


def test_load_day_drops_duplicate_header_rows(tmp_path):
    columns = [" Flow Duration", " Destination Port", " Timestamp", " Label"]
    rows = [
        [100, 80, "03/07/2017 09:00:00", "BENIGN"],
        # A duplicate header row embedded as data, as seen in real CIC-IDS2017 CSVs.
        ["Flow Duration", "Destination Port", "Timestamp", "Label"],
        [200, 443, "03/07/2017 09:00:05", "DDoS"],
    ]
    path = _write_raw_csv(tmp_path, rows, columns)

    df = load_day(path, day_num=1)

    assert len(df) == 2
    assert "Label" not in df["Label"].values


def test_load_day_replaces_inf_with_nan(tmp_path):
    columns = [" Flow Duration", " Flow Bytes/s", " Destination Port", " Timestamp", " Label"]
    rows = [
        [100, np.inf, 80, "03/07/2017 09:00:00", "BENIGN"],
        [200, 5.0, 443, "03/07/2017 09:00:05", "DDoS"],
    ]
    path = _write_raw_csv(tmp_path, rows, columns)

    df = load_day(path, day_num=1)

    assert not np.isinf(df["Flow Bytes/s"]).any()
    assert df["Flow Bytes/s"].dtype == np.float64


def test_load_day_parses_unparsable_timestamps_as_nat_and_sorts_chronologically(tmp_path):
    columns = [" Flow Duration", " Destination Port", " Timestamp", " Label"]
    rows = [
        [300, 80, "03/07/2017 09:00:30", "BENIGN"],
        [100, 80, "not-a-timestamp", "BENIGN"],
        [200, 80, "03/07/2017 09:00:10", "BENIGN"],
    ]
    path = _write_raw_csv(tmp_path, rows, columns)

    df = load_day(path, day_num=1)

    assert df["Flow Start Time"].isna().sum() == 1
    valid = df.dropna(subset=["Flow Start Time"])
    assert list(valid["Flow Start Time"]) == sorted(valid["Flow Start Time"])


def test_load_day_uses_float64_for_numeric_columns(tmp_path):
    columns = [" Flow Duration", " Destination Port", " Timestamp", " Label"]
    rows = [[100, 80, "03/07/2017 09:00:00", "BENIGN"]]
    path = _write_raw_csv(tmp_path, rows, columns)

    df = load_day(path, day_num=1)

    assert df["Flow Duration"].dtype == np.float64
    assert df["Destination Port"].dtype == np.float64
