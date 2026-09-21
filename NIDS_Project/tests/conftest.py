"""Synthetic, dataset-free fixtures shared across the NIDS test suite.

None of these fixtures touch the real CIC-IDS2017 CSVs — every test in
this suite must be runnable with nothing but this repository checked out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from nids.feature_engineering import (
    _GROUP_A_CANDIDATES,
    _GROUP_B_CANDIDATES,
    _GROUP_C_CANDIDATES,
    _GROUP_D_CANDIDATES,
)

ALL_GROUP_FEATURE_COLS = sorted(
    set(_GROUP_A_CANDIDATES) | set(_GROUP_B_CANDIDATES) | set(_GROUP_C_CANDIDATES) | set(_GROUP_D_CANDIDATES)
)

_FLAG_COLS = [
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
]

# Label counts asked for explicitly: Heartbleed (2 rows), Infiltration (3 rows),
# both below SMOTE_MIN_SAMPLES so they exercise the "class weighting only" path.
SMALL_DF_LABEL_COUNTS = {
    "BENIGN": 120,
    "DDoS": 50,
    "PortScan": 20,
    "Bot": 5,
    "Infiltration": 3,
    "Heartbleed": 2,
}


@pytest.fixture
def small_df() -> pd.DataFrame:
    """A 200-row, already-cleaned CIC-IDS2017-shaped DataFrame.

    Represents the output of ``data_loading.load_day`` (stripped column
    names, "Label"/"Day"/"Flow Start Time" present) rather than a raw
    CSV — raw-quirk handling is exercised separately in
    ``test_data_loading.py`` against an actual temp CSV file.
    """
    rng = np.random.default_rng(config.SEED)
    n = sum(SMALL_DF_LABEL_COUNTS.values())
    assert n == 200

    labels = []
    for lbl, cnt in SMALL_DF_LABEL_COUNTS.items():
        labels.extend([lbl] * cnt)
    rng.shuffle(labels)

    data = {
        "Source IP": [f"10.0.0.{i % 50}" for i in range(n)],
        "Destination IP": [f"10.0.1.{i % 30}" for i in range(n)],
        "Source Port": rng.integers(1024, 65535, size=n).astype("float64"),
        "Destination Port": rng.choice([80, 443, 22, 21, 23, 3389, 445, 8080], size=n).astype("float64"),
        "Protocol": rng.choice([6, 17], size=n).astype("float64"),
    }

    start = pd.Timestamp("2017-07-03 09:00:00")
    data["Flow Start Time"] = [start + pd.Timedelta(seconds=int(i * 3)) for i in range(n)]
    data["Day"] = 1

    for col in ALL_GROUP_FEATURE_COLS:
        if col in _FLAG_COLS:
            continue
        data[col] = rng.exponential(scale=100.0, size=n)

    for col in _FLAG_COLS:
        data[col] = rng.integers(0, 2, size=n).astype("float64")

    data["Label"] = labels

    df = pd.DataFrame(data)

    # A handful of known CIC-IDS2017 data-quality issues, injected on purpose
    # so preprocessing tests have something real to clean.
    df.loc[3, "Flow Bytes/s"] = np.inf
    df.loc[7, "Flow Packets/s"] = -np.inf
    df.loc[11, "Flow Duration"] = np.nan

    return df


@pytest.fixture
def small_feat_cols(small_df):
    from nids.preprocessing import get_feature_cols

    return get_feature_cols(small_df)


@pytest.fixture
def label_encoder(small_df):
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    le.fit(small_df["Label"])
    return le


@pytest.fixture
def class_counts(small_df, label_encoder):
    counts = small_df["Label"].value_counts()
    return {int(label_encoder.transform([lbl])[0]): int(cnt) for lbl, cnt in counts.items()}


@pytest.fixture
def session_flows_df():
    """A small, hand-built flow set covering session-boundary edge cases:

    - Rows 0-2: same 5-tuple (forward direction), gaps < timeout -> 1 session.
    - Row 3: same peer, reverse direction (tests symmetric key) but the
      gap since row 2 is still < timeout -> stays in that same session.
    - Row 4: same peer, gap > SESSION_TIMEOUT since row 3 -> new session.
    - Row 5: RST flag set on row 4's session before this -> already
      forces a new session at row 5 too (double-covered on purpose).
    - Row 6: same peer as rows 0-4 but on Day 2 -> new session (day boundary).
    - Row 7: an entirely different peer, single flow -> single-event session.
    """
    base = pd.Timestamp("2017-07-03 09:00:00")
    rows = [
        dict(Source_IP="10.0.0.1", Destination_IP="10.0.1.1", Source_Port=5000, Destination_Port=80,
             Protocol=6, Flow_Start_Time=base, Day=1, FIN=0, RST=0),
        dict(Source_IP="10.0.0.1", Destination_IP="10.0.1.1", Source_Port=5000, Destination_Port=80,
             Protocol=6, Flow_Start_Time=base + pd.Timedelta(seconds=10), Day=1, FIN=0, RST=0),
        dict(Source_IP="10.0.0.1", Destination_IP="10.0.1.1", Source_Port=5000, Destination_Port=80,
             Protocol=6, Flow_Start_Time=base + pd.Timedelta(seconds=20), Day=1, FIN=0, RST=0),
        dict(Source_IP="10.0.1.1", Destination_IP="10.0.0.1", Source_Port=80, Destination_Port=5000,
             Protocol=6, Flow_Start_Time=base + pd.Timedelta(seconds=30), Day=1, FIN=0, RST=1),
        dict(Source_IP="10.0.0.1", Destination_IP="10.0.1.1", Source_Port=5000, Destination_Port=80,
             Protocol=6, Flow_Start_Time=base + pd.Timedelta(seconds=200), Day=1, FIN=0, RST=0),
        dict(Source_IP="10.0.0.1", Destination_IP="10.0.1.1", Source_Port=5000, Destination_Port=80,
             Protocol=6, Flow_Start_Time=base + pd.Timedelta(seconds=210), Day=1, FIN=0, RST=0),
        dict(Source_IP="10.0.0.1", Destination_IP="10.0.1.1", Source_Port=5000, Destination_Port=80,
             Protocol=6, Flow_Start_Time=base + pd.Timedelta(days=1), Day=2, FIN=0, RST=0),
        dict(Source_IP="10.0.2.9", Destination_IP="10.0.2.10", Source_Port=6000, Destination_Port=443,
             Protocol=6, Flow_Start_Time=base + pd.Timedelta(seconds=5), Day=1, FIN=0, RST=0),
    ]

    df = pd.DataFrame(
        {
            "Source IP": [r["Source_IP"] for r in rows],
            "Destination IP": [r["Destination_IP"] for r in rows],
            "Source Port": [r["Source_Port"] for r in rows],
            "Destination Port": [r["Destination_Port"] for r in rows],
            "Protocol": [r["Protocol"] for r in rows],
            "Flow Start Time": [r["Flow_Start_Time"] for r in rows],
            "Day": [r["Day"] for r in rows],
            "FIN Flag Count": [r["FIN"] for r in rows],
            "RST Flag Count": [r["RST"] for r in rows],
            "Label": ["BENIGN"] * len(rows),
        }
    )
    return df


@pytest.fixture
def sample_token_sequences():
    """A handful of hand-built behavioral token sequences (training-like, no timestamps needed)."""
    return [
        ["CONNECTION_ATTEMPT", "SESSION_ESTABLISHED", "DATA_TRANSFER", "CONNECTION_TERMINATION"],
        ["CONNECTION_ATTEMPT", "SESSION_ESTABLISHED", "DATA_TRANSFER", "CONNECTION_TERMINATION"],
        ["SCAN_ACTIVITY", "SCAN_ACTIVITY", "SCAN_ACTIVITY"],
        ["CONNECTION_ATTEMPT", "AUTH_FAIL", "AUTH_FAIL", "AUTH_SUCCESS", "POST_AUTH_ACTIVITY"],
        ["CONNECTION_ATTEMPT", "FORCED_TERMINATION"],
        ["DOS_INDICATOR", "DOS_INDICATOR", "FORCED_TERMINATION"],
    ]


@pytest.fixture
def sample_session_records():
    """Session records shaped like ``events.encode_sessions`` output."""
    base = pd.Timestamp("2017-07-03 09:00:00")
    max_len = config.MAX_SEQ_LEN

    def _pad(tokens):
        ids = [config.TOKEN2ID[t] for t in tokens]
        ts = [base + pd.Timedelta(seconds=5 * i) for i in range(len(tokens))]
        pad_n = max_len - len(tokens)
        return (
            tokens + [None] * pad_n,
            ids + [config.PAD_TOKEN_ID] * pad_n,
            ts + [pd.NaT] * pad_n,
        )

    records = []
    specs = [
        ("S001", ["CONNECTION_ATTEMPT", "SESSION_ESTABLISHED", "DATA_TRANSFER"], "BENIGN"),
        ("S002", ["SCAN_ACTIVITY", "SCAN_ACTIVITY"], "PortScan"),
        ("S003", ["CONNECTION_ATTEMPT", "AUTH_FAIL", "AUTH_FAIL"], "BENIGN"),
        ("S004", ["DOS_INDICATOR", "DOS_INDICATOR", "FORCED_TERMINATION"], "DDoS"),
        ("S005", ["CONNECTION_ATTEMPT"], "BENIGN"),
    ]
    for sid, tokens, label in specs:
        seq, ids, ts = _pad(tokens)
        records.append(
            {"session_id": sid, "token_seq": seq, "token_ids": ids, "label": label, "timestamps": ts}
        )
    return records
