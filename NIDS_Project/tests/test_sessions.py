"""Tests for Stage 4 session reconstruction."""

from __future__ import annotations

import pandas as pd

import config
from nids.sessions import reconstruct_sessions


def test_session_respects_timeout(session_flows_df):
    out = reconstruct_sessions(session_flows_df)
    # Rows at t=0,10,20,30 form one session; the row at t=200 is > 60s after
    # t=30, so it must start a new session even ignoring the RST at t=30.
    row_30 = out[out["Flow Start Time"] == pd.Timestamp("2017-07-03 09:00:30")].iloc[0]
    row_200 = out[out["Flow Start Time"] == pd.Timestamp("2017-07-03 09:03:20")].iloc[0]
    assert row_30["SessionID"] != row_200["SessionID"]


def test_symmetric_key_groups_forward_and_reverse_flows(session_flows_df):
    out = reconstruct_sessions(session_flows_df)
    row_0 = out[out["Flow Start Time"] == pd.Timestamp("2017-07-03 09:00:00")].iloc[0]
    row_30 = out[out["Flow Start Time"] == pd.Timestamp("2017-07-03 09:00:30")].iloc[0]
    # row_30 is the reverse-direction flow of the same peer pair, within timeout.
    assert row_0["SessionID"] == row_30["SessionID"]


def test_session_ends_on_rst_flag(session_flows_df):
    out = reconstruct_sessions(session_flows_df)
    row_30 = out[out["Flow Start Time"] == pd.Timestamp("2017-07-03 09:00:30")].iloc[0]
    # row_30 has RST=1; the very next same-peer flow (t=200) must be a new session
    # (already true from the timeout, but this asserts the RST reasoning holds too
    # via a tight-gap variant).
    tight_gap_df = session_flows_df.copy()
    tight_gap_df.loc[4, "Flow Start Time"] = pd.Timestamp("2017-07-03 09:00:35")  # only +5s after RST row
    out2 = reconstruct_sessions(tight_gap_df)
    row_30_2 = out2[out2["Flow Start Time"] == pd.Timestamp("2017-07-03 09:00:30")].iloc[0]
    row_35_2 = out2[out2["Flow Start Time"] == pd.Timestamp("2017-07-03 09:00:35")].iloc[0]
    assert row_30_2["SessionID"] != row_35_2["SessionID"]


def test_no_session_spans_day_boundary(session_flows_df):
    out = reconstruct_sessions(session_flows_df)
    day1_rows = out[(out["Source IP"] == "10.0.0.1") & (out["Day"] == 1)]
    day2_rows = out[(out["Source IP"] == "10.0.0.1") & (out["Day"] == 2)]
    assert set(day1_rows["SessionID"]) & set(day2_rows["SessionID"]) == set()


def test_single_event_sessions_are_valid_and_retained(session_flows_df):
    out = reconstruct_sessions(session_flows_df)
    other_peer = out[out["Source IP"] == "10.0.2.9"]
    assert len(other_peer) == 1
    session_size = out[out["SessionID"] == other_peer.iloc[0]["SessionID"]]
    assert len(session_size) == 1


def test_no_session_spans_train_test_boundary_by_construction(session_flows_df):
    # Sessions are only ever built per-split, so simulate a "train" and "test"
    # call on disjoint slices and confirm SessionIDs never coincide meaningfully
    # across the two independent reconstructions touching the same peer.
    train_slice = session_flows_df.iloc[:4].copy()
    test_slice = session_flows_df.iloc[4:].copy()

    train_out = reconstruct_sessions(train_slice, session_prefix="TR")
    test_out = reconstruct_sessions(test_slice, session_prefix="TE")

    assert set(train_out["SessionID"]).isdisjoint(set(test_out["SessionID"]))
    assert all(sid.startswith("TR") for sid in train_out["SessionID"])
    assert all(sid.startswith("TE") for sid in test_out["SessionID"])


def test_max_session_length_cap():
    base = pd.Timestamp("2017-07-03 09:00:00")
    n = 5
    # Consecutive flows 30s apart (within timeout) but spanning far past
    # SESSION_MAX_LENGTH in total.
    step = config.SESSION_MAX_LENGTH // (n - 1) + 10
    rows = []
    for i in range(n):
        rows.append(
            dict(
                Source_IP="10.0.0.5", Destination_IP="10.0.1.5", Source_Port=5000,
                Destination_Port=80, Protocol=6, Flow_Start_Time=base + pd.Timedelta(seconds=i * step),
                Day=1, FIN=0, RST=0,
            )
        )
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
            "Label": ["BENIGN"] * n,
        }
    )

    out = reconstruct_sessions(df)
    assert out["SessionID"].nunique() > 1
