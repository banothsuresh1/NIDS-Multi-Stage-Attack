"""Tests for Stage 5 behavioral event encoding (token rule table)."""

from __future__ import annotations

import inspect

import pandas as pd

import config
from nids.events import assign_token, encode_sessions


def _row(**overrides):
    base = {
        "SYN Flag Count": 0, "ACK Flag Count": 0, "RST Flag Count": 0, "FIN Flag Count": 0,
        "Total Length of Fwd Packets": 0, "Total Length of Bwd Packets": 0,
        "Total Fwd Packets": 0, "Total Backward Packets": 0,
        "Destination Port": 8080, "Flow Duration": 10_000_000,  # 10s
        "Flow Packets/s": 1.0, "Flow Bytes/s": 100.0,
    }
    base.update(overrides)
    return pd.Series(base)


def test_connection_attempt_rule():
    row = _row(**{"SYN Flag Count": 1, "ACK Flag Count": 0, "Total Length of Fwd Packets": 2})
    assert assign_token(row) == "CONNECTION_ATTEMPT"


def test_session_established_rule():
    row = _row(**{
        "SYN Flag Count": 1, "ACK Flag Count": 1, "RST Flag Count": 0, "FIN Flag Count": 0,
        "Total Fwd Packets": 2, "Total Backward Packets": 2,
    })
    assert assign_token(row) == "SESSION_ESTABLISHED"


def test_auth_fail_rule():
    row = _row(**{
        "Destination Port": 22, "SYN Flag Count": 1, "ACK Flag Count": 1, "RST Flag Count": 1,
        "Total Fwd Packets": 2, "Total Backward Packets": 2,
        "Total Length of Fwd Packets": 20, "Total Length of Bwd Packets": 20,
    })
    assert assign_token(row) == "AUTH_FAIL"


def test_auth_success_rule():
    row = _row(**{
        "Destination Port": 22, "RST Flag Count": 0,
        "Total Fwd Packets": 3, "Total Backward Packets": 3,
        "Total Length of Fwd Packets": 200, "Total Length of Bwd Packets": 200,
    })
    assert assign_token(row) == "AUTH_SUCCESS"


def test_scan_activity_rule_high_flow_rate():
    # payload >= 10 so this doesn't also match the earlier CONNECTION_ATTEMPT rule.
    row = _row(**{
        "SYN Flag Count": 1, "ACK Flag Count": 0, "Flow Packets/s": 500.0, "Flow Duration": 10_000_000,
        "Total Length of Fwd Packets": 50,
    })
    assert assign_token(row) == "SCAN_ACTIVITY"


def test_scan_activity_rule_short_duration():
    row = _row(**{
        "SYN Flag Count": 1, "ACK Flag Count": 0, "Flow Packets/s": 1.0, "Flow Duration": 100_000,
        "Total Length of Fwd Packets": 50,
    })
    assert assign_token(row) == "SCAN_ACTIVITY"


def test_data_transfer_rule():
    row = _row(**{
        "SYN Flag Count": 1, "ACK Flag Count": 1, "Destination Port": 8080,
        "Total Length of Fwd Packets": 2000, "Total Length of Bwd Packets": 2000,
    })
    assert assign_token(row) == "DATA_TRANSFER"


def test_connection_termination_rule():
    row = _row(**{"FIN Flag Count": 1, "RST Flag Count": 0})
    assert assign_token(row) == "CONNECTION_TERMINATION"


def test_forced_termination_rule():
    row = _row(**{"RST Flag Count": 1, "FIN Flag Count": 0})
    assert assign_token(row) == "FORCED_TERMINATION"


def test_dos_indicator_rule():
    row = _row(**{"Total Fwd Packets": 400, "Total Backward Packets": 200, "Flow Duration": 1_000_000})
    assert assign_token(row) == "DOS_INDICATOR"


def test_post_auth_activity_rule():
    row = _row(**{
        "Destination Port": 3389, "SYN Flag Count": 1, "ACK Flag Count": 1, "RST Flag Count": 0,
        "Total Length of Fwd Packets": 600, "Total Length of Bwd Packets": 0,
    })
    assert assign_token(row) == "POST_AUTH_ACTIVITY"


def test_default_token_is_connection_attempt_for_unmatched_rows():
    row = _row()  # all flags zero, no payload -> matches nothing specific
    assert assign_token(row) == "CONNECTION_ATTEMPT"


def test_assign_token_never_references_label_column():
    source = inspect.getsource(assign_token)
    body = source.split('"""', 2)[-1]  # drop the def line + docstring, keep only the code body
    assert '"Label"' not in body
    assert "['Label']" not in body
    # Also make sure passing a row without a Label column at all works fine.
    row = _row()
    assert isinstance(assign_token(row), str)


def test_encode_sessions_pads_and_truncates(session_flows_df):
    from nids.sessions import reconstruct_sessions

    sessions_df = reconstruct_sessions(session_flows_df)
    records = encode_sessions(sessions_df, max_seq_len=config.MAX_SEQ_LEN)

    for rec in records:
        assert len(rec["token_ids"]) == config.MAX_SEQ_LEN
        assert len(rec["token_seq"]) == config.MAX_SEQ_LEN
        assert len(rec["timestamps"]) == config.MAX_SEQ_LEN
        assert rec["label"] is not None
