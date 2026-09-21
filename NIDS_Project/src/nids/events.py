"""Stage 5: Behavioral Event Encoding.

Maps every flow into one of 10 behavioral event tokens using only
network-observable, rule-based signals (TCP flags, packet/byte counts,
destination port). The ground-truth Label column is never consulted —
this is what makes the resulting token sequences usable as unsupervised
input to sequential pattern mining and the attack-state graph.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

import config

# ---------------------------------------------------------------------------
# Token rule table (published verbatim, must be reproducible):
#
# TOKEN                    RULE
# ------------------------------------------------------------------------
# CONNECTION_ATTEMPT       SYN flag set, ACK not set, payload < 10 bytes
# SESSION_ESTABLISHED      SYN+ACK flags, pkt_count >= 3, no RST/FIN
# AUTH_FAIL                dst_port in AUTH_PORTS, SYN+ACK+RST pattern,
#                          pkt_count < 10, small payload (< 100 bytes)
# AUTH_SUCCESS             dst_port in AUTH_PORTS, sustained bidirectional
#                          transfer, pkt_count >= 5, no RST
# SCAN_ACTIVITY            high flow rate (> 100 flows/s) OR very short
#                          duration (< 0.5s), SYN-only pattern
# DATA_TRANSFER            large payload (> 1000 bytes), bidirectional,
#                          no auth port, established connection
# CONNECTION_TERMINATION   FIN flag set, graceful close
# FORCED_TERMINATION       RST flag set
# DOS_INDICATOR            pkt_count > 500 OR byte_rate > 1MB/s,
#                          short flow duration (< 5s)
# POST_AUTH_ACTIVITY       dst_port in AUTH_PORTS, large outbound payload
#                          (> 500 bytes) after successful auth pattern
# ------------------------------------------------------------------------
# Default: CONNECTION_ATTEMPT if no rule matches.
# ---------------------------------------------------------------------------


def _get(row, col, default=0.0):
    val = row[col] if col in row else default
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return val


def assign_token(flow_row) -> str:
    """Rule-based token assignment for a single flow. Never reads Label."""
    syn = _get(flow_row, "SYN Flag Count", 0) > 0
    ack = _get(flow_row, "ACK Flag Count", 0) > 0
    rst = _get(flow_row, "RST Flag Count", 0) > 0
    fin = _get(flow_row, "FIN Flag Count", 0) > 0

    fwd_bytes = _get(flow_row, "Total Length of Fwd Packets", 0)
    bwd_bytes = _get(flow_row, "Total Length of Bwd Packets", 0)
    payload = fwd_bytes + bwd_bytes
    bidirectional = fwd_bytes > 0 and bwd_bytes > 0

    fwd_pkts = _get(flow_row, "Total Fwd Packets", 0)
    bwd_pkts = _get(flow_row, "Total Backward Packets", 0)
    pkt_count = fwd_pkts + bwd_pkts

    dst_port = int(_get(flow_row, "Destination Port", -1))
    duration_s = _get(flow_row, "Flow Duration", 0) / 1e6  # CICFlowMeter reports microseconds
    flow_pkts_per_s = _get(flow_row, "Flow Packets/s", 0.0)
    byte_rate = _get(flow_row, "Flow Bytes/s", 0.0)

    established = syn and ack

    if syn and not ack and payload < 10:
        return "CONNECTION_ATTEMPT"

    if syn and ack and pkt_count >= 3 and not rst and not fin:
        return "SESSION_ESTABLISHED"

    if dst_port in config.AUTH_PORTS and syn and ack and rst and pkt_count < 10 and payload < 100:
        return "AUTH_FAIL"

    if dst_port in config.AUTH_PORTS and bidirectional and pkt_count >= 5 and not rst:
        return "AUTH_SUCCESS"

    if (flow_pkts_per_s > 100 or duration_s < 0.5) and syn and not ack:
        return "SCAN_ACTIVITY"

    if payload > 1000 and bidirectional and dst_port not in config.AUTH_PORTS and established:
        return "DATA_TRANSFER"

    if fin and not rst:
        return "CONNECTION_TERMINATION"

    if rst:
        return "FORCED_TERMINATION"

    if (pkt_count > 500 or byte_rate > 1_000_000) and duration_s < 5:
        return "DOS_INDICATOR"

    if dst_port in config.AUTH_PORTS and fwd_bytes > 500 and established and not rst:
        return "POST_AUTH_ACTIVITY"

    return "CONNECTION_ATTEMPT"


def _session_label(labels: list[str]) -> str:
    """A session is labeled by its dominant non-BENIGN flow, else BENIGN.

    Multi-stage attacks are exactly sessions containing a mix of benign
    reconnaissance-looking flows and a malicious payload flow, so a
    session is only "BENIGN" when every flow in it is benign.
    """
    non_benign = [l for l in labels if l != "BENIGN"]
    if not non_benign:
        return "BENIGN"
    return Counter(non_benign).most_common(1)[0][0]


def encode_sessions(session_df: pd.DataFrame, max_seq_len: int = config.MAX_SEQ_LEN):
    """Turn a session-annotated flow DataFrame into padded token sequences.

    Returns
    -------
    list[dict] with keys: session_id, token_seq, token_ids, label, timestamps
    """
    if "SessionID" not in session_df.columns:
        raise KeyError("encode_sessions requires a 'SessionID' column (run reconstruct_sessions first)")

    records = []
    for session_id, group in session_df.groupby("SessionID", sort=False):
        group = group.sort_values("Flow Start Time", kind="mergesort")

        tokens = [assign_token(row) for _, row in group.iterrows()]
        timestamps = list(group["Flow Start Time"])
        labels = list(group["Label"]) if "Label" in group.columns else []
        label = _session_label(labels) if labels else None

        token_ids_full = [config.TOKEN2ID[t] for t in tokens]
        seq_len = len(token_ids_full)

        if seq_len >= max_seq_len:
            token_ids = token_ids_full[:max_seq_len]
            tokens_out = tokens[:max_seq_len]
            timestamps_out = timestamps[:max_seq_len]
        else:
            pad_needed = max_seq_len - seq_len
            token_ids = token_ids_full + [config.PAD_TOKEN_ID] * pad_needed
            tokens_out = tokens + [None] * pad_needed
            timestamps_out = timestamps + [pd.NaT] * pad_needed

        records.append(
            {
                "session_id": session_id,
                "token_seq": tokens_out,
                "token_ids": token_ids,
                "label": label,
                "timestamps": timestamps_out,
            }
        )
    return records
