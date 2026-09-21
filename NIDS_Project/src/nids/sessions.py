"""Stage 4: Bidirectional Session Reconstruction.

Groups individual network flows into bidirectional communication
sessions using a symmetric 5-tuple key, so that "who dialed whom first"
does not create two different sessions for the same conversation.

Session boundaries are created by:
  - a timeout gap (> SESSION_TIMEOUT seconds since the previous flow),
  - an explicit TCP FIN/RST flag observed in the previous flow,
  - a day boundary (sessions never span two different collection days),
  - a hard cap on total session duration (SESSION_MAX_LENGTH seconds).

Call this once per chronological split (train/val/test) so that, by
construction, no session can span a train/test boundary: a session is
implicitly assigned to the partition of its first flow because it is
only ever built from rows belonging to that partition's DataFrame.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

_REQUIRED_COLS = [
    "Source IP", "Destination IP", "Source Port", "Destination Port",
    "Protocol", "Flow Start Time", "Day",
]


def _seconds_between(t1, t2) -> float:
    if pd.isna(t1) or pd.isna(t2):
        return np.inf
    return (t2 - t1) / np.timedelta64(1, "s")


def reconstruct_sessions(df: pd.DataFrame, session_prefix: str = "S") -> pd.DataFrame:
    """Assign a SessionID to every flow in ``df`` via symmetric 5-tuple keys.

    Parameters
    ----------
    df : pd.DataFrame
        A single chronological split (train, val, or test) — never a
        concatenation across splits, so sessions cannot span the
        train/test boundary.
    session_prefix : str
        Prefix for generated SessionIDs (e.g. "TR", "VAL", "TE") so IDs
        stay unique when splits are later combined for reporting.

    Returns
    -------
    pd.DataFrame
        A copy of ``df`` sorted chronologically with a new "SessionID"
        column. Single-event sessions are valid and retained.
    """
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"reconstruct_sessions: missing required columns {missing}")

    df = df.sort_values(["Day", "Flow Start Time"], kind="mergesort").reset_index(drop=True)

    src_ip = df["Source IP"].astype(str).to_numpy()
    dst_ip = df["Destination IP"].astype(str).to_numpy()
    src_port = pd.to_numeric(df["Source Port"], errors="coerce").to_numpy()
    dst_port = pd.to_numeric(df["Destination Port"], errors="coerce").to_numpy()
    protocol = df["Protocol"].to_numpy()

    swap = src_ip > dst_ip
    ip_lo = np.where(swap, dst_ip, src_ip)
    ip_hi = np.where(swap, src_ip, dst_ip)
    port_lo = np.minimum(src_port, dst_port)
    port_hi = np.maximum(src_port, dst_port)

    session_key = pd.Series(
        list(zip(ip_lo, ip_hi, protocol, port_lo, port_hi)), index=df.index
    )

    terminates = pd.Series(False, index=df.index)
    if "FIN Flag Count" in df.columns:
        terminates |= df["FIN Flag Count"].fillna(0) > 0
    if "RST Flag Count" in df.columns:
        terminates |= df["RST Flag Count"].fillna(0) > 0

    times = df["Flow Start Time"].to_numpy()
    days = df["Day"].to_numpy()

    session_ids = np.empty(len(df), dtype=object)
    counter = 0

    for _, group_positions in session_key.groupby(session_key, sort=False).groups.items():
        idx = np.asarray(group_positions)

        session_start_time = None
        prev_terminated = False
        prev_day = None
        prev_time = None
        current_sid = None

        for pos in idx:
            t = times[pos]
            d = days[pos]
            gap_exceeded = prev_time is not None and _seconds_between(prev_time, t) > config.SESSION_TIMEOUT
            length_exceeded = (
                session_start_time is not None
                and _seconds_between(session_start_time, t) > config.SESSION_MAX_LENGTH
            )
            new_session = (
                current_sid is None
                or d != prev_day
                or prev_terminated
                or gap_exceeded
                or length_exceeded
            )
            if new_session:
                counter += 1
                current_sid = f"{session_prefix}{counter:08d}"
                session_start_time = t
            session_ids[pos] = current_sid
            prev_terminated = bool(terminates.iloc[pos])
            prev_day = d
            prev_time = t

    df["SessionID"] = session_ids

    n_sessions = df["SessionID"].nunique()
    session_lengths = df.groupby("SessionID").size()
    logger.info(
        "reconstruct_sessions: %d flows -> %d sessions (mean length=%.2f, "
        "median=%.1f, single-event sessions=%d)",
        len(df), n_sessions, session_lengths.mean(), session_lengths.median(),
        int((session_lengths == 1).sum()),
    )
    return df
