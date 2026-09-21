"""Stage 12: Streaming Evaluation.

Replays the test set through 60-second tumbling windows with a 10-second
stride to measure real-time detection behavior.

Terminology used precisely throughout this module (see spec):
  - "Streaming evaluation": what this module does — NOT "online learning".
  - "Detection time": elapsed time from a session's first event to
    Risk_t >= RISK_SUSPICIOUS_THRESH.
  - "Alert latency": time from attack onset to SOC alert delivery
    (approximated here by detection time within the windowed replay).
  - "Graph update": incremental EMA update of TemporalAttackGraph edge
    weights — parameter-free, adaptive, and explicitly allowed here.
  - "Model update": NONE. RF/BiLSTM/XGBoost weights are frozen; they are
    only ever called with .predict/.predict_proba in this module.
"""

from __future__ import annotations

import bisect
import logging
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .fusion import fuse
from .models import predict_proba_all
from .pattern_mining import compute_sp_score

import config

logger = logging.getLogger(__name__)


def _session_start(record):
    real_ts = [t for t in record["timestamps"] if pd.notna(t)]
    return min(real_ts) if real_ts else pd.NaT


def _session_graph_weight(attack_graph, token_seq):
    toks = [t for t in token_seq if t is not None]
    if len(toks) <= 1:
        return 0.0
    weights = [
        attack_graph.G[u][v]["weight"]
        for u, v in zip(toks[:-1], toks[1:])
        if attack_graph.G.has_edge(u, v)
    ]
    return float(np.mean(weights)) if weights else 0.0


def _mean_inter_event_seconds(timestamps):
    real_ts = sorted(t for t in timestamps if pd.notna(t))
    if len(real_ts) <= 1:
        return 0.0
    diffs = [
        (real_ts[i + 1] - real_ts[i]) / np.timedelta64(1, "s") for i in range(len(real_ts) - 1)
    ]
    return float(np.mean(diffs))


def _false_positive_rate(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return float("nan")
    negatives = y_true == 0
    if negatives.sum() == 0:
        return 0.0
    fp = np.sum((y_pred == 1) & negatives)
    return float(fp / negatives.sum())


def simulate_streaming(
    test_sessions,
    pipeline_components,
    window_size=config.STREAM_WINDOW_SIZE,
    stride=config.STREAM_STRIDE,
):
    """Replay ``test_sessions`` through tumbling windows and score streaming behavior.

    Parameters
    ----------
    test_sessions : list[dict]
        Records from ``events.encode_sessions`` run on the TEST split.
    pipeline_components : dict
        {
          "rf", "lstm", "xgb": frozen, already-trained detectors,
          "attack_graph": a TemporalAttackGraph already fit on training sessions,
          "risk_decider": a fitted AdaptiveRiskDecider (thresholds frozen),
          "fusion_weights": tuple frozen from Stage 7 val calibration,
          "frequent_itemsets", "sequential_patterns": Stage 8 outputs,
          "K": number of classes,
          "delta_t_mean_train": mean inter-event gap (s) computed on training sessions,
          "build_features_fn": callable(list[session_record]) -> dict with
              "X_AB", "token_ids", "X_BD" for predict_proba_all, row-aligned
              to the input list.
        }
    window_size, stride : int, seconds (config.STREAM_WINDOW_SIZE / STREAM_STRIDE)

    Returns
    -------
    dict with streaming_macro_f1, fpr, latency_ms_per_event, throughput_log,
    detection_times_s, n_sessions_processed, n_windows.
    """
    rf = pipeline_components["rf"]
    lstm = pipeline_components["lstm"]
    xgb = pipeline_components["xgb"]
    attack_graph = pipeline_components["attack_graph"]
    risk_decider = pipeline_components["risk_decider"]
    fusion_weights = pipeline_components["fusion_weights"]
    frequent_itemsets = pipeline_components.get("frequent_itemsets")
    sequential_patterns = pipeline_components.get("sequential_patterns")
    K = pipeline_components["K"]
    delta_t_mean_train = pipeline_components["delta_t_mean_train"]
    build_features_fn = pipeline_components["build_features_fn"]

    sessions_sorted = sorted(
        (s for s in test_sessions if pd.notna(_session_start(s))), key=_session_start
    )
    if not sessions_sorted:
        raise ValueError("simulate_streaming: no test sessions with valid timestamps")

    t0 = _session_start(sessions_sorted[0])
    t_end = _session_start(sessions_sorted[-1])

    # Session start times, pre-sorted (sessions_sorted is sorted by this same
    # key), so each window's membership is a bisect range lookup instead of a
    # full O(n_sessions) scan — this is what keeps a real-scale replay
    # (hundreds of thousands of sessions x tens of thousands of windows)
    # tractable instead of O(windows * sessions).
    session_start_times = [_session_start(s) for s in sessions_sorted]

    window_starts = []
    w = t0
    while w <= t_end:
        window_starts.append(w)
        w = w + pd.Timedelta(seconds=stride)

    processed_session_ids = set()
    y_true, y_pred_binary = [], []
    detection_times = {}
    latencies_ms = []
    throughput_log = []

    for w_start in window_starts:
        w_end = w_start + pd.Timedelta(seconds=window_size)
        lo = bisect.bisect_left(session_start_times, w_start)
        hi = bisect.bisect_left(session_start_times, w_end)
        window_sessions = sessions_sorted[lo:hi]
        if not window_sessions:
            continue

        t_wall_start = time.perf_counter()

        features = build_features_fn(window_sessions)
        P_A, P_B, P_C = predict_proba_all(rf, lstm, xgb, features)  # MODEL UPDATE: NONE (inference only)

        SP_t = np.array(
            [compute_sp_score(s, frequent_itemsets, sequential_patterns, K) for s in window_sessions]
        )
        # TC_t computed against the graph state as of the START of this window,
        # then the graph is EMA-refreshed with this window's observations
        # (GRAPH UPDATE: adaptive incremental statistics, not model learning).
        TC_t = np.array(
            [attack_graph.compute_tc(s["token_seq"], s["timestamps"]) for s in window_sessions]
        )
        for s in window_sessions:
            attack_graph.update(s["token_seq"])

        G_w = np.array([_session_graph_weight(attack_graph, s["token_seq"]) for s in window_sessions])
        deltas = np.array([_mean_inter_event_seconds(s["timestamps"]) for s in window_sessions])
        inv_dt = risk_decider.compute_inv_dt(deltas, delta_t_mean_train)

        # FUSION WEIGHT INTEGRITY: fusion_weights are frozen, never re-estimated here.
        R_t = fuse(P_A, P_B, P_C, SP_t, TC_t, fusion_weights)
        risk_score, alert_tier = risk_decider.predict(R_t, SP_t, TC_t, G_w, inv_dt)

        window_elapsed = time.perf_counter() - t_wall_start
        n_events = sum(len([t for t in s["token_seq"] if t is not None]) for s in window_sessions)
        if n_events > 0:
            latencies_ms.append(window_elapsed * 1000.0 / n_events)
        throughput_log.append(
            {
                "window_start": w_start,
                "n_sessions": len(window_sessions),
                "n_events": n_events,
                "throughput_events_per_s": (n_events / window_elapsed) if window_elapsed > 0 else float("inf"),
            }
        )

        for i, s in enumerate(window_sessions):
            sid = s["session_id"]
            if alert_tier[i] == "ATTACK" and sid not in detection_times:
                first_ts = _session_start(s)
                detection_times[sid] = (
                    (w_end - first_ts) / np.timedelta64(1, "s") if pd.notna(first_ts) else None
                )
            if sid not in processed_session_ids:
                processed_session_ids.add(sid)
                y_true.append(1 if s.get("label") not in (None, "BENIGN") else 0)
                y_pred_binary.append(1 if alert_tier[i] != "BENIGN" else 0)

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred_binary)
    macro_f1 = (
        f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0) if len(y_true_arr) else float("nan")
    )
    fpr = _false_positive_rate(y_true_arr, y_pred_arr)

    results = {
        "streaming_macro_f1": float(macro_f1),
        "fpr": fpr,
        "latency_ms_per_event": {
            "mean": float(np.mean(latencies_ms)) if latencies_ms else float("nan"),
            "p50": float(np.percentile(latencies_ms, 50)) if latencies_ms else float("nan"),
            "p95": float(np.percentile(latencies_ms, 95)) if latencies_ms else float("nan"),
            "p99": float(np.percentile(latencies_ms, 99)) if latencies_ms else float("nan"),
        },
        "throughput_log": throughput_log,
        "detection_times_s": detection_times,
        "n_sessions_processed": len(processed_session_ids),
        "n_windows": len(throughput_log),
    }
    logger.info(
        "simulate_streaming: %d windows, %d sessions, streaming_macro_f1=%.4f, fpr=%.4f",
        results["n_windows"], results["n_sessions_processed"], macro_f1, fpr,
    )
    return results
