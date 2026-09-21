"""Tests for Stage 12 streaming evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import config
from nids.attack_graph import TemporalAttackGraph
from nids.risk import AdaptiveRiskDecider
from nids.streaming import simulate_streaming


def _make_session(sid, tokens, label, start):
    ids = [config.TOKEN2ID[t] for t in tokens]
    ts = [start + pd.Timedelta(seconds=3 * i) for i in range(len(tokens))]
    pad = config.MAX_SEQ_LEN - len(tokens)
    return {
        "session_id": sid,
        "token_seq": tokens + [None] * pad,
        "token_ids": ids + [config.PAD_TOKEN_ID] * pad,
        "label": label,
        "timestamps": ts + [pd.NaT] * pad,
    }


@pytest.fixture(scope="module")
def streaming_pipeline():
    rng = np.random.default_rng(0)
    n_feat, K = 6, 2  # BENIGN vs everything-else-collapsed-to-1-class for a tiny fast test
    X = rng.normal(size=(50, n_feat))
    y = rng.integers(0, K, size=50)

    rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    xgb = XGBClassifier(n_estimators=10, max_depth=2, random_state=0, eval_metric="mlogloss").fit(X, y)

    from tensorflow.keras import layers, models as keras_models

    inputs = layers.Input(shape=(config.MAX_SEQ_LEN,), dtype="int32")
    h = layers.Embedding(input_dim=config.VOCAB_SIZE, output_dim=4)(inputs)
    h = layers.Bidirectional(layers.LSTM(4))(h)
    outputs = layers.Dense(K, activation="softmax")(h)
    lstm = keras_models.Model(inputs, outputs)
    lstm.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

    graph = TemporalAttackGraph()
    graph.fit([
        ["CONNECTION_ATTEMPT", "SESSION_ESTABLISHED", "DATA_TRANSFER"],
        # A diluting transition so CONNECTION_ATTEMPT->SESSION_ESTABLISHED
        # starts below 1.0 and EMA updates are actually observable.
        ["CONNECTION_ATTEMPT", "FORCED_TERMINATION"],
        ["SCAN_ACTIVITY", "SCAN_ACTIVITY"],
    ])

    risk_decider = AdaptiveRiskDecider()
    n_val = 40
    risk_decider.fit(
        rng.random((n_val, K)), rng.random((n_val, K)), rng.random(n_val),
        rng.random(n_val), rng.random(n_val), rng.integers(0, 2, size=n_val),
    )

    def build_features_fn(session_records):
        n = len(session_records)
        return {
            "X_AB": rng.normal(size=(n, n_feat)),
            "token_ids": np.array([r["token_ids"] for r in session_records], dtype=np.int32),
            "X_BD": rng.normal(size=(n, n_feat)),
        }

    components = {
        "rf": rf, "lstm": lstm, "xgb": xgb, "attack_graph": graph, "risk_decider": risk_decider,
        "fusion_weights": config.DEFAULT_FUSION_WEIGHTS, "frequent_itemsets": None,
        "sequential_patterns": None, "K": K, "delta_t_mean_train": 5.0,
        "build_features_fn": build_features_fn,
    }
    return components


@pytest.fixture
def test_sessions_fixture():
    start = pd.Timestamp("2017-07-06 10:00:00")
    return [
        _make_session("T1", ["CONNECTION_ATTEMPT", "SESSION_ESTABLISHED", "DATA_TRANSFER"], "BENIGN", start),
        _make_session("T2", ["SCAN_ACTIVITY", "SCAN_ACTIVITY"], "PortScan", start + pd.Timedelta(seconds=20)),
        _make_session("T3", ["DOS_INDICATOR", "DOS_INDICATOR"], "DDoS", start + pd.Timedelta(seconds=70)),
        _make_session("T4", ["CONNECTION_ATTEMPT"], "BENIGN", start + pd.Timedelta(seconds=90)),
    ]


def test_rf_lstm_xgb_weights_frozen_during_streaming(streaming_pipeline, test_sessions_fixture):
    rf_params_before = streaming_pipeline["rf"].get_params()
    xgb_before = streaming_pipeline["xgb"].get_booster().save_raw()
    lstm_weights_before = [w.copy() for w in streaming_pipeline["lstm"].get_weights()]

    simulate_streaming(test_sessions_fixture, streaming_pipeline, window_size=60, stride=10)

    assert streaming_pipeline["rf"].get_params() == rf_params_before
    assert streaming_pipeline["xgb"].get_booster().save_raw() == xgb_before
    for before, after in zip(lstm_weights_before, streaming_pipeline["lstm"].get_weights()):
        assert np.allclose(before, after)


def test_attack_graph_ema_weights_do_update_during_streaming(streaming_pipeline, test_sessions_fixture):
    graph = streaming_pipeline["attack_graph"]
    w_before = graph.G["CONNECTION_ATTEMPT"]["SESSION_ESTABLISHED"]["weight"]

    simulate_streaming(test_sessions_fixture, streaming_pipeline, window_size=60, stride=10)

    w_after = graph.G["CONNECTION_ATTEMPT"]["SESSION_ESTABLISHED"]["weight"]
    assert w_after != w_before


def test_latency_measured_per_event(streaming_pipeline, test_sessions_fixture):
    results = simulate_streaming(test_sessions_fixture, streaming_pipeline, window_size=60, stride=10)

    assert "latency_ms_per_event" in results
    assert results["latency_ms_per_event"]["mean"] >= 0
    assert results["n_windows"] > 0
    assert results["n_sessions_processed"] == len(test_sessions_fixture)
