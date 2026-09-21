"""Tests for Stage 13 explainability & evidence chain."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import config
from nids.attack_graph import TemporalAttackGraph
from nids.explainability import generate_evidence_chain

_EXPECTED_KEYS = {
    "session_id", "risk_score", "alert_tier", "top_features_rf", "top_features_xgb",
    "lstm_token_attributions", "graph_path", "novel_edges", "attack_family",
    "kill_chain_stage", "recommended_action", "analyst_summary",
}


@pytest.fixture(scope="module")
def tiny_models_and_session():
    rng = np.random.default_rng(0)
    n_feat, K = 8, 3
    X = rng.normal(size=(60, n_feat))
    y = rng.integers(0, K, size=60)

    rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    xgb = XGBClassifier(n_estimators=10, max_depth=2, random_state=0, eval_metric="mlogloss").fit(X, y)

    from tensorflow.keras import layers, models as keras_models

    inputs = layers.Input(shape=(config.MAX_SEQ_LEN,), dtype="int32")
    x = layers.Embedding(input_dim=config.VOCAB_SIZE, output_dim=4)(inputs)
    x = layers.Bidirectional(layers.LSTM(4))(x)
    outputs = layers.Dense(K, activation="softmax")(x)
    lstm = keras_models.Model(inputs, outputs)
    lstm.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

    graph = TemporalAttackGraph()
    graph.fit([["CONNECTION_ATTEMPT", "SESSION_ESTABLISHED", "DATA_TRANSFER"]])

    token_seq = ["CONNECTION_ATTEMPT", "AUTH_FAIL", "AUTH_FAIL", None, None]
    token_ids = np.array(
        [config.TOKEN2ID.get(t, config.PAD_TOKEN_ID) for t in token_seq], dtype=np.int32
    )
    token_ids = np.pad(token_ids, (0, config.MAX_SEQ_LEN - len(token_ids)), constant_values=config.PAD_TOKEN_ID)
    token_seq_padded = token_seq + [None] * (config.MAX_SEQ_LEN - len(token_seq))

    session = {
        "session_id": "S_TEST_1",
        "risk_score": 0.82,
        "alert_tier": "ATTACK",
        "predicted_class": 1,
        "X_AB": X[0],
        "X_BD": X[0],
        "token_ids": token_ids,
        "token_seq": token_seq_padded,
    }
    feat_names = {"rf": [f"f{i}" for i in range(n_feat)], "xgb": [f"f{i}" for i in range(n_feat)]}

    return session, rf, xgb, lstm, graph, feat_names


def test_evidence_report_contains_all_required_keys(tiny_models_and_session):
    session, rf, xgb, lstm, graph, feat_names = tiny_models_and_session
    report = generate_evidence_chain(session, rf, xgb, lstm, graph, feat_names, top_k=5)

    assert _EXPECTED_KEYS.issubset(report.keys())


def test_top_features_rf_has_exactly_5_entries(tiny_models_and_session):
    session, rf, xgb, lstm, graph, feat_names = tiny_models_and_session
    report = generate_evidence_chain(session, rf, xgb, lstm, graph, feat_names, top_k=5)

    assert len(report["top_features_rf"]) == 5


def test_top_features_xgb_has_exactly_5_entries(tiny_models_and_session):
    session, rf, xgb, lstm, graph, feat_names = tiny_models_and_session
    report = generate_evidence_chain(session, rf, xgb, lstm, graph, feat_names, top_k=5)

    assert len(report["top_features_xgb"]) == 5


def test_graph_path_contains_valid_vocab_tokens(tiny_models_and_session):
    session, rf, xgb, lstm, graph, feat_names = tiny_models_and_session
    report = generate_evidence_chain(session, rf, xgb, lstm, graph, feat_names, top_k=5)

    for u, v, w in report["graph_path"]:
        assert u in config.VOCAB
        assert v in config.VOCAB
        assert isinstance(w, float)
