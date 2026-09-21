"""Tests for Stage 6 multi-model parallel detection (RF + BiLSTM + XGBoost)."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder

import config
from nids import models


@pytest.fixture(scope="module")
def tiny_tabular_data():
    rng = np.random.default_rng(0)
    n_train, n_val, n_feat, K = 120, 30, 6, 3
    X_train = rng.normal(size=(n_train, n_feat))
    y_train = rng.integers(0, K, size=n_train)
    X_val = rng.normal(size=(n_val, n_feat))
    y_val = rng.integers(0, K, size=n_val)
    class_weights = {c: 1.0 for c in range(K)}
    feat_names = [f"f{i}" for i in range(n_feat)]
    return X_train, y_train, X_val, y_val, class_weights, feat_names, K


@pytest.fixture(scope="module")
def tiny_lstm_setup(monkeypatch_module_config):
    rng = np.random.default_rng(0)
    le = LabelEncoder().fit(["BENIGN", "DDoS", "PortScan"])

    def _make_records(n):
        records = []
        for i in range(n):
            tokens = rng.choice(config.VOCAB, size=5).tolist()
            ids = [config.TOKEN2ID[t] for t in tokens]
            pad = config.MAX_SEQ_LEN - len(ids)
            ids = ids + [config.PAD_TOKEN_ID] * pad
            label = le.classes_[i % 3]
            records.append({"session_id": f"S{i}", "token_ids": ids, "label": label})
        return records

    return _make_records(40), _make_records(10), le


@pytest.fixture(scope="module")
def monkeypatch_module_config():
    # Shrink LSTM training cost drastically for test speed.
    original = (config.LSTM_UNITS, config.LSTM_EPOCHS, config.LSTM_EMBED_DIM, config.LSTM_PATIENCE)
    config.LSTM_UNITS = 4
    config.LSTM_EPOCHS = 2
    config.LSTM_EMBED_DIM = 4
    config.LSTM_PATIENCE = 1
    yield
    config.LSTM_UNITS, config.LSTM_EPOCHS, config.LSTM_EMBED_DIM, config.LSTM_PATIENCE = original


def test_train_rf_output_probability_shape(tiny_tabular_data):
    X_train, y_train, X_val, y_val, class_weights, feat_names, K = tiny_tabular_data
    rf, val_metrics = models.train_rf(X_train, y_train, X_val, y_val, class_weights, feat_names)

    probs = rf.predict_proba(X_val)
    assert probs.shape == (len(X_val), K)
    assert "macro_f1" in val_metrics


def test_train_xgb_output_probability_shape(tiny_tabular_data):
    X_train, y_train, X_val, y_val, class_weights, feat_names, K = tiny_tabular_data
    xgb, val_metrics = models.train_xgb(X_train, y_train, X_val, y_val, class_weights, feat_names)

    probs = xgb.predict_proba(X_val)
    assert probs.shape == (len(X_val), K)
    assert "macro_f1" in val_metrics


def test_model_outputs_are_valid_probability_distributions(tiny_tabular_data):
    X_train, y_train, X_val, y_val, class_weights, feat_names, K = tiny_tabular_data
    rf, _ = models.train_rf(X_train, y_train, X_val, y_val, class_weights, feat_names)
    xgb, _ = models.train_xgb(X_train, y_train, X_val, y_val, class_weights, feat_names)

    for model in (rf, xgb):
        probs = model.predict_proba(X_val)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
        assert (probs >= 0).all() and (probs <= 1).all()


def test_train_lstm_trains_on_original_sequences_only(tiny_lstm_setup):
    train_records, val_records, le = tiny_lstm_setup
    class_weights = {i: 1.0 for i in range(3)}

    lstm, val_metrics = models.train_lstm(train_records, val_records, class_weights, K=3, label_encoder=le)

    X_val = np.array([r["token_ids"] for r in val_records], dtype=np.int32)
    probs = lstm.predict(X_val, verbose=0)
    assert probs.shape == (len(val_records), 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)


def test_train_lstm_function_never_calls_smote_enn_augment():
    source = inspect.getsource(models.train_lstm)
    assert "smote_enn_augment" not in source


def test_predict_proba_all_shapes(tiny_tabular_data, tiny_lstm_setup):
    X_train, y_train, X_val, y_val, class_weights, feat_names, K3 = tiny_tabular_data
    rf, _ = models.train_rf(X_train, y_train, X_val, y_val, class_weights, feat_names)
    xgb, _ = models.train_xgb(X_train, y_train, X_val, y_val, class_weights, feat_names)

    train_records, val_records, le = tiny_lstm_setup
    lstm_class_weights = {i: 1.0 for i in range(3)}
    lstm, _ = models.train_lstm(train_records, val_records, lstm_class_weights, K=3, label_encoder=le)

    n = 5
    session_data = {
        "X_AB": X_val[:n],
        "token_ids": np.array([r["token_ids"] for r in val_records[:n]], dtype=np.int32),
        "X_BD": X_val[:n],
    }
    P_A, P_B, P_C = models.predict_proba_all(rf, lstm, xgb, session_data)

    assert P_A.shape == (n, 3)
    assert P_B.shape == (n, 3)
    assert P_C.shape == (n, 3)
