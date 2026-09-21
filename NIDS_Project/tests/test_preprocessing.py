"""Tests for Stage 2 preprocessing & imbalance handling."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import config
from nids import models
from nids.preprocessing import (
    compute_class_weights,
    encode_labels,
    get_feature_cols,
    preprocess,
    smote_enn_augment,
)


def _split_df(small_df):
    n = len(small_df)
    return (
        small_df.iloc[: n // 2].copy(),
        small_df.iloc[n // 2 : n // 2 + 20].copy(),
        small_df.iloc[n // 2 + 20 :].copy(),
    )


def test_preprocess_no_inf_and_no_nan_in_feature_columns(small_df):
    df_train, df_val, df_test = _split_df(small_df)

    train_clean, val_clean, test_clean, scaler, feat_cols = preprocess(df_train, df_val, df_test)

    for split in (train_clean, val_clean, test_clean):
        assert not np.isinf(split[feat_cols].to_numpy()).any()
        assert not split[feat_cols].isna().any().any()


def test_preprocess_scaler_fit_on_train_only(small_df):
    df_train, df_val, df_test = _split_df(small_df)

    train_clean, val_clean, test_clean, scaler, feat_cols = preprocess(df_train, df_val, df_test)

    # Scaled train columns must lie within [0, 1] since MinMaxScaler was fit on train.
    assert train_clean[feat_cols].to_numpy().min() >= -1e-9
    assert train_clean[feat_cols].to_numpy().max() <= 1 + 1e-9

    # scaler.data_min_/data_max_ must equal train's (clipped) raw min/max, not val/test's.
    assert scaler.n_samples_seen_ == len(train_clean)


def test_preprocess_val_test_not_refit(small_df, monkeypatch):
    df_train, df_val, df_test = _split_df(small_df)

    calls = {"fit": 0}
    from sklearn.preprocessing import MinMaxScaler

    original_fit_transform = MinMaxScaler.fit_transform

    def counting_fit_transform(self, *a, **kw):
        calls["fit"] += 1
        return original_fit_transform(self, *a, **kw)

    monkeypatch.setattr(MinMaxScaler, "fit_transform", counting_fit_transform)
    preprocess(df_train, df_val, df_test)

    assert calls["fit"] == 1


def test_compute_class_weights_minority_greater_than_majority(small_df):
    df_train = small_df.copy()
    le, classes, K = encode_labels(df_train, small_df.iloc[:1].copy(), small_df.iloc[:1].copy())
    weights = compute_class_weights(df_train, le, K)

    heartbleed_id = int(le.transform(["Heartbleed"])[0])
    benign_id = int(le.transform(["BENIGN"])[0])

    assert weights[heartbleed_id] > weights[benign_id]


def test_encode_labels_maps_unseen_val_test_labels_to_benign(small_df):
    df_train = small_df.iloc[:150].copy()
    df_val = small_df.iloc[150:170].copy()
    df_test = small_df.iloc[170:].copy()

    df_val.loc[df_val.index[0], "Label"] = "TotallyUnseenAttack"

    le, classes, K = encode_labels(df_train, df_val, df_test)

    assert df_val.loc[df_val.index[0], "Label"] == "BENIGN"
    assert "TotallyUnseenAttack" not in classes


def test_smote_enn_augment_increases_or_maintains_minority_count(small_df):
    df_train = small_df.copy()
    le, classes, K = encode_labels(df_train, small_df.iloc[:1].copy(), small_df.iloc[:1].copy())
    feat_cols = get_feature_cols(df_train)
    X = df_train[feat_cols].fillna(0.0).to_numpy(dtype=np.float64)
    y = df_train["LabelID"].to_numpy()

    counts = {int(c): int(n) for c, n in zip(*np.unique(y, return_counts=True))}
    X_aug, y_aug = smote_enn_augment(X, y, counts, K, seed=config.SEED)

    ddos_id = int(le.transform(["DDoS"])[0])
    n_before = counts[ddos_id]
    n_after = int((y_aug == ddos_id).sum())
    assert n_after >= n_before * 0.5  # ENN may remove some, but SMOTE should still have grown the class materially


def test_smote_enn_augment_skips_rare_classes_below_min_samples(small_df):
    df_train = small_df.copy()
    le, classes, K = encode_labels(df_train, small_df.iloc[:1].copy(), small_df.iloc[:1].copy())
    feat_cols = get_feature_cols(df_train)
    X = df_train[feat_cols].fillna(0.0).to_numpy(dtype=np.float64)
    y = df_train["LabelID"].to_numpy()
    counts = {int(c): int(n) for c, n in zip(*np.unique(y, return_counts=True))}

    heartbleed_id = int(le.transform(["Heartbleed"])[0])
    infiltration_id = int(le.transform(["Infiltration"])[0])
    assert counts[heartbleed_id] < config.SMOTE_MIN_SAMPLES
    assert counts[infiltration_id] < config.SMOTE_MIN_SAMPLES

    X_aug, y_aug = smote_enn_augment(X, y, counts, K, seed=config.SEED)

    assert int((y_aug == heartbleed_id).sum()) == counts[heartbleed_id]
    assert int((y_aug == infiltration_id).sum()) == counts[infiltration_id]


def test_smote_enn_augment_handles_tiny_classes_without_raising():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 4))
    y = np.array([0] * 18 + [1, 1])  # class 1 has only 2 samples
    counts = {0: 18, 1: 2}

    X_aug, y_aug = smote_enn_augment(X, y, counts, K=2, seed=0)

    assert len(X_aug) == len(y_aug)
    assert set(np.unique(y_aug)).issubset({0, 1})


def test_smote_enn_augment_output_contains_only_valid_class_labels(small_df):
    df_train = small_df.copy()
    le, classes, K = encode_labels(df_train, small_df.iloc[:1].copy(), small_df.iloc[:1].copy())
    feat_cols = get_feature_cols(df_train)
    X = df_train[feat_cols].fillna(0.0).to_numpy(dtype=np.float64)
    y = df_train["LabelID"].to_numpy()
    counts = {int(c): int(n) for c, n in zip(*np.unique(y, return_counts=True))}

    X_aug, y_aug = smote_enn_augment(X, y, counts, K, seed=config.SEED)

    assert y_aug.min() >= 0
    assert y_aug.max() <= K - 1


def test_smote_enn_augment_returns_float64_arrays(small_df):
    df_train = small_df.copy()
    le, classes, K = encode_labels(df_train, small_df.iloc[:1].copy(), small_df.iloc[:1].copy())
    feat_cols = get_feature_cols(df_train)
    X = df_train[feat_cols].fillna(0.0).to_numpy(dtype=np.float64)
    y = df_train["LabelID"].to_numpy()
    counts = {int(c): int(n) for c, n in zip(*np.unique(y, return_counts=True))}

    X_aug, y_aug = smote_enn_augment(X, y, counts, K, seed=config.SEED)

    assert X_aug.dtype == np.float64


def test_only_smote_enn_augment_function_name_is_used_for_augmentation():
    """No `smote_knn_augment` or any other augmentation-function variant anywhere."""
    import nids.preprocessing as preprocessing_module

    source = inspect.getsource(preprocessing_module)
    assert "def smote_knn_augment" not in source
    assert not hasattr(preprocessing_module, "smote_knn_augment")
    assert hasattr(preprocessing_module, "smote_enn_augment")


def test_smote_enn_augment_is_never_called_for_lstm_or_session_data():
    """The temporal-stream firewall: train_lstm must never call smote_enn_augment."""
    source = inspect.getsource(models.train_lstm)
    assert "smote_enn_augment" not in source

    import nids.sessions as sessions_module
    import nids.events as events_module
    import nids.attack_graph as attack_graph_module
    import nids.pattern_mining as pattern_mining_module

    for mod in (sessions_module, events_module, attack_graph_module, pattern_mining_module):
        assert "smote_enn_augment" not in inspect.getsource(mod)
