"""Stage 2: Preprocessing & Class Imbalance Handling.

Produces two parallel branches downstream:
  Branch A - original (unaugmented) data, used for BiLSTM / pattern
             mining / attack graph (the "temporal-stream firewall").
  Branch B - SMOTE+ENN augmented tabular data, used only for RF/XGBoost.

``smote_enn_augment`` is the single, only augmentation function in this
project. Do not add a ``smote_knn_augment`` or any other variant.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import config

logger = logging.getLogger(__name__)

_META_COLS = {"Label", "Day", "SessionID", "Token", "TokenID", "LabelID", "Flow Start Time"}


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Numeric feature columns, excluding identifier/meta columns.

    Non-numeric identifier columns (Flow ID, Source IP, Destination IP,
    Flow Start Time as string) are dtype-excluded automatically since
    they are not of numeric dtype; ``_META_COLS`` covers the numeric
    identifier/meta columns that must also be excluded (Day, LabelID, ...).
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return [c for c in numeric_cols if c not in _META_COLS]


def preprocess(df_train, df_val, df_test):
    """Clean, clip, and scale the three chronological splits.

    Returns
    -------
    (train_clean, val_clean, test_clean, scaler, feat_cols)
    """
    feat_cols = get_feature_cols(df_train)

    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=feat_cols)
        df[feat_cols] = df[feat_cols].astype("float64")
        return df

    train_clean = _clean(df_train)
    val_clean = _clean(df_val)
    test_clean = _clean(df_test)

    # Outlier clipping: percentile fit on train only, applied everywhere.
    upper_bounds = train_clean[feat_cols].quantile(config.OUTLIER_CLIP_PERCENTILE / 100.0)
    for split in (train_clean, val_clean, test_clean):
        split[feat_cols] = split[feat_cols].clip(upper=upper_bounds, axis=1)

    # MinMax scaling: fit on train only, never refit on val/test.
    scaler = MinMaxScaler()
    train_clean[feat_cols] = scaler.fit_transform(train_clean[feat_cols]).astype("float64")
    val_clean[feat_cols] = scaler.transform(val_clean[feat_cols]).astype("float64")
    test_clean[feat_cols] = scaler.transform(test_clean[feat_cols]).astype("float64")

    logger.info(
        "preprocess: train=%s val=%s test=%s feat_cols=%d",
        train_clean.shape,
        val_clean.shape,
        test_clean.shape,
        len(feat_cols),
    )
    return train_clean, val_clean, test_clean, scaler, feat_cols


def safe_label_transform(le, labels):
    """Encode a list/array of label strings via ``le``, mapping any label
    ``le`` never saw during ``fit`` to BENIGN's id instead of raising.

    Used everywhere a session- or flow-level label needs to become a
    K-way classifier target after ``encode_labels`` has run — session
    labels in particular are built from the untouched, original "Label"
    strings (see ``encode_labels``), so they can still contain classes
    outside ``le.classes_`` for val/test sessions that include a
    novel/zero-day attack type.
    """
    known = set(le.classes_)
    labels = pd.Series(list(labels)).astype(str)
    labels_safe = labels.where(labels.isin(known), "BENIGN")
    return le.transform(labels_safe).astype(int)


def encode_labels(df_train, df_val, df_test):
    """Fit a LabelEncoder on train labels only; map unseen val/test labels to BENIGN.

    Mutates df_train/df_val/df_test in place, adding a "LabelID" column
    (the K-way classifier target, with any unseen val/test class encoded
    as BENIGN's id since the classifier has no way to recognise a class
    it never saw in training). The original "Label" string column is
    left untouched — a novel/zero-day attack type is still genuinely an
    attack for binary risk-scoring purposes downstream (Stage 11), even
    though the K-way model can only ever call it "BENIGN".

    Returns
    -------
    (le, CLASSES, K)
    """
    le = LabelEncoder()
    df_train["Label"] = df_train["Label"].astype(str)
    le.fit(df_train["Label"])
    classes = list(le.classes_)
    classes_set = set(classes)
    K = len(classes)

    df_train["LabelID"] = le.transform(df_train["Label"])

    for name, df in (("val", df_val), ("test", df_test)):
        labels = df["Label"].astype(str)
        n_unseen = int((~labels.isin(classes_set)).sum())
        if n_unseen:
            logger.warning(
                "%s split: %d rows have a Label unseen in training; encoding their "
                "LabelID as BENIGN for the K-way target, but keeping the true Label "
                "string intact for downstream binary risk scoring",
                name, n_unseen,
            )
        df["LabelID"] = safe_label_transform(le, labels)

    logger.info("encode_labels: K=%d classes=%s", K, classes)
    return le, classes, K


def compute_class_weights(df_train, le, K):
    """W_c = N_train / (K * N_c) for each class c, keyed by encoded class id."""
    n_train = len(df_train)
    counts = df_train["Label"].value_counts()
    weights = {}
    for cls in le.classes_:
        class_id = int(le.transform([cls])[0])
        n_c = int(counts.get(cls, 0))
        weights[class_id] = (n_train / (K * n_c)) if n_c > 0 else 0.0
    return weights


def smote_enn_augment(X_train, y_train, class_counts, K, seed=config.SEED):
    """SMOTE+ENN augmentation for Branch B tabular data only.

    THIS IS THE ONLY AUGMENTATION FUNCTION in the project — do not add a
    ``smote_knn_augment`` or any other variant.

    TEMPORAL-STREAM FIREWALL: only call this for RF/XGBoost tabular
    training data. Never call it for LSTM sequences, FP-Growth/PrefixSpan
    session data, or the attack graph — those always train on original,
    unaugmented session data (synthetic flows have no valid timestamps
    or behavioral tokens and would corrupt temporal structure).

    Classes with fewer than ``config.SMOTE_MIN_SAMPLES`` samples are left
    untouched (class weighting handles them instead — e.g. Heartbleed,
    Infiltration). The majority class is randomly subsampled to
    ``config.SMOTE_MAX_MAJORITY_ROWS`` rows before SMOTEENN runs, so it
    finishes in minutes rather than 30+ on the full training set.

    Parameters
    ----------
    X_train : array-like, shape (n_samples, n_features)
    y_train : array-like, shape (n_samples,) of encoded class ids
    class_counts : dict[int, int]
        Per-class sample counts in X_train/y_train.
    K : int
        Total number of classes.
    seed : int

    Returns
    -------
    (X_aug, y_aug) : tuple[np.ndarray, np.ndarray]
        float64 feature matrix and its label array.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train)

    minority_classes = {c for c, n in class_counts.items() if n < config.SMOTE_MIN_SAMPLES}
    if minority_classes:
        logger.info(
            "smote_enn_augment: skipping classes below %d samples (class weighting instead): %s",
            config.SMOTE_MIN_SAMPLES,
            sorted(minority_classes),
        )

    augment_mask = ~np.isin(y_train, list(minority_classes))
    X_to_augment, y_to_augment = X_train[augment_mask], y_train[augment_mask]
    X_skip, y_skip = X_train[~augment_mask], y_train[~augment_mask]

    before_counts = pd.Series(y_train).value_counts().sort_index().to_dict()
    logger.info("smote_enn_augment: class counts before: %s", before_counts)

    if len(X_to_augment) == 0 or len(np.unique(y_to_augment)) < 2:
        logger.warning(
            "smote_enn_augment: fewer than 2 augmentable classes present; returning original data"
        )
        return X_train, y_train

    # ── Subsample majority class to cap total rows before SMOTE ──────────
    # Keeps ALL minority-within-augment class samples, caps only the
    # majority class to SMOTE_MAX_MAJORITY_ROWS so SMOTE+ENN finishes in
    # minutes instead of 30+ on the full ~975k-row training set.
    unique_aug_classes, aug_class_cnts = np.unique(y_to_augment, return_counts=True)
    majority_class = unique_aug_classes[np.argmax(aug_class_cnts)]

    majority_mask = y_to_augment == majority_class
    minority_mask = ~majority_mask

    X_majority = X_to_augment[majority_mask]
    y_majority = y_to_augment[majority_mask]
    X_minority = X_to_augment[minority_mask]
    y_minority = y_to_augment[minority_mask]

    if len(X_majority) > config.SMOTE_MAX_MAJORITY_ROWS:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_majority), size=config.SMOTE_MAX_MAJORITY_ROWS, replace=False)
        X_majority = X_majority[idx]
        y_majority = y_majority[idx]
        logger.info(
            "smote_enn_augment: majority class %s subsampled %d -> %d rows "
            "before SMOTE+ENN to keep runtime bounded.",
            majority_class,
            int(majority_mask.sum()),
            config.SMOTE_MAX_MAJORITY_ROWS,
        )

    X_to_augment = np.vstack([X_majority, X_minority])
    y_to_augment = np.concatenate([y_majority, y_minority])

    subsampled_counts = pd.Series(y_to_augment).value_counts().sort_index().to_dict()
    logger.info(
        "smote_enn_augment: class counts before SMOTE+ENN (after subsample): %s",
        subsampled_counts,
    )
    # ── End subsample block ───────────────────────────────────────────────

    min_class_count = int(pd.Series(y_to_augment).value_counts().min())
    k_neighbors = min(config.SMOTE_K_NEIGHBORS, max(1, min_class_count - 1))

    try:
        smote_enn = SMOTEENN(
            smote=SMOTE(k_neighbors=k_neighbors, random_state=seed),
            enn=EditedNearestNeighbours(n_neighbors=config.ENN_N_NEIGHBORS),
            random_state=seed,
        )
        X_res, y_res = smote_enn.fit_resample(X_to_augment, y_to_augment)
    except ValueError as exc:
        logger.warning(
            "smote_enn_augment: SMOTEENN raised %s; falling back to SMOTE-only (skip ENN)", exc
        )
        try:
            smote_only = SMOTE(k_neighbors=k_neighbors, random_state=seed)
            X_res, y_res = smote_only.fit_resample(X_to_augment, y_to_augment)
        except ValueError as exc2:
            logger.warning(
                "smote_enn_augment: SMOTE-only fallback also raised %s; returning original data",
                exc2,
            )
            return X_train, y_train

    after_counts = pd.Series(y_res).value_counts().sort_index().to_dict()
    logger.info("smote_enn_augment: class counts after augmentation: %s", after_counts)

    X_aug = np.vstack([X_res, X_skip]).astype(np.float64)
    y_aug = np.concatenate([y_res, y_skip]).astype(y_train.dtype)

    return X_aug, y_aug
