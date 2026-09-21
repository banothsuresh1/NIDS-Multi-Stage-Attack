"""Stage 6: Multi-Model Parallel Detection (RF + BiLSTM + XGBoost).

Three independent detectors trained on different feature views:
  - Random Forest   on Group A+B (flow-statistical + protocol) features,
                     Branch B (SMOTE+ENN augmented) tabular data.
  - BiLSTM          on behavioral token sequences, Branch A (original,
                     un-augmented) session data — the temporal-stream
                     firewall means SMOTE+ENN output never reaches here.
  - XGBoost         on Group B+D (protocol + TCP flag) features,
                     Branch B (SMOTE+ENN augmented) tabular data.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

import config

logger = logging.getLogger(__name__)


def _classification_metrics(y_true, y_pred, K):
    labels = list(range(K))
    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {
        "macro_f1": float(macro_f1),
        "per_class_f1": {c: float(f) for c, f in zip(labels, per_class_f1)},
    }


def train_rf(X_train_B, y_train_B, X_val, y_val, class_weights, feat_cols_AB):
    """Train the Random Forest detector on Branch B (SMOTE+ENN) Group A+B features."""
    rf = RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,
        max_features=config.RF_MAX_FEATURES,
        class_weight=class_weights,
        random_state=config.SEED,
        n_jobs=-1,
    )
    rf.fit(X_train_B, y_train_B)

    y_pred = rf.predict(X_val)
    K = len(class_weights)
    val_metrics = _classification_metrics(y_val, y_pred, K)

    if len(feat_cols_AB) == rf.n_features_in_:
        importances = sorted(
            zip(feat_cols_AB, rf.feature_importances_), key=lambda t: t[1], reverse=True
        )
        val_metrics["top_features"] = importances[:10]

    logger.info("train_rf: val macro-F1=%.4f", val_metrics["macro_f1"])
    return rf, val_metrics


def _build_lstm_model(K):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    inputs = layers.Input(shape=(config.MAX_SEQ_LEN,), dtype="int32")
    # NOTE: input_dim uses config.VOCAB_SIZE (= len(VOCAB) + 1), one row larger
    # than the 10-token vocabulary, to give the padding id (config.PAD_TOKEN_ID)
    # a valid embedding row instead of indexing out of range.
    x = layers.Embedding(input_dim=config.VOCAB_SIZE, output_dim=config.LSTM_EMBED_DIM)(inputs)
    x = layers.Bidirectional(layers.LSTM(config.LSTM_UNITS, return_sequences=False))(x)
    x = layers.Dropout(config.LSTM_DROPOUT)(x)
    outputs = layers.Dense(K, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LSTM_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_lstm(token_sequences_train, token_sequences_val, class_weights, K, label_encoder=None):
    """Train the BiLSTM detector on ORIGINAL session token sequences only.

    CRITICAL (temporal-stream firewall): ``token_sequences_train`` must be
    built from un-augmented sessions — SMOTE+ENN synthetic rows have no
    valid timestamps or behavioral tokens and must never be resequenced
    and fed here.

    ``label_encoder`` should be the same ``sklearn.LabelEncoder`` returned
    by ``preprocessing.encode_labels`` so this model's class ids line up
    with RF's and XGBoost's. If omitted, a local encoder is fit on the
    training labels seen here (alignment with other models is then only
    guaranteed if their class sets and sort order coincide).
    """
    import tensorflow as tf

    if label_encoder is None:
        labels_seen = sorted({r["label"] for r in token_sequences_train if r["label"] is not None})
        label_encoder = LabelEncoder().fit(labels_seen)
        logger.warning(
            "train_lstm: no label_encoder passed; fit a local one on training labels only."
        )

    def _to_arrays(records):
        from .preprocessing import safe_label_transform

        X = np.array([r["token_ids"] for r in records], dtype=np.int32)
        y = safe_label_transform(label_encoder, [r["label"] for r in records])
        return X, y

    X_train, y_train = _to_arrays(token_sequences_train)
    X_val, y_val = _to_arrays(token_sequences_val)

    model = _build_lstm_model(K)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=config.LSTM_PATIENCE, restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=config.LSTM_EPOCHS,
        batch_size=config.LSTM_BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=2,
    )

    val_probs = model.predict(X_val, batch_size=config.LSTM_BATCH_SIZE, verbose=0)
    val_preds = np.argmax(val_probs, axis=1)
    val_metrics = _classification_metrics(y_val, val_preds, K)

    logger.info("train_lstm: val macro-F1=%.4f", val_metrics["macro_f1"])
    return model, val_metrics


def train_xgb(X_train_B, y_train_B, X_val, y_val, class_weights, feat_cols_BD):
    """Train the XGBoost detector on Branch B (SMOTE+ENN) Group B+D features."""
    sample_weight = np.array([class_weights[int(c)] for c in y_train_B], dtype=np.float64)

    xgb = XGBClassifier(
        n_estimators=config.XGB_N_ESTIMATORS,
        max_depth=config.XGB_MAX_DEPTH,
        learning_rate=config.XGB_LR,
        subsample=config.XGB_SUBSAMPLE,
        colsample_bytree=config.XGB_COLSAMPLE,
        objective="multi:softprob",
        num_class=len(class_weights),
        eval_metric="mlogloss",
        random_state=config.SEED,
        n_jobs=-1,
    )
    # NOTE: use_label_encoder was removed from xgboost>=1.6 and is no longer
    # a valid constructor argument; labels here are already 0..K-1 integers
    # from preprocessing.encode_labels, so no encoding step is needed.
    xgb.fit(X_train_B, y_train_B, sample_weight=sample_weight)

    y_pred = xgb.predict(X_val)
    K = len(class_weights)
    val_metrics = _classification_metrics(y_val, y_pred, K)

    if len(feat_cols_BD) == xgb.n_features_in_:
        importances = sorted(
            zip(feat_cols_BD, xgb.feature_importances_), key=lambda t: t[1], reverse=True
        )
        val_metrics["top_features"] = importances[:10]

    logger.info("train_xgb: val macro-F1=%.4f", val_metrics["macro_f1"])
    return xgb, val_metrics


def predict_proba_all(rf, lstm, xgb, session_data):
    """Produce calibrated probability vectors from all three detectors.

    Parameters
    ----------
    session_data : dict
        {"X_AB": array (n, |A|+|B|) for RF,
         "token_ids": array (n, MAX_SEQ_LEN) int32 for BiLSTM,
         "X_BD": array (n, |B|+|D|) for XGBoost}
        All arrays row-aligned to the same session order.

    Returns
    -------
    (P_A, P_B, P_C) : each np.ndarray of shape (n_sessions, K)
    """
    X_AB = np.asarray(session_data["X_AB"], dtype=np.float64)
    token_ids = np.asarray(session_data["token_ids"], dtype=np.int32)
    X_BD = np.asarray(session_data["X_BD"], dtype=np.float64)

    P_A = rf.predict_proba(X_AB)
    P_B = lstm.predict(token_ids, verbose=0)
    P_C = xgb.predict_proba(X_BD)

    return P_A, P_B, P_C
