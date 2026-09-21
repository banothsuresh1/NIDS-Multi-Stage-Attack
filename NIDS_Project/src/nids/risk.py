"""Stage 11: Adaptive Risk Decision.

A logistic-regression meta-learner turns the fused evidence
(R_t, SP_t, TC_t, graph weight, inverse inter-arrival time) into a
single [0, 1] risk score and a 3-tier alert (BENIGN / SUSPICIOUS /
ATTACK). Fitted on validation-set predictions and ground truth only —
this and fusion-weight calibration are the only places labels are used
after Stage 1 — and its thresholds are then frozen for test/streaming.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.linear_model import LogisticRegression

import config

logger = logging.getLogger(__name__)


class AdaptiveRiskDecider:
    def __init__(
        self,
        benign_thresh=config.RISK_BENIGN_THRESH,
        suspicious_thresh=config.RISK_SUSPICIOUS_THRESH,
        meta_lr_c=config.META_LR_C,
    ):
        self.benign_thresh = benign_thresh
        self.suspicious_thresh = suspicious_thresh
        self.meta_lr_c = meta_lr_c
        self.model = LogisticRegression(C=meta_lr_c, max_iter=1000, random_state=config.SEED)
        self._fitted = False

    @staticmethod
    def _stack_features(R_t, SP_t, TC_t, G_w, inv_dt):
        R_t = np.asarray(R_t, dtype=np.float64)
        SP_t = np.asarray(SP_t, dtype=np.float64)
        TC_t = np.asarray(TC_t, dtype=np.float64).reshape(-1, 1)
        G_w = np.asarray(G_w, dtype=np.float64).reshape(-1, 1)
        inv_dt = np.asarray(inv_dt, dtype=np.float64).reshape(-1, 1)
        return np.hstack([R_t, SP_t, TC_t, G_w, inv_dt])

    def fit(self, R_t_val, SP_val, TC_val, G_w_val, inv_dt_val, y_val):
        """Fit the meta-learner on VALIDATION set predictions and ground truth ONLY.

        ``y_val`` must already be binarised: 1 = any attack (non-BENIGN)
        class, 0 = BENIGN. Never call this with training or test labels.

        If the validation split happens to contain only one class (e.g. a
        very small or attack-free validation day), a logistic regression
        cannot be fit; falls back to a constant classifier that always
        predicts that single observed class, rather than crashing.
        """
        X = self._stack_features(R_t_val, SP_val, TC_val, G_w_val, inv_dt_val)
        y = np.asarray(y_val).astype(int)

        if len(np.unique(y)) < 2:
            logger.warning(
                "AdaptiveRiskDecider.fit: validation labels contain only one class "
                "(%s); falling back to a constant classifier instead of logistic regression.",
                np.unique(y),
            )
            from sklearn.dummy import DummyClassifier

            self.model = DummyClassifier(strategy="constant", constant=int(y[0]))
            self.model.fit(X, y)
        else:
            self.model.fit(X, y)

        self._fitted = True
        logger.info(
            "AdaptiveRiskDecider.fit: meta-learner coefficients learned on %d val sessions "
            "(positive rate=%.3f)", len(y), float(y.mean()) if len(y) else 0.0,
        )
        return self

    def predict(self, R_t, SP_t, TC_t, G_w, inv_dt):
        """Predict risk_score in [0, 1] and the alert tier, using thresholds frozen at fit time."""
        if not self._fitted:
            raise RuntimeError("AdaptiveRiskDecider.predict called before fit()")

        X = self._stack_features(R_t, SP_t, TC_t, G_w, inv_dt)
        risk_score = self.model.predict_proba(X)[:, 1]

        alert_tier = np.where(
            risk_score < self.benign_thresh,
            "BENIGN",
            np.where(risk_score < self.suspicious_thresh, "SUSPICIOUS", "ATTACK"),
        )
        return risk_score, alert_tier

    @staticmethod
    def compute_inv_dt(delta_t, delta_t_mean_train, eps=1e-3):
        """Delta_t_norm = (delta_t + eps) / delta_t_mean_train; inv_dt = 1 / delta_t_norm.

        Prevents division-by-zero for simultaneous DoS events (delta_t -> 0).
        """
        delta_t = np.asarray(delta_t, dtype=np.float64)
        denom = max(float(delta_t_mean_train), eps)
        delta_t_norm = (delta_t + eps) / denom
        return 1.0 / delta_t_norm
