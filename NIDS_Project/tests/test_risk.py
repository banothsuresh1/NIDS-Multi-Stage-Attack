"""Tests for Stage 11 adaptive risk decision."""

from __future__ import annotations

import numpy as np

import config
from nids.risk import AdaptiveRiskDecider


def _synthetic_val(n, K, rng):
    R_t = rng.random((n, K))
    SP = rng.random((n, K))
    TC = rng.random(n)
    G_w = rng.random(n)
    inv_dt = rng.random(n)
    y = rng.integers(0, 2, size=n)
    return R_t, SP, TC, G_w, inv_dt, y


def test_fit_uses_only_validation_data_and_learns_coefficients():
    rng = np.random.default_rng(0)
    R_t, SP, TC, G_w, inv_dt, y = _synthetic_val(60, 5, rng)

    decider = AdaptiveRiskDecider()
    decider.fit(R_t, SP, TC, G_w, inv_dt, y)

    assert decider._fitted is True
    # Coefficients must be learned (non-trivial), not a fixed/manual default.
    assert not np.allclose(decider.model.coef_, 0.0)


def test_risk_score_in_unit_interval():
    rng = np.random.default_rng(1)
    K = 5
    R_t, SP, TC, G_w, inv_dt, y = _synthetic_val(60, K, rng)
    decider = AdaptiveRiskDecider()
    decider.fit(R_t, SP, TC, G_w, inv_dt, y)

    R_test, SP_test, TC_test, G_w_test, inv_dt_test, _ = _synthetic_val(20, K, rng)
    risk_score, alert_tier = decider.predict(R_test, SP_test, TC_test, G_w_test, inv_dt_test)

    assert (risk_score >= 0).all() and (risk_score <= 1).all()


def test_alert_tiers_mapped_to_thresholds():
    decider = AdaptiveRiskDecider(benign_thresh=0.35, suspicious_thresh=0.75)
    rng = np.random.default_rng(2)
    R_t, SP, TC, G_w, inv_dt, y = _synthetic_val(80, 4, rng)
    decider.fit(R_t, SP, TC, G_w, inv_dt, y)

    R_test, SP_test, TC_test, G_w_test, inv_dt_test, _ = _synthetic_val(50, 4, rng)
    risk_score, alert_tier = decider.predict(R_test, SP_test, TC_test, G_w_test, inv_dt_test)

    for score, tier in zip(risk_score, alert_tier):
        if score < 0.35:
            assert tier == "BENIGN"
        elif score < 0.75:
            assert tier == "SUSPICIOUS"
        else:
            assert tier == "ATTACK"


def test_predict_before_fit_raises():
    decider = AdaptiveRiskDecider()
    rng = np.random.default_rng(3)
    R_t, SP, TC, G_w, inv_dt, _ = _synthetic_val(5, 3, rng)
    try:
        decider.predict(R_t, SP, TC, G_w, inv_dt)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_compute_inv_dt_guards_division_by_zero():
    inv_dt = AdaptiveRiskDecider.compute_inv_dt(np.array([0.0, 1.0, 5.0]), delta_t_mean_train=2.0)
    assert np.isfinite(inv_dt).all()
    # Simultaneous events (delta_t=0) should yield the LARGEST inv_dt (highest urgency).
    assert inv_dt[0] > inv_dt[1] > inv_dt[2]
