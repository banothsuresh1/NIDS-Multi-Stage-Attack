"""Tests for Stage 7 adaptive evidence fusion."""

from __future__ import annotations

import numpy as np

import config
from nids.fusion import fuse, grid_search_fusion_weights


def _random_probs(rng, n, K):
    raw = rng.random(size=(n, K))
    return raw / raw.sum(axis=1, keepdims=True)


def test_fusion_weights_sum_to_one():
    rng = np.random.default_rng(0)
    n, K = 40, 5
    P_A, P_B, P_C = (_random_probs(rng, n, K) for _ in range(3))
    SP = _random_probs(rng, n, K)
    TC = rng.random(n)
    y = rng.integers(0, K, size=n)

    weights = grid_search_fusion_weights(P_A, P_B, P_C, SP, TC, y)
    assert abs(sum(weights) - 1.0) < 1e-9


def test_fuse_output_shape():
    rng = np.random.default_rng(1)
    n, K = 10, 5
    P_A, P_B, P_C = (_random_probs(rng, n, K) for _ in range(3))
    SP = _random_probs(rng, n, K)
    TC = rng.random(n)

    R_t = fuse(P_A, P_B, P_C, SP, TC, config.DEFAULT_FUSION_WEIGHTS)
    assert R_t.shape == (n, K)


def test_fuse_tc_broadcasts_across_classes():
    n, K = 3, 4
    P_A = np.zeros((n, K))
    P_B = np.zeros((n, K))
    P_C = np.zeros((n, K))
    SP = np.zeros((n, K))
    TC = np.array([1.0, 0.5, 0.0])
    weights = (0.0, 0.0, 0.0, 0.0, 1.0)

    R_t = fuse(P_A, P_B, P_C, SP, TC, weights)
    for k in range(K):
        assert np.allclose(R_t[:, k], TC)


def test_calibrated_weights_frozen_not_updated_during_test(monkeypatch):
    """fuse() must never itself call grid_search_fusion_weights or read test labels."""
    import inspect

    from nids import fusion as fusion_module

    fuse_source = inspect.getsource(fusion_module.fuse)
    assert "grid_search_fusion_weights" not in fuse_source
    assert "y_test" not in fuse_source and "y_val" not in fuse_source
