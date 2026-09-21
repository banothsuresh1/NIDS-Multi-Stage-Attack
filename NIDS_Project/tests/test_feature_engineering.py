"""Tests for Stage 3 feature engineering & group assignment."""

from __future__ import annotations

import config
from nids.feature_engineering import assign_feature_groups


def test_assign_feature_groups_matches_target_sizes(small_feat_cols):
    group_A, group_B, group_C, group_D = assign_feature_groups(small_feat_cols)

    assert len(group_A) == config.GROUP_A_SIZE
    assert len(group_B) == config.GROUP_B_SIZE
    assert len(group_C) == config.GROUP_C_SIZE
    assert len(group_D) == config.GROUP_D_SIZE


def test_assign_feature_groups_no_overlap_between_groups(small_feat_cols):
    group_A, group_B, group_C, group_D = assign_feature_groups(small_feat_cols)

    all_assigned = group_A + group_B + group_C + group_D
    assert len(all_assigned) == len(set(all_assigned))


def test_assign_feature_groups_drops_identifier_columns(small_feat_cols):
    feat_cols = list(small_feat_cols) + ["Flow ID", "Source IP", "Destination IP", "Source Port"]
    group_A, group_B, group_C, group_D = assign_feature_groups(feat_cols)

    for group in (group_A, group_B, group_C, group_D):
        assert "Flow ID" not in group
        assert "Source IP" not in group
        assert "Destination IP" not in group
        assert "Source Port" not in group


def test_assign_feature_groups_keeps_destination_port(small_feat_cols):
    group_A, group_B, group_C, group_D = assign_feature_groups(small_feat_cols)
    assert "Destination Port" in group_B


def test_assign_feature_groups_mi_fallback_fills_shortfall():
    import numpy as np

    rng = np.random.default_rng(0)
    # Only 2 of Group C's candidates present -> shortfall must be filled via MI ranking.
    feat_cols = ["Flow Duration", "Idle Mean", "extra_feat_1", "extra_feat_2", "extra_feat_3"]
    X = rng.normal(size=(50, len(feat_cols)))
    y = rng.integers(0, 2, size=50)

    group_A, group_B, group_C, group_D = assign_feature_groups(feat_cols, X_train=X, y_train=y)

    assert "Flow Duration" in group_C
    assert "Idle Mean" in group_C
    assert len(group_C) <= config.GROUP_C_SIZE
