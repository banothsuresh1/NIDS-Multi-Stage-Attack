"""Tests for Stages 9-10: temporal attack-state graph & consistency scoring."""

from __future__ import annotations

import pandas as pd

import config
from nids.attack_graph import TemporalAttackGraph


def test_tc_is_neutral_for_single_event_sessions():
    g = TemporalAttackGraph()
    g.fit([["CONNECTION_ATTEMPT", "SESSION_ESTABLISHED"]])

    tc = g.compute_tc(["CONNECTION_ATTEMPT"], [pd.Timestamp("2017-07-03 09:00:00")])
    assert tc == 0.5


def test_novel_edge_assigned_novel_edge_weight():
    g = TemporalAttackGraph()
    g.fit([["CONNECTION_ATTEMPT", "SESSION_ESTABLISHED"]])

    updates = g.update(["AUTH_FAIL", "POST_AUTH_ACTIVITY"])  # never seen in training
    u, v, w, is_novel = updates[0]
    assert is_novel is True
    assert w == config.NOVEL_EDGE_WEIGHT
    assert ("AUTH_FAIL", "POST_AUTH_ACTIVITY") in g.anomalous_transitions


def test_ema_update_formula_for_known_edge():
    g = TemporalAttackGraph(rho=0.9)
    g.fit([["CONNECTION_ATTEMPT", "SESSION_ESTABLISHED"]])

    w_before = g.G["CONNECTION_ATTEMPT"]["SESSION_ESTABLISHED"]["weight"]
    g.update(["CONNECTION_ATTEMPT", "SESSION_ESTABLISHED"])
    w_after = g.G["CONNECTION_ATTEMPT"]["SESSION_ESTABLISHED"]["weight"]

    expected = 0.9 * w_before + 0.1 * 1.0
    assert abs(w_after - expected) < 1e-9


def test_graph_fitted_on_training_data_only():
    g = TemporalAttackGraph()
    training_sessions = [
        ["CONNECTION_ATTEMPT", "SESSION_ESTABLISHED", "DATA_TRANSFER"],
        ["CONNECTION_ATTEMPT", "SESSION_ESTABLISHED", "DATA_TRANSFER"],
    ]
    g.fit(training_sessions)

    assert g.G["CONNECTION_ATTEMPT"]["SESSION_ESTABLISHED"]["weight"] == 1.0
    assert ("CONNECTION_ATTEMPT", "SESSION_ESTABLISHED") in g._known_edges
    # An edge never observed in training must start at weight 0.
    assert g.G["SCAN_ACTIVITY"]["DOS_INDICATOR"]["weight"] == 0.0


def test_repeated_tokens_count_as_separate_transitions_not_self_loop():
    g = TemporalAttackGraph()
    g.fit([["AUTH_FAIL", "AUTH_FAIL", "AUTH_FAIL"]])

    # AUTH_FAIL -> AUTH_FAIL transition should have probability 1.0 (2 out of 2 transitions from AUTH_FAIL).
    assert g.G["AUTH_FAIL"]["AUTH_FAIL"]["weight"] == 1.0

    ts = [pd.Timestamp("2017-07-03 09:00:00") + pd.Timedelta(seconds=i) for i in range(3)]
    tc = g.compute_tc(["AUTH_FAIL", "AUTH_FAIL", "AUTH_FAIL"], ts)
    assert tc > 0  # both transitions contributed independently


def test_top_transitions_sorted_descending():
    g = TemporalAttackGraph()
    g.fit([
        ["CONNECTION_ATTEMPT", "SESSION_ESTABLISHED"],
        ["CONNECTION_ATTEMPT", "SESSION_ESTABLISHED"],
        ["SCAN_ACTIVITY", "DOS_INDICATOR"],
    ])
    top = g.top_transitions(k=5)
    weights = [w for _, _, w in top]
    assert weights == sorted(weights, reverse=True)
