"""Tests for Stage 8 frequent & sequential pattern mining."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nids.pattern_mining import compute_sp_score, mine_fp_growth, mine_prefixspan


def test_mine_fp_growth_finds_common_itemset(sample_token_sequences):
    session_token_sets = [set(seq) for seq in sample_token_sequences]

    itemsets = mine_fp_growth(session_token_sets, min_support=0.2)

    assert len(itemsets) > 0
    assert {"support", "itemsets"}.issubset(itemsets.columns)
    assert (itemsets["support"] >= 0.2).all()


def test_mine_fp_growth_with_class_support(sample_token_sequences):
    session_token_sets = [set(seq) for seq in sample_token_sequences]
    session_labels = [0, 0, 1, 2, 3, 3]
    K = 4

    itemsets = mine_fp_growth(session_token_sets, min_support=0.2, session_labels=session_labels, K=K)

    assert "class_support" in itemsets.columns
    for cs in itemsets["class_support"]:
        assert cs.shape == (K,)
        assert abs(cs.sum() - 1.0) < 1e-9


def test_mine_prefixspan_respects_min_support():
    base = pd.Timestamp("2017-07-03 09:00:00")
    seqs = [
        [("A", base), ("B", base + pd.Timedelta(seconds=1))],
        [("A", base), ("B", base + pd.Timedelta(seconds=1))],
        [("A", base), ("B", base + pd.Timedelta(seconds=1))],
        [("C", base)],
    ]
    patterns = mine_prefixspan(seqs, min_support=0.5, max_gap=8)
    pattern_tuples = {p["pattern"] for p in patterns}
    assert ("A",) in pattern_tuples
    assert ("A", "B") in pattern_tuples
    assert ("C",) not in pattern_tuples  # support 0.25 < 0.5


def test_mine_prefixspan_respects_max_gap():
    base = pd.Timestamp("2017-07-03 09:00:00")
    # gap of 100s far exceeds max_gap=8 -> "A","B" should not be considered adjacent-frequent
    seqs = [
        [("A", base), ("B", base + pd.Timedelta(seconds=100))],
        [("A", base), ("B", base + pd.Timedelta(seconds=100))],
    ]
    patterns = mine_prefixspan(seqs, min_support=0.5, max_gap=8)
    pattern_tuples = {p["pattern"] for p in patterns}
    assert ("A", "B") not in pattern_tuples


def test_compute_sp_score_shape_and_matches_pattern():
    K = 3
    session = {
        "token_seq": ["CONNECTION_ATTEMPT", "AUTH_FAIL", None, None],
        "timestamps": [pd.Timestamp("2017-07-03 09:00:00"), pd.Timestamp("2017-07-03 09:00:02"), pd.NaT, pd.NaT],
    }

    itemsets = pd.DataFrame(
        {
            "support": [0.5],
            "itemsets": [frozenset({"CONNECTION_ATTEMPT", "AUTH_FAIL"})],
            "class_support": [np.array([0.1, 0.8, 0.1])],
        }
    )
    sequential_patterns = [
        {"pattern": ("CONNECTION_ATTEMPT", "AUTH_FAIL"), "support": 0.4, "class_support": np.array([0.0, 1.0, 0.0])}
    ]

    sp_vector = compute_sp_score(session, itemsets, sequential_patterns, K)

    assert sp_vector.shape == (K,)
    assert abs(sp_vector.sum() - 1.0) < 1e-9
    assert sp_vector[1] > sp_vector[0]  # class 1 dominates both matched patterns


def test_compute_sp_score_returns_zeros_for_no_match():
    K = 3
    session = {"token_seq": ["SCAN_ACTIVITY", None], "timestamps": [pd.Timestamp("2017-07-03 09:00:00"), pd.NaT]}
    itemsets = pd.DataFrame({"support": [], "itemsets": [], "class_support": []})
    sp_vector = compute_sp_score(session, itemsets, [], K)
    assert np.allclose(sp_vector, np.zeros(K))
