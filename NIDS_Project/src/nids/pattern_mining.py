"""Stage 8: Frequent & Sequential Pattern Mining (FP-Growth + PrefixSpan).

Discovers co-occurrence and ordered attack-behavior patterns in
behavioral token sequences, mined from TRAINING sessions only (original,
un-augmented). SMOTE+ENN synthetic flows have no valid timestamps or
behavioral tokens and must never reach pattern mining.

Pattern discovery itself is unsupervised (token co-occurrence/order
only). Ground-truth training labels are used only as a post-hoc
annotation step, attaching a per-class support distribution to each
discovered pattern so ``compute_sp_score`` can turn "this session
matches these patterns" into a per-class prior vector for fusion.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder

import config

logger = logging.getLogger(__name__)


def _attach_class_support(patterns, item_sets, session_labels, K):
    """For each pattern (a set or tuple of tokens), compute how its
    containing training sessions distribute across the K classes,
    normalised to sum to 1 per pattern (a per-pattern class prior).
    """
    if session_labels is None:
        return [None] * len(patterns)

    labels = np.asarray(session_labels)
    class_supports = []
    for pattern, contains in zip(patterns, item_sets):
        matched_labels = labels[contains]
        if len(matched_labels) == 0:
            class_supports.append(np.zeros(K, dtype=np.float64))
            continue
        counts = np.bincount(matched_labels, minlength=K).astype(np.float64)
        class_supports.append(counts / counts.sum())
    return class_supports


def mine_fp_growth(session_token_sets, min_support=config.FP_GROWTH_MIN_SUPPORT,
                    session_labels=None, K=None):
    """Mine frequent co-occurring token itemsets via FP-Growth.

    Parameters
    ----------
    session_token_sets : list[set[str]]
        One unordered set of unique tokens per TRAINING session.
    min_support : float
    session_labels : array-like of int, optional
        Encoded class id per session (same order/length as
        ``session_token_sets``), used only to annotate each itemset with
        a per-class support distribution for ``compute_sp_score``.
    K : int, required if session_labels is given.

    Returns
    -------
    pd.DataFrame with columns: support, itemsets, [class_support]
    """
    te = TransactionEncoder()
    te_ary = te.fit(session_token_sets).transform(session_token_sets)
    onehot_df = pd.DataFrame(te_ary, columns=te.columns_)

    itemsets_df = fpgrowth(onehot_df, min_support=min_support, use_colnames=True)
    itemsets_df = itemsets_df.sort_values("support", ascending=False).reset_index(drop=True)

    if session_labels is not None:
        contains = [
            onehot_df[list(itemset)].all(axis=1).to_numpy()
            for itemset in itemsets_df["itemsets"]
        ]
        itemsets_df["class_support"] = _attach_class_support(
            itemsets_df["itemsets"].tolist(), contains, session_labels, K
        )

    logger.info("mine_fp_growth: %d frequent itemsets (min_support=%.3f)", len(itemsets_df), min_support)
    return itemsets_df


def _contains_pattern_with_gap(events, pattern, max_gap):
    """Greedy earliest-match containment check: does ``events`` (a
    chronological list of (token, timestamp)) contain ``pattern`` (a
    tuple of tokens) as a subsequence where every two consecutive
    matched events are at most ``max_gap`` seconds apart?

    Uses the standard greedy earliest-occurrence strategy: once a
    pattern symbol is matched, later occurrences of the same symbol
    without matching the gap constraint are skipped rather than
    restarting the whole match, keeping this O(len(events)).
    """
    if not pattern:
        return True
    pi = 0
    last_time = None
    for tok, ts in events:
        if tok != pattern[pi]:
            continue
        if pi > 0 and last_time is not None and pd.notna(ts) and pd.notna(last_time):
            gap = (ts - last_time) / np.timedelta64(1, "s")
            if gap > max_gap:
                continue
        pi += 1
        last_time = ts
        if pi == len(pattern):
            return True
    return False


def mine_prefixspan(session_token_sequences, min_support=config.PREFIXSPAN_MIN_SUPPORT,
                     max_gap=config.PREFIXSPAN_MAX_GAP, max_pattern_len=5,
                     session_labels=None, K=None):
    """Mine frequent ordered token patterns with a max-gap temporal constraint.

    A level-wise (Apriori-style) PrefixSpan variant: candidate (k+1)-length
    patterns extend only patterns that were already frequent at length k,
    and containment additionally requires every two consecutive matched
    events be within ``max_gap`` seconds of each other.

    Parameters
    ----------
    session_token_sequences : list[list[tuple[str, pd.Timestamp]]]
        One chronological, un-padded (token, timestamp) event list per
        TRAINING session.
    min_support : float
    max_gap : float
        Max seconds between consecutive matched events (config.PREFIXSPAN_MAX_GAP).
    max_pattern_len : int
        Search depth cap (vocabulary has only 10 symbols, so this bounds
        the otherwise combinatorial level-wise search).
    session_labels, K : see ``mine_fp_growth``.

    Returns
    -------
    list[dict] each: {"pattern": tuple[str], "support": float, "count": int,
                       "class_support": np.ndarray | None}
    """
    n_total = len(session_token_sequences)
    if n_total == 0:
        return []
    min_count = max(1, int(np.ceil(min_support * n_total)))

    def _support_mask(pattern):
        return np.array(
            [_contains_pattern_with_gap(events, pattern, max_gap) for events in session_token_sequences]
        )

    # Derive the alphabet from the data itself (not hardcoded to config.VOCAB)
    # so this stays a general, self-contained sequential-pattern miner; in
    # production this alphabet is always the 10-token behavioral vocabulary.
    alphabet = sorted({tok for events in session_token_sequences for tok, _ in events})

    frequent_patterns = []
    current_level = []
    for tok in alphabet:
        mask = _support_mask((tok,))
        count = int(mask.sum())
        if count >= min_count:
            current_level.append((tok,))
            frequent_patterns.append({"pattern": (tok,), "support": count / n_total, "count": count, "_mask": mask})

    length = 1
    while current_level and length < max_pattern_len:
        next_level = []
        for pat in current_level:
            for tok in alphabet:
                new_pat = pat + (tok,)
                mask = _support_mask(new_pat)
                count = int(mask.sum())
                if count >= min_count:
                    next_level.append(new_pat)
                    frequent_patterns.append(
                        {"pattern": new_pat, "support": count / n_total, "count": count, "_mask": mask}
                    )
        current_level = next_level
        length += 1

    if session_labels is not None:
        labels = np.asarray(session_labels)
        for p in frequent_patterns:
            matched_labels = labels[p["_mask"]]
            if len(matched_labels) == 0:
                p["class_support"] = np.zeros(K, dtype=np.float64)
            else:
                counts = np.bincount(matched_labels, minlength=K).astype(np.float64)
                p["class_support"] = counts / counts.sum()
    else:
        for p in frequent_patterns:
            p["class_support"] = None

    for p in frequent_patterns:
        del p["_mask"]

    frequent_patterns.sort(key=lambda d: d["support"], reverse=True)
    logger.info(
        "mine_prefixspan: %d frequent sequential patterns (min_support=%.3f, max_gap=%ds)",
        len(frequent_patterns), min_support, max_gap,
    )
    return frequent_patterns


def compute_sp_score(session, frequent_itemsets, sequential_patterns, K):
    """Aggregate matched-pattern class priors into a per-class score vector.

    Parameters
    ----------
    session : dict
        A record from ``events.encode_sessions`` (has "token_seq" and
        "timestamps"; padding entries are ``None``/``NaT`` and ignored).
    frequent_itemsets : pd.DataFrame
        Output of ``mine_fp_growth`` with a "class_support" column.
    sequential_patterns : list[dict]
        Output of ``mine_prefixspan`` with a "class_support" key.
    K : int

    Returns
    -------
    sp_vector : np.ndarray, shape (K,)
        Support-weighted, class-normalised pattern-match prior. All
        zeros if the session matches no mined pattern.
    """
    tokens = [t for t in session["token_seq"] if t is not None]
    token_set = set(tokens)
    events = [
        (t, ts) for t, ts in zip(session["token_seq"], session["timestamps"]) if t is not None
    ]

    sp_vector = np.zeros(K, dtype=np.float64)

    if frequent_itemsets is not None and len(frequent_itemsets) and "class_support" in frequent_itemsets.columns:
        for itemset, support, class_support in zip(
            frequent_itemsets["itemsets"], frequent_itemsets["support"], frequent_itemsets["class_support"]
        ):
            if class_support is None:
                continue
            if set(itemset).issubset(token_set):
                sp_vector += support * class_support

    if sequential_patterns:
        for p in sequential_patterns:
            if p.get("class_support") is None:
                continue
            if _contains_pattern_with_gap(events, p["pattern"], config.PREFIXSPAN_MAX_GAP):
                sp_vector += p["support"] * p["class_support"]

    total = sp_vector.sum()
    if total > 0:
        sp_vector = sp_vector / total
    return sp_vector
