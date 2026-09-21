"""Stage 3: Feature Engineering & Semantic Group Assignment.

Splits the numeric feature matrix into four domain-knowledge-driven
groups used by different downstream models:
    Group A (32) - flow-statistical            -> Random Forest
    Group B (18) - protocol/communication       -> shared RF + XGBoost
    Group C (15) - derived temporal             -> BiLSTM
    Group D (13) - TCP behavioral flag counts   -> XGBoost

Identifier columns that would let a model memorise a host rather than
generalise (Flow ID, Source IP, Destination IP, Source Port) are removed
unconditionally; Destination Port is kept because it carries protocol
semantics (e.g. 22=SSH, 3389=RDP).
"""

from __future__ import annotations

import logging

import numpy as np

import config

logger = logging.getLogger(__name__)

# Identifier-like columns that must never enter a feature group even if
# they slipped through get_feature_cols (e.g. an integer-encoded IP).
_ALWAYS_DROP = {"Flow ID", "Source IP", "Destination IP", "Source Port"}

# Ordered, non-overlapping candidate pools built from the standard
# CICFlowMeter / CIC-IDS2017 column vocabulary (after `.str.strip()`).
# Extra candidates beyond the target size act as fallback if a preferred
# column is absent from a particular CSV variant.
_GROUP_A_CANDIDATES = [
    "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min",
    "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
]

_GROUP_B_CANDIDATES = [
    "Destination Port", "Protocol",
    "Flow IAT Max", "Flow IAT Min",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward",
    "Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
]

_GROUP_C_CANDIDATES = [
    "Flow Duration",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Flow IAT Mean", "Flow IAT Std",
]

_GROUP_D_CANDIDATES = [
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Down/Up Ratio",
]

_TARGET_SIZES = {
    "A": config.GROUP_A_SIZE,
    "B": config.GROUP_B_SIZE,
    "C": config.GROUP_C_SIZE,
    "D": config.GROUP_D_SIZE,
}


def _mi_fill(remaining_cols, target_size, X_train, y_train, feat_col_index):
    """Rank remaining columns by mutual information with the label and take the top-k."""
    from sklearn.feature_selection import mutual_info_classif

    if not remaining_cols or X_train is None or y_train is None:
        return []
    idx = [feat_col_index[c] for c in remaining_cols]
    mi = mutual_info_classif(
        X_train[:, idx], y_train, discrete_features=False, random_state=config.SEED
    )
    ranked = [c for _, c in sorted(zip(mi, remaining_cols), reverse=True)]
    return ranked[:target_size]


def assign_feature_groups(feat_cols, X_train=None, y_train=None):
    """Assign feature columns into Groups A/B/C/D by semantic candidate lists.

    Parameters
    ----------
    feat_cols : list[str]
        Numeric feature columns from ``preprocessing.get_feature_cols``.
    X_train, y_train : optional
        Training matrix (aligned column-for-column with ``feat_cols``) and
        label vector, used only as a mutual-information fallback to fill a
        group up to its target size when the semantic candidate list comes
        up short on this particular CSV variant.

    Returns
    -------
    (group_A, group_B, group_C, group_D) : tuple[list[str], ...]
    """
    available = [c for c in feat_cols if c not in _ALWAYS_DROP]
    available_set = set(available)
    feat_col_index = {c: i for i, c in enumerate(feat_cols)}

    used: set[str] = set()
    groups: dict[str, list[str]] = {}
    for name, candidates in (
        ("A", _GROUP_A_CANDIDATES),
        ("B", _GROUP_B_CANDIDATES),
        ("C", _GROUP_C_CANDIDATES),
        ("D", _GROUP_D_CANDIDATES),
    ):
        matched = [c for c in candidates if c in available_set and c not in used]
        matched = matched[: _TARGET_SIZES[name]]
        used.update(matched)
        groups[name] = matched

    for name in ("A", "B", "C", "D"):
        target = _TARGET_SIZES[name]
        shortfall = target - len(groups[name])
        if shortfall > 0:
            remaining = [c for c in available if c not in used]
            logger.warning(
                "Feature group %s short by %d columns (matched %d/%d); "
                "attempting mutual-information fallback",
                name, shortfall, len(groups[name]), target,
            )
            fill = _mi_fill(remaining, shortfall, X_train, y_train, feat_col_index)
            groups[name].extend(fill)
            used.update(fill)
            if len(groups[name]) < target:
                logger.warning(
                    "Feature group %s still short after MI fallback: %d/%d "
                    "(no training data supplied or too few candidate columns remain)",
                    name, len(groups[name]), target,
                )

    for name in ("A", "B", "C", "D"):
        logger.info("Group %s (%d/%d): %s", name, len(groups[name]), _TARGET_SIZES[name], groups[name])

    return groups["A"], groups["B"], groups["C"], groups["D"]
