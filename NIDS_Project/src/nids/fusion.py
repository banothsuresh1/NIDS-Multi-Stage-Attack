"""Stage 7: Adaptive Evidence Fusion.

Combines the three detector probability vectors with the pattern-mining
prior and the graph-based temporal consistency score into a single
fused risk vector R_t.

FUSION WEIGHT INTEGRITY: weights are grid-searched and frozen on the
validation set only. They must never be re-estimated on test or
streaming data — doing so would require test-time ground truth, i.e.
label leakage.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import f1_score

import config

logger = logging.getLogger(__name__)


def fuse(P_A, P_B, P_C, SP_t, TC_t, weights):
    """R_t = w_A*P_A + w_B*P_B + w_C*P_C + w_S*SP_t + w_T*TC_t.

    Parameters
    ----------
    P_A, P_B, P_C : array (n_sessions, K)
    SP_t : array (n_sessions, K) - pattern-mining prior per class
    TC_t : array (n_sessions,) or (n_sessions, 1) - scalar consistency
        score, broadcast across all K classes
    weights : tuple (w_A, w_B, w_C, w_S, w_T)

    Returns
    -------
    R_t : np.ndarray, shape (n_sessions, K)
    """
    w_A, w_B, w_C, w_S, w_T = weights
    P_A = np.asarray(P_A, dtype=np.float64)
    P_B = np.asarray(P_B, dtype=np.float64)
    P_C = np.asarray(P_C, dtype=np.float64)
    SP_t = np.asarray(SP_t, dtype=np.float64)
    TC_t = np.asarray(TC_t, dtype=np.float64).reshape(-1, 1)

    R_t = w_A * P_A + w_B * P_B + w_C * P_C + w_S * SP_t + w_T * TC_t
    return R_t


def grid_search_fusion_weights(P_A_val, P_B_val, P_C_val, SP_val, TC_val, y_val):
    """Grid search fusion weights on the VALIDATION set only, frozen for test.

    Returns
    -------
    best_weights : tuple (w_A, w_B, w_C, w_S, w_T), normalised to sum to 1.0
    """
    grid = config.FUSION_GRID
    best_score = -1.0
    best_weights = config.DEFAULT_FUSION_WEIGHTS
    seen = set()

    for w_A in grid["w_A"]:
        for w_B in grid["w_B"]:
            for w_C in grid["w_C"]:
                for w_S in grid["w_S"]:
                    for w_T in grid["w_T"]:
                        total = w_A + w_B + w_C + w_S + w_T
                        if total <= 0:
                            continue
                        weights = (
                            w_A / total, w_B / total, w_C / total,
                            w_S / total, w_T / total,
                        )
                        if weights in seen:
                            continue
                        seen.add(weights)

                        R_t = fuse(P_A_val, P_B_val, P_C_val, SP_val, TC_val, weights)
                        preds = np.argmax(R_t, axis=1)
                        score = f1_score(y_val, preds, average="macro", zero_division=0)

                        if score > best_score:
                            best_score = score
                            best_weights = weights

    logger.info(
        "grid_search_fusion_weights: evaluated %d weight combinations; "
        "best=%s val_macro_f1=%.4f",
        len(seen), best_weights, best_score,
    )
    return best_weights
