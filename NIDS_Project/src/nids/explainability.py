"""Stage 13: Explainability & Evidence Chain.

Builds a three-modality, analyst-facing explanation for one alerted
session: TreeSHAP for RF/XGBoost, DeepSHAP (with an occlusion-based
fallback) for the BiLSTM, and the concrete attack-state graph path the
session walked, annotated with edge weights and novel-transition flags.
"""

from __future__ import annotations

import logging

import numpy as np

import config

logger = logging.getLogger(__name__)

_ATTACK_FAMILY_RULES = [
    ({"DOS_INDICATOR"}, "Denial of Service"),
    ({"AUTH_FAIL"}, "Brute Force / Credential Access"),
    ({"SCAN_ACTIVITY"}, "Reconnaissance / Port Scan"),
    ({"POST_AUTH_ACTIVITY"}, "Post-Exploitation / Data Exfiltration"),
    ({"FORCED_TERMINATION"}, "Connection Disruption"),
]

_KILL_CHAIN_STAGE = {
    "SCAN_ACTIVITY": "Reconnaissance",
    "CONNECTION_ATTEMPT": "Delivery",
    "SESSION_ESTABLISHED": "Delivery",
    "AUTH_FAIL": "Exploitation",
    "AUTH_SUCCESS": "Installation",
    "DATA_TRANSFER": "Actions on Objectives",
    "POST_AUTH_ACTIVITY": "Actions on Objectives",
    "DOS_INDICATOR": "Actions on Objectives (Denial of Service)",
    "CONNECTION_TERMINATION": "Command & Control Teardown",
    "FORCED_TERMINATION": "Command & Control Teardown",
}

_RECOMMENDED_ACTION = {
    "ATTACK": "Isolate the host, block the peer, and escalate to SOC L2 for immediate investigation.",
    "SUSPICIOUS": "Increase monitoring on this session's endpoints and correlate with other alerts before escalating.",
    "BENIGN": "No action required; retain in the audit log.",
}


def _tree_shap_top_features(model, X_row, feat_names, predicted_class, top_k):
    """Top-k |SHAP value| features for ``predicted_class`` from a tree model."""
    import shap

    try:
        background = X_row  # TreeExplainer's interventional/tree-path-dependent mode needs no background
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_row.reshape(1, -1))

        if isinstance(sv, list):
            class_idx = min(predicted_class, len(sv) - 1)
            values = np.asarray(sv[class_idx])[0]
        else:
            arr = np.asarray(sv)
            if arr.ndim == 3:  # (n_samples, n_features, n_classes)
                class_idx = min(predicted_class, arr.shape[-1] - 1)
                values = arr[0, :, class_idx]
            else:  # (n_samples, n_features) - binary/regression-style output
                values = arr[0]

        if len(values) != len(feat_names):
            logger.warning(
                "TreeSHAP value length (%d) != feat_names length (%d); truncating to min",
                len(values), len(feat_names),
            )
        n = min(len(values), len(feat_names))
        ranked = sorted(zip(feat_names[:n], values[:n]), key=lambda t: abs(t[1]), reverse=True)
        return [(f, float(v)) for f, v in ranked[:top_k]]
    except Exception as exc:  # pragma: no cover - SHAP/backend version drift
        logger.warning("TreeSHAP failed (%s); returning empty feature attribution", exc)
        return []


def _lstm_token_attributions(lstm, token_ids, predicted_class, background_tokens, top_k):
    """DeepSHAP token attribution, falling back to leave-one-out occlusion.

    ``shap.DeepExplainer`` is sensitive to the installed TensorFlow
    version, so a failure there degrades gracefully to occlusion: mask
    each real token position with the padding id and measure the drop in
    the predicted class's probability.
    """
    real_positions = [i for i, t in enumerate(token_ids) if t != config.PAD_TOKEN_ID]
    if not real_positions:
        return []

    try:
        import shap

        bg = background_tokens if background_tokens is not None else token_ids.reshape(1, -1)
        explainer = shap.DeepExplainer(lstm, bg)
        sv = explainer.shap_values(token_ids.reshape(1, -1))
        values = np.asarray(sv)
        if values.ndim == 3:  # (n_samples, seq_len, n_classes)
            class_idx = min(predicted_class, values.shape[-1] - 1)
            per_position = values[0, :, class_idx]
        else:
            per_position = values[0]
        attributions = [
            (config.ID2TOKEN.get(int(token_ids[i]), "PAD"), i, float(per_position[i]))
            for i in real_positions
        ]
    except Exception as exc:  # pragma: no cover - SHAP/backend version drift
        logger.warning("DeepSHAP failed (%s); falling back to occlusion-based attribution", exc)
        base_probs = lstm.predict(token_ids.reshape(1, -1), verbose=0)[0]
        base_p = base_probs[predicted_class]
        attributions = []
        for i in real_positions:
            occluded = token_ids.copy()
            occluded[i] = config.PAD_TOKEN_ID
            occ_probs = lstm.predict(occluded.reshape(1, -1), verbose=0)[0]
            drop = float(base_p - occ_probs[predicted_class])
            attributions.append((config.ID2TOKEN.get(int(token_ids[i]), "PAD"), i, drop))

    attributions.sort(key=lambda t: abs(t[2]), reverse=True)
    return attributions[:top_k]


def _graph_path_evidence(attack_graph, token_seq):
    toks = [t for t in token_seq if t is not None]
    graph_path, novel_edges = [], []
    for u, v in zip(toks[:-1], toks[1:]):
        w = attack_graph.G[u][v]["weight"] if attack_graph.G.has_edge(u, v) else attack_graph.novel_edge_weight
        graph_path.append((u, v, float(w)))
        if abs(w - attack_graph.novel_edge_weight) < 1e-9:
            novel_edges.append((u, v))
    return graph_path, novel_edges


def _infer_attack_family(token_seq):
    tokens_present = {t for t in token_seq if t is not None}
    for trigger_tokens, family in _ATTACK_FAMILY_RULES:
        if trigger_tokens & tokens_present:
            return family
    if "DATA_TRANSFER" in tokens_present:
        return "Suspicious Data Transfer"
    return "Unclassified / Generic Intrusion"


def _infer_kill_chain_stage(token_seq):
    toks = [t for t in token_seq if t is not None]
    if not toks:
        return "Unknown"
    stage_order = list(_KILL_CHAIN_STAGE.values())
    best_stage, best_rank = "Unknown", -1
    for t in toks:
        stage = _KILL_CHAIN_STAGE.get(t)
        if stage is None:
            continue
        rank = stage_order.index(stage) if stage in stage_order else -1
        # Prefer the LAST token's stage as the most-advanced stage reached.
        best_stage, best_rank = stage, rank
    return best_stage


def generate_evidence_chain(session, rf, xgb, lstm, attack_graph, feat_names, top_k=5):
    """Build the full evidence chain for one alerted session.

    Parameters
    ----------
    session : dict
        {"session_id", "risk_score", "alert_tier", "predicted_class" (int),
         "X_AB" (np.ndarray, RF feature row), "X_BD" (np.ndarray, XGB
         feature row), "token_ids" (np.ndarray, MAX_SEQ_LEN),
         "token_seq" (list[str|None]),
         "background_tokens" (optional np.ndarray for DeepSHAP)}
    rf, xgb, lstm : frozen, already-trained models
    attack_graph : TemporalAttackGraph
    feat_names : dict {"rf": list[str], "xgb": list[str]}
        RF (Group A+B) and XGBoost (Group B+D) feature names — kept
        separate because the two models consume different feature sets.
    top_k : int

    Returns
    -------
    dict matching the spec's evidence_report structure.
    """
    predicted_class = int(session.get("predicted_class", 0))
    token_seq = session["token_seq"]

    top_features_rf = _tree_shap_top_features(
        rf, np.asarray(session["X_AB"], dtype=np.float64), feat_names["rf"], predicted_class, top_k
    )
    top_features_xgb = _tree_shap_top_features(
        xgb, np.asarray(session["X_BD"], dtype=np.float64), feat_names["xgb"], predicted_class, top_k
    )
    lstm_token_attributions = _lstm_token_attributions(
        lstm,
        np.asarray(session["token_ids"], dtype=np.int32),
        predicted_class,
        session.get("background_tokens"),
        top_k,
    )
    graph_path, novel_edges = _graph_path_evidence(attack_graph, token_seq)

    attack_family = _infer_attack_family(token_seq)
    kill_chain_stage = _infer_kill_chain_stage(token_seq)
    alert_tier = session.get("alert_tier", "SUSPICIOUS")
    recommended_action = _RECOMMENDED_ACTION.get(alert_tier, _RECOMMENDED_ACTION["SUSPICIOUS"])

    top_rf_str = ", ".join(f"{f} ({v:+.3f})" for f, v in top_features_rf[:3]) or "none"
    novel_str = f"{len(novel_edges)} novel transition(s) observed" if novel_edges else "no novel transitions"
    analyst_summary = (
        f"Session {session.get('session_id', '?')} was scored {session.get('risk_score', float('nan')):.3f} "
        f"({alert_tier}), consistent with {attack_family} at the {kill_chain_stage} stage of the kill chain. "
        f"The strongest contributing flow features were {top_rf_str}, and the behavioral token path was "
        f"{' -> '.join(t for t in token_seq if t is not None) or 'empty'} ({novel_str}). "
        f"Recommended action: {recommended_action}"
    )

    return {
        "session_id": session.get("session_id"),
        "risk_score": float(session.get("risk_score", float("nan"))),
        "alert_tier": alert_tier,
        "top_features_rf": top_features_rf,
        "top_features_xgb": top_features_xgb,
        "lstm_token_attributions": lstm_token_attributions,
        "graph_path": graph_path,
        "novel_edges": novel_edges,
        "attack_family": attack_family,
        "kill_chain_stage": kill_chain_stage,
        "recommended_action": recommended_action,
        "analyst_summary": analyst_summary,
    }
