"""Stages 9-10: Temporal Attack-State Graph & Sequence Consistency Scoring.

A directed graph over the 10 behavioral event tokens whose edge weights
are transition probabilities learned from training sessions, then kept
fresh during streaming via an exponential moving average (EMA) — this is
incremental, parameter-free statistics on the graph, never a re-training
of RF/BiLSTM/XGBoost ("graph update", not "model update").
"""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


class TemporalAttackGraph:
    """Directed token-transition graph with EMA-updated edge weights."""

    def __init__(self, vocab=None, rho=config.EMA_RHO, novel_edge_weight=config.NOVEL_EDGE_WEIGHT):
        self.vocab = list(vocab) if vocab is not None else list(config.VOCAB)
        self.rho = rho
        self.novel_edge_weight = novel_edge_weight

        self.G = nx.DiGraph()
        self.G.add_nodes_from(self.vocab)
        for u in self.vocab:
            for v in self.vocab:
                self.G.add_edge(u, v, weight=0.0)

        self._known_edges: set[tuple[str, str]] = set()
        self._fitted = False
        self.anomalous_transitions: list[tuple[str, str]] = []

    def fit(self, training_sessions):
        """Learn initial transition probabilities from ORIGINAL training sessions.

        Parameters
        ----------
        training_sessions : list[list[str]]
            One chronological token sequence per training session (real
            tokens only, padding excluded). No synthetic/SMOTE data.
        """
        counts: dict[tuple[str, str], int] = {}
        totals: dict[str, int] = {}
        n_transitions = 0
        for seq in training_sessions:
            toks = [t for t in seq if t is not None]
            for u, v in zip(toks[:-1], toks[1:]):
                counts[(u, v)] = counts.get((u, v), 0) + 1
                totals[u] = totals.get(u, 0) + 1
                n_transitions += 1

        for (u, v), c in counts.items():
            self.G[u][v]["weight"] = c / totals[u]
            self._known_edges.add((u, v))

        self._fitted = True
        logger.info(
            "TemporalAttackGraph.fit: %d unique transitions (%d total) over %d training sessions",
            len(counts), n_transitions, len(training_sessions),
        )
        return self

    def update(self, session_token_sequence):
        """Incrementally EMA-update edge weights from one observed session.

        For a transition already known (from training or a prior
        update): W_t = rho * W_{t-1} + (1-rho) * W_new, with W_new = 1.0
        (a "this edge fired" observation).

        For a transition never seen before (not in training, never
        updated): weight is set directly to ``novel_edge_weight`` and the
        transition is flagged as anomalous (a potential novel attack path).
        """
        toks = [t for t in session_token_sequence if t is not None]
        updates = []
        for u, v in zip(toks[:-1], toks[1:]):
            if (u, v) not in self._known_edges:
                self.G[u][v]["weight"] = self.novel_edge_weight
                self._known_edges.add((u, v))
                self.anomalous_transitions.append((u, v))
                updates.append((u, v, self.novel_edge_weight, True))
            else:
                w_old = self.G[u][v]["weight"]
                w_updated = self.rho * w_old + (1 - self.rho) * 1.0
                self.G[u][v]["weight"] = w_updated
                updates.append((u, v, w_updated, False))
        return updates

    def compute_tc(self, session_token_sequence, timestamps, lambda_decay=config.LAMBDA_DECAY):
        """TC_t = sum_i [w(s_i -> s_i+1) * exp(-lambda * dt_i)] / (n-1).

        Edge cases:
          - n=1 (single-event session) -> 0.5 (neutral)
          - repeated tokens (e.g. AUTH_FAIL x N) count as separate
            transitions, not merged into one self-loop
          - a novel (unseen) transition uses w = novel_edge_weight
        """
        pairs = [(tok, ts) for tok, ts in zip(session_token_sequence, timestamps) if tok is not None]
        n = len(pairs)
        if n <= 1:
            return 0.5

        total = 0.0
        for i in range(n - 1):
            u, t1 = pairs[i]
            v, t2 = pairs[i + 1]
            w = self.G[u][v]["weight"] if (u, v) in self._known_edges else self.novel_edge_weight
            if pd.isna(t1) or pd.isna(t2):
                dt = 0.0
            else:
                dt = max(0.0, (t2 - t1) / np.timedelta64(1, "s"))
            total += w * np.exp(-lambda_decay * dt)

        return total / (n - 1)

    def top_transitions(self, k=10):
        """Top-k highest-weight (u, v, weight) transitions currently in the graph."""
        edges = [(u, v, d["weight"]) for u, v, d in self.G.edges(data=True) if d["weight"] > 0]
        return sorted(edges, key=lambda e: e[2], reverse=True)[:k]
