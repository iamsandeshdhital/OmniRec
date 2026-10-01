"""Graph-based model — PageRank-style propagation over user-item bipartite graph."""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional
from .base import BaseModel


class GraphModel(BaseModel):
    name = "graph"

    def __init__(self, damping=0.85, iterations=50, weight=1.0):
        super().__init__(weight)
        self.damping = damping
        self.iterations = iterations
        self.node_scores: Dict[str, float] = {}
        self.user_neighbors: Dict[str, List[str]] = {}
        self.item_neighbors: Dict[str, List[str]] = {}

    def fit(self, user_events: Dict[str, List[Dict]], **kwargs) -> "GraphModel":
        # build bipartite graph
        for uid, events in user_events.items():
            self.user_neighbors[uid] = []
            for e in events:
                iid = e["item_id"]
                self.user_neighbors[uid].append(iid)
                self.item_neighbors.setdefault(iid, []).append(uid)

        # personalized PageRank from each user
        all_nodes = set(self.user_neighbors.keys()) | set(self.item_neighbors.keys())
        for uid in self.user_neighbors:
            self.node_scores[uid] = self._ppr(uid, all_nodes)

        self.is_fitted = True
        return self

    def _ppr(self, source: str, all_nodes) -> Dict[str, float]:
        scores = {n: 0.0 for n in all_nodes}
        scores[source] = 1.0
        for _ in range(self.iterations):
            new_scores = {n: 0.0 for n in all_nodes}
            for node, s in scores.items():
                if s == 0:
                    continue
                if node in self.user_neighbors:
                    nbrs = self.user_neighbors[node]
                else:
                    nbrs = self.item_neighbors.get(node, [])
                if not nbrs:
                    new_scores[node] += s
                    continue
                share = s / len(nbrs)
                for nb in nbrs:
                    new_scores[nb] += share
            # damping
            for n in new_scores:
                new_scores[n] = (1 - self.damping) * (1.0 if n == source else 0.0) \
                                + self.damping * new_scores[n]
            scores = new_scores
        return scores

    def score(self, user_id: str, item_ids: List[str],
              user_profile: Optional[np.ndarray] = None) -> np.ndarray:
        if user_id not in self.node_scores:
            return np.zeros(len(item_ids), dtype=np.float32)
        scores = self.node_scores[user_id]
        return np.array([scores.get(iid, 0.0) for iid in item_ids],
                        dtype=np.float32) * self.weight
