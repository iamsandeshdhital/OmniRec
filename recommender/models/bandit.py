"""Multi-armed bandit — exploration/exploitation for cold-start and diversity."""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional
from .base import BaseModel


class BanditModel(BaseModel):
    name = "bandit"

    def __init__(self, strategy="ucb1", weight=1.0):
        super().__init__(weight)
        self.strategy = strategy
        self.counts: Dict[str, int] = {}
        self.values: Dict[str, float] = {}
        self.total_pulls = 0

    def fit(self, user_events: Dict[str, List[Dict]], **kwargs) -> "BanditModel":
        for uid, events in user_events.items():
            for e in events:
                iid = e["item_id"]
                r = 1.0 if e.get("action") in ("like", "love", "save", "share") else 0.0
                self.counts[iid] = self.counts.get(iid, 0) + 1
                self.values[iid] = self.values.get(iid, 0.0) + r
                self.total_pulls += 1
        # convert to mean values
        for iid in self.values:
            self.values[iid] /= self.counts[iid]
        self.is_fitted = True
        return self

    def score(self, user_id: str, item_ids: List[str],
              user_profile: Optional[np.ndarray] = None) -> np.ndarray:
        scores = []
        for iid in item_ids:
            if self.strategy == "ucb1":
                if iid not in self.counts:
                    scores.append(float("inf"))  # explore unseen
                else:
                    exploit = self.values[iid]
                    explore = np.sqrt(2 * np.log(self.total_pulls + 1) / self.counts[iid])
                    scores.append(exploit + explore)
            else:  # epsilon-greedy fallback
                scores.append(self.values.get(iid, 0.5))
        return np.array(scores, dtype=np.float32) * self.weight
