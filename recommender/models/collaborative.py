"""Collaborative filtering via matrix factorization (SGD)."""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional
from .base import BaseModel


class CollaborativeModel(BaseModel):
    name = "collaborative"

    def __init__(self, n_factors=32, lr=0.01, reg=0.02, epochs=20, weight=1.0):
        super().__init__(weight)
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.user_factors: Dict[str, np.ndarray] = {}
        self.item_factors: Dict[str, np.ndarray] = {}
        self.user_bias: Dict[str, float] = {}
        self.item_bias: Dict[str, float] = {}
        self.global_bias = 0.0

    def fit(self, user_events: Dict[str, List[Dict]], **kwargs) -> "CollaborativeModel":
        # build rating matrix from implicit feedback
        ratings: Dict[str, Dict[str, float]] = {}
        all_items = set()
        for uid, events in user_events.items():
            ratings[uid] = {}
            for e in events:
                iid = e["item_id"]
                all_items.add(iid)
                # implicit rating from action
                r = self._action_rating(e.get("action", "view"))
                ratings[uid][iid] = max(ratings[uid].get(iid, 0), r)

        all_items = sorted(all_items)
        self.item_ids = all_items
        self.item_to_idx = {i: k for k, i in enumerate(all_items)}

        # init factors
        for uid in ratings:
            self.user_factors[uid] = np.random.normal(0, 0.1, self.n_factors)
            self.user_bias[uid] = 0.0
        for iid in all_items:
            self.item_factors[iid] = np.random.normal(0, 0.1, self.n_factors)
            self.item_bias[iid] = 0.0

        all_r = [r for u in ratings.values() for r in u.values()]
        self.global_bias = float(np.mean(all_r)) if all_r else 0.0

        # SGD
        for _ in range(self.epochs):
            for uid, items in ratings.items():
                for iid, r in items.items():
                    pred = self._predict(uid, iid)
                    err = r - pred
                    self.user_bias[uid] += self.lr * (err - self.reg * self.user_bias[uid])
                    self.item_bias[iid] += self.lr * (err - self.reg * self.item_bias[iid])
                    uf = self.user_factors[uid].copy()
                    self.user_factors[uid] += self.lr * (err * self.item_factors[iid]
                                                        - self.reg * self.user_factors[uid])
                    self.item_factors[iid] += self.lr * (err * uf
                                                        - self.reg * self.item_factors[iid])
        self.is_fitted = True
        return self

    def _predict(self, uid: str, iid: str) -> float:
        if uid not in self.user_factors or iid not in self.item_factors:
            return self.global_bias
        return (self.global_bias + self.user_bias[uid] + self.item_bias[iid]
                + float(np.dot(self.user_factors[uid], self.item_factors[iid])))

    def _action_rating(self, action: str) -> float:
        table = {
            "love": 5, "like": 4, "save": 4, "share": 5, "comment": 4,
            "view": 2, "click": 3, "follow": 5, "replay": 4,
            "skip": 1, "hide": 0, "report": 0, "block": 0,
            "dwell": 3, "scroll_slow": 2, "scroll_fast": 1,
        }
        return table.get(action, 2)

    def score(self, user_id: str, item_ids: List[str],
              user_profile: Optional[np.ndarray] = None) -> np.ndarray:
        scores = [self._predict(user_id, iid) for iid in item_ids]
        return np.array(scores, dtype=np.float32) * self.weight
