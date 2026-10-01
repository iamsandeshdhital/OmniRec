"""Neural collaborative model — projection layer + MLP over user/item aspect vectors.
Optimized: projects 5000+ dims down to 128 before the MLP, making training fast."""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional
from .base import BaseModel


class NeuralModel(BaseModel):
    name = "neural"

    def __init__(self, proj_dim=128, hidden=(64, 32), lr=0.01, epochs=15, weight=1.0):
        super().__init__(weight)
        self.proj_dim = proj_dim
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self.user_proj: np.ndarray = None  # (proj_dim, input_dim)
        self.item_proj: np.ndarray = None  # (proj_dim, input_dim)
        self.user_profiles: Dict[str, np.ndarray] = {}
        self.item_vectors: Dict[str, np.ndarray] = {}

    def _init_mlp(self, input_dim: int):
        dims = [input_dim] + list(self.hidden) + [1]
        self.weights = []
        self.biases = []
        for i in range(len(dims) - 1):
            w = np.random.normal(0, np.sqrt(2.0 / dims[i]), (dims[i], dims[i + 1]))
            b = np.zeros(dims[i + 1])
            self.weights.append(w)
            self.biases.append(b)

    def _forward(self, x: np.ndarray):
        activations = [x]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = activations[-1] @ w + b
            if i < len(self.weights) - 1:
                z = np.maximum(z, 0)
            activations.append(z)
        return activations

    def fit(self, user_events: Dict[str, List[Dict]], **kwargs) -> "NeuralModel":
        from ..features import FeatureExtractor
        ext = kwargs.get("extractor") or FeatureExtractor()
        dim = ext.dim

        item_events: Dict[str, List[Dict]] = {}
        for uid, events in user_events.items():
            for e in events:
                item_events.setdefault(e["item_id"], []).append(e)
        for iid, evs in item_events.items():
            self.item_vectors[iid] = ext.item_profile(evs)
        for uid, events in user_events.items():
            self.user_profiles[uid] = ext.user_profile(events)

        # projection matrices: input_dim -> proj_dim
        rng = np.random.default_rng(42)
        self.user_proj = rng.normal(0, 0.01, (self.proj_dim, dim))
        self.item_proj = rng.normal(0, 0.01, (self.proj_dim, dim))

        # MLP on concatenated projected vectors
        self._init_mlp(self.proj_dim * 2)

        # Precompute projected vectors ONCE (projection is fixed/random)
        self._proj_user = {uid: self.user_proj @ self.user_profiles[uid]
                           for uid in self.user_profiles}
        self._proj_item = {iid: self.item_proj @ self.item_vectors[iid]
                           for iid in self.item_vectors}

        # build training pairs (capped for speed)
        pairs = []
        for uid, events in user_events.items():
            for e in events:
                if e.get("action") in ("like", "love", "save", "share", "follow"):
                    pairs.append((uid, e["item_id"], 1.0))
                elif e.get("action") in ("skip", "hide", "report"):
                    pairs.append((uid, e["item_id"], 0.0))

        if not pairs:
            self.is_fitted = True
            return self

        max_pairs = 3000
        if len(pairs) > max_pairs:
            indices = rng.choice(len(pairs), max_pairs, replace=False)
            pairs = [pairs[i] for i in indices]

        for _ in range(self.epochs):
            rng.shuffle(pairs)
            for uid, iid, label in pairs:
                if uid not in self._proj_user or iid not in self._proj_item:
                    continue
                x = np.concatenate([self._proj_user[uid], self._proj_item[iid]])
                acts = self._forward(x)
                pred = acts[-1][0]
                err = pred - label
                grad = np.array([err])
                for i in range(len(self.weights) - 1, -1, -1):
                    dw = np.outer(acts[i], grad)
                    db = grad.copy()
                    if i > 0:
                        grad = (self.weights[i] @ grad) * (acts[i] > 0)
                    self.weights[i] -= self.lr * dw
                    self.biases[i] -= self.lr * db
                # projection is fixed (random projection) — only MLP is trained
        self.is_fitted = True
        return self

    def score(self, user_id: str, item_ids: List[str],
              user_profile: Optional[np.ndarray] = None) -> np.ndarray:
        if user_profile is None:
            user_profile = self.user_profiles.get(user_id)
        if user_profile is None:
            return np.zeros(len(item_ids), dtype=np.float32)
        # project user once, reuse precomputed item projections
        u_proj = self.user_proj @ user_profile
        scores = []
        for iid in item_ids:
            i_proj = self._proj_item.get(iid)
            if i_proj is None:
                scores.append(0.0)
                continue
            x = np.concatenate([u_proj, i_proj])
            s = self._forward(x)[-1][0]
            scores.append(float(s))
        return np.array(scores, dtype=np.float32) * self.weight
