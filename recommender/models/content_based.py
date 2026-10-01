"""Content-based filtering — cosine similarity between user profile and items."""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional
from .base import BaseModel


class ContentBasedModel(BaseModel):
    name = "content_based"

    def __init__(self, weight: float = 1.0):
        super().__init__(weight)
        self.item_vectors: Dict[str, np.ndarray] = {}
        self.user_profiles: Dict[str, np.ndarray] = {}

    def fit(self, user_events: Dict[str, List[Dict]], **kwargs) -> "ContentBasedModel":
        from ..features import FeatureExtractor
        ext = kwargs.get("extractor") or FeatureExtractor()

        # build item vectors from all events referencing each item
        item_events: Dict[str, List[Dict]] = {}
        for uid, events in user_events.items():
            for e in events:
                item_events.setdefault(e["item_id"], []).append(e)

        for iid, evs in item_events.items():
            self.item_vectors[iid] = ext.item_profile(evs)

        for uid, events in user_events.items():
            self.user_profiles[uid] = ext.user_profile(events)

        self.is_fitted = True
        return self

    def score(self, user_id: str, item_ids: List[str],
              user_profile: Optional[np.ndarray] = None) -> np.ndarray:
        if user_profile is None:
            user_profile = self.user_profiles.get(user_id)
        if user_profile is None:
            return np.zeros(len(item_ids), dtype=np.float32)

        u = user_profile / (np.linalg.norm(user_profile) + 1e-8)
        scores = []
        for iid in item_ids:
            v = self.item_vectors.get(iid)
            if v is None:
                scores.append(0.0)
                continue
            v = v / (np.linalg.norm(v) + 1e-8)
            scores.append(float(np.dot(u, v)))
        return np.array(scores, dtype=np.float32) * self.weight
