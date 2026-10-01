"""
Feature extraction — converts raw platform events into dense aspect vectors
that the models consume.

Optimized: builds a value→index lookup at init so transform() is O(1) per
field instead of scanning all 5000+ aspects.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple
from .aspects import AspectRegistry, get_registry


class FeatureExtractor:
    """
    Maps raw interaction events onto the 5000+ aspect space.

    Each event is a dict like:
        {"user_id": ..., "item_id": ..., "platform": "youtube",
         "action": "like", "topic": "gaming", "modality": "video", ...}

    The extractor produces a fixed-length vector indexed by aspect id.
    """

    def __init__(self, registry: Optional[AspectRegistry] = None):
        self.registry = registry or get_registry(5000)
        self.aspect_ids = self.registry.all_ids()
        self.id_to_idx = {aid: i for i, aid in enumerate(self.aspect_ids)}
        self.dim = len(self.aspect_ids)

        # Build lookup: value -> list of aspect indices containing that value
        self._value_index: Dict[str, List[int]] = {}
        for aid, idx in self.id_to_idx.items():
            # index the last component (the leaf value) and the full id
            parts = aid.split(".")
            for p in parts:
                self._value_index.setdefault(p, []).append(idx)
            # also index the category
            self._value_index.setdefault(parts[0], []).append(idx)

    def transform(self, event: Dict) -> np.ndarray:
        """Convert a single event dict into a dense aspect vector. O(1) per field."""
        vec = np.zeros(self.dim, dtype=np.float32)

        # Map event fields to aspect values
        field_values = {
            "platform": event.get("platform", ""),
            "action": event.get("action", ""),
            "topic": event.get("topic", ""),
            "modality": event.get("modality", ""),
            "format": event.get("format", ""),
            "sentiment": event.get("sentiment", ""),
            "surface": event.get("surface", ""),
            "device": event.get("device", ""),
            "time_of_day": event.get("time_of_day", ""),
            "relation": event.get("relation", ""),
            "motivation": event.get("motivation", ""),
            "intent": event.get("intent", ""),
            "mood": event.get("mood", ""),
            "creator_tier": event.get("creator_tier", ""),
            "velocity": event.get("velocity", ""),
            "rating": event.get("rating", ""),
        }

        for val in field_values.values():
            if val and val in self._value_index:
                for idx in self._value_index[val]:
                    vec[idx] = 1.0

        # continuous signals — distribute across hashed indices
        for key in ["dwell_time", "scroll_depth", "replay_count", "share_count",
                    "comment_length", "watch_ratio", "return_probability"]:
            if key in event:
                idx = hash(key) % self.dim
                vec[idx] = float(event[key])

        return vec

    def transform_batch(self, events: List[Dict]) -> np.ndarray:
        return np.stack([self.transform(e) for e in events])

    def user_profile(self, user_events: List[Dict]) -> np.ndarray:
        """Aggregate a user's events into a single preference vector with recency bias."""
        if not user_events:
            return np.zeros(self.dim, dtype=np.float32)
        vecs = self.transform_batch(user_events)
        n = len(vecs)
        weights = np.exp(np.linspace(-1, 0, n))
        weights /= weights.sum()
        profile = np.average(vecs, axis=0, weights=weights)
        return profile.astype(np.float32)

    def item_profile(self, item_events: List[Dict]) -> np.ndarray:
        return self.user_profile(item_events)


if __name__ == "__main__":
    ext = FeatureExtractor()
    print(f"Feature dimension: {ext.dim}")
    event = {
        "platform": "youtube", "action": "like", "topic": "gaming",
        "modality": "video", "format": "longform", "sentiment": "positive",
        "surface": "home", "device": "mobile", "time_of_day": "evening",
        "relation": "friend", "motivation": "entertainment",
        "intent": "kill_time", "mood": "excited", "creator_tier": "verified",
        "velocity": "rising", "rating": "teen", "dwell_time": 0.8,
    }
    vec = ext.transform(event)
    print(f"Non-zero aspects activated: {np.count_nonzero(vec)}")
    print(f"Vector shape: {vec.shape}")
