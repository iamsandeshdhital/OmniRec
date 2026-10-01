"""
Synthetic data generation — creates realistic multi-platform interaction
datasets for training and evaluating the recommendation engine.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple
import random

from .platforms import platform_names, PLATFORMS
from .aspects import ASPECT_CATEGORIES


TOPICS = ASPECT_CATEGORIES["content"]["axes"]["topic"]
MODALITIES = ASPECT_CATEGORIES["content"]["axes"]["modality"]
FORMATS = ASPECT_CATEGORIES["content"]["axes"]["format"]
SENTIMENTS = ASPECT_CATEGORIES["content"]["axes"]["sentiment"]
ACTIONS = ASPECT_CATEGORIES["behavior"]["axes"]["action"]
SURFACES = ASPECT_CATEGORIES["behavior"]["axes"]["surface"]
DEVICES = ASPECT_CATEGORIES["context"]["axes"]["device"]
TIMES = ASPECT_CATEGORIES["context"]["axes"]["time_of_day"]
RELATIONS = ASPECT_CATEGORIES["social_graph"]["axes"]["relation"]
MOTIVATIONS = ASPECT_CATEGORIES["psychographic"]["axes"]["motivation"]
INTENTS = ASPECT_CATEGORIES["psychographic"]["axes"]["intent"]
MOODS = ASPECT_CATEGORIES["psychographic"]["axes"]["mood"]
CREATOR_TIERS = ASPECT_CATEGORIES["creator"]["axes"]["tier"]
VELOCITIES = ASPECT_CATEGORIES["engagement"]["axes"]["velocity"]
RATINGS = ASPECT_CATEGORIES["privacy_safety"]["axes"]["rating"]


class SyntheticDataGenerator:
    """Generates users, items, and interaction events across platforms."""

    def __init__(self, n_users=200, n_items=500, n_events=20000, seed=42):
        self.n_users = n_users
        self.n_items = n_items
        self.n_events = n_events
        self.rng = np.random.default_rng(seed)
        self.random = random.Random(seed)

        self.user_ids = [f"u{i}" for i in range(n_users)]
        self.item_ids = [f"i{i}" for i in range(n_items)]
        self.platforms = platform_names()

        # latent taste vectors for users and items (ground truth)
        self.user_taste = self.rng.normal(0, 1, (n_users, 32)).astype(np.float32)
        self.item_taste = self.rng.normal(0, 1, (n_items, 32)).astype(np.float32)

        # user topic affinities
        self.user_topic_aff = self.rng.dirichlet(
            np.ones(len(TOPICS)) * 0.5, size=n_users).astype(np.float32)

    def _sample_event(self) -> Dict:
        u = self.random.randint(0, self.n_users - 1)
        # item choice biased by taste similarity
        sims = self.item_taste @ self.user_taste[u]
        sims = np.exp(sims - sims.max())
        sims /= sims.sum()
        i = self.rng.choice(self.n_items, p=sims)

        platform = self.random.choice(self.platforms)
        pobj = PLATFORMS[platform]

        topic = self.random.choices(TOPICS, weights=self.user_topic_aff[u])[0]

        event = {
            "user_id": self.user_ids[u],
            "item_id": self.item_ids[i],
            "platform": platform,
            "action": self.random.choice(ACTIONS),
            "topic": topic,
            "modality": self.random.choice(MODALITIES),
            "format": self.random.choice(FORMATS),
            "sentiment": self.random.choice(SENTIMENTS),
            "surface": self.random.choice(pobj.surfaces),
            "device": self.random.choice(DEVICES),
            "time_of_day": self.random.choice(TIMES),
            "relation": self.random.choice(RELATIONS),
            "motivation": self.random.choice(MOTIVATIONS),
            "intent": self.random.choice(INTENTS),
            "mood": self.random.choice(MOODS),
            "creator_tier": self.random.choice(CREATOR_TIERS),
            "velocity": self.random.choice(VELOCITIES),
            "rating": self.random.choice(RATINGS),
            "dwell_time": float(self.rng.beta(2, 5)),
            "scroll_depth": float(self.rng.beta(3, 2)),
            "watch_ratio": float(self.rng.beta(2, 3)),
            "replay_count": int(self.rng.poisson(1)),
            "share_count": int(self.rng.poisson(0.5)),
            "comment_length": int(self.rng.exponential(50)),
            "return_probability": float(self.rng.beta(2, 2)),
        }
        return event

    def generate(self) -> List[Dict]:
        return [self._sample_event() for _ in range(self.n_events)]

    def generate_train_test(self, test_ratio=0.2) -> Tuple[List[Dict], List[Dict]]:
        events = self.generate()
        self.random.shuffle(events)
        split = int(len(events) * (1 - test_ratio))
        return events[:split], events[split:]


if __name__ == "__main__":
    gen = SyntheticDataGenerator(n_users=50, n_items=100, n_events=1000)
    events = gen.generate()
    print(f"Generated {len(events)} events")
    print("Sample:", {k: events[0][k] for k in ["platform", "action", "topic"]})
    platforms = set(e["platform"] for e in events)
    print(f"Platforms covered: {len(platforms)}")
