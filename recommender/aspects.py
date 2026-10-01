"""
Aspect Registry — 5000+ variables that drive recommendations.

Each aspect is a measurable signal. They are grouped into categories and
generated combinatorially so the registry scales past 5000 while remaining
fully inspectable and configurable.

Categories
----------
content        — intrinsic properties of the item (text, visual, audio, metadata)
behavior       — how users interact with items (clicks, dwell, scroll, shares)
context        — environment of the consumption event (device, network, place)
temporal       — time-based patterns (recency, seasonality, session rhythm)
social_graph   — network structure and interaction topology
psychographic  — inferred taste, personality, sentiment, intent
cross_platform — signals aggregated across every connected platform
creator        — author/channel identity, authority, production quality
engagement    — velocity and distribution of reactions
monetization   — commercial signals (ad affinity, purchase intent)
privacy_safety — trust, sensitivity, brand-safety scoring
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json


# ---------------------------------------------------------------------------
# Category definitions.  Each category declares how many base aspects it
# contributes and the "axes" used to expand them combinatorially.
# ---------------------------------------------------------------------------

ASPECT_CATEGORIES: Dict[str, dict] = {
    "content": {
        "description": "Intrinsic item properties",
        "axes": {
            "modality": ["text", "image", "video", "audio", "live", "story", "reel", "short"],
            "topic": ["news", "sports", "music", "gaming", "food", "travel", "fashion",
                      "tech", "science", "art", "comedy", "education", "finance", "health",
                      "politics", "lifestyle", "diy", "nature", "auto", "pets"],
            "format": ["article", "clip", "longform", "carousel", "poll", "quiz",
                       "tutorial", "review", "vlog", "meme", "interview", "documentary"],
            "language": ["en", "es", "hi", "fr", "de", "ja", "ko", "pt", "ar", "zh"],
            "sentiment": ["positive", "negative", "neutral", "inspirational", "controversial"],
        },
    },
    "behavior": {
        "description": "User interaction signals",
        "axes": {
            "action": ["view", "like", "love", "share", "save", "comment", "click",
                       "skip", "hide", "report", "follow", "unfollow", "dwell",
                       "scroll_fast", "scroll_slow", "replay", "mute", "block"],
            "surface": ["feed", "explore", "search", "reels", "stories", "watch",
                        "notifications", "dm", "profile", "trending"],
            "depth": ["shallow", "medium", "deep", "complete", "abandoned"],
        },
    },
    "context": {
        "description": "Consumption environment",
        "axes": {
            "device": ["mobile", "desktop", "tablet", "tv", "wearable", "console"],
            "network": ["wifi", "4g", "5g", "offline", "metered"],
            "location_type": ["home", "work", "commute", "travel", "venue", "outdoors"],
            "time_of_day": ["dawn", "morning", "afternoon", "evening", "night", "late_night"],
            "day_type": ["weekday", "weekend", "holiday", "event_day"],
        },
    },
    "temporal": {
        "description": "Time-based patterns",
        "axes": {
            "recency": ["now", "hour", "day", "week", "month", "quarter", "year"],
            "season": ["spring", "summer", "autumn", "winter"],
            "cadence": ["burst", "steady", "declining", "resurgent", "dormant"],
            "session_position": ["first", "early", "middle", "late", "last"],
        },
    },
    "social_graph": {
        "description": "Network topology signals",
        "axes": {
            "relation": ["self", "friend", "follower", "following", "mutual",
                         "friend_of_friend", "community", "stranger"],
            "interaction": ["liked", "commented", "shared", "mentioned", "tagged",
                            "co_viewed", "co_engaged", "replied"],
            "influence_tier": ["nano", "micro", "mid", "macro", "mega", "celebrity"],
            "cluster": ["tight_knit", "interest_based", "geographic", "professional",
                        "family", "acquaintance"],
        },
    },
    "psychographic": {
        "description": "Inferred taste and intent",
        "axes": {
            "big5": ["openness", "conscientiousness", "extraversion", "agreeableness",
                     "neuroticism"],
            "motivation": ["entertainment", "socializing", "information", "shopping",
                           "self_expression", "relaxation", "learning", "status"],
            "intent": ["kill_time", "research", "purchase", "connect", "create",
                       "vent", "celebrate", "commiserate"],
            "mood": ["happy", "sad", "anxious", "bored", "excited", "calm", "angry",
                     "nostalgic"],
        },
    },
    "cross_platform": {
        "description": "Aggregated cross-platform signals",
        "axes": {
            "platform": ["facebook", "instagram", "snapchat", "youtube", "twitter",
                         "netflix", "tiktok", "twitch", "reddit", "linkedin",
                         "pinterest", "spotify", "discord"],
            "signal": ["affinity", "reach", "virality", "saves", "shares",
                       "watch_time", "completion", "return_rate", "churn_risk"],
            "window": ["realtime", "daily", "weekly", "monthly"],
        },
    },
    "creator": {
        "description": "Author/channel signals",
        "axes": {
            "tier": ["new", "emerging", "established", "verified", "official"],
            "quality": ["amateur", "semi_pro", "professional", "studio", "broadcast"],
            "consistency": ["sporadic", "regular", "daily", "seasonal"],
            "authenticity": ["curated", "raw", "filtered", "sponsored", "organic"],
        },
    },
    "engagement": {
        "description": "Reaction velocity & distribution",
        "axes": {
            "velocity": ["flat", "rising", "peaking", "viral", "decaying"],
            "distribution": ["concentrated", "distributed", "polarized", "unanimous"],
            "ratio": ["low", "medium", "high", "exceptional"],
        },
    },
    "monetization": {
        "description": "Commercial signals",
        "axes": {
            "signal": ["ad_click", "product_view", "cart_add", "purchase", "subscription",
                       "tip", "sponsor_click", "affiliate"],
            "funnel_stage": ["awareness", "consideration", "decision", "retention",
                             "advocacy"],
        },
    },
    "privacy_safety": {
        "description": "Trust and safety scoring",
        "axes": {
            "rating": ["everyone", "teen", "mature", "restricted"],
            "risk": ["safe", "sensitive", "graphic", "misinformation", "spam"],
            "brand_safety": ["full", "partial", "none"],
        },
    },
}


@dataclass
class Aspect:
    """A single measurable variable in the recommendation system."""
    id: str
    name: str
    category: str
    description: str
    data_type: str          # continuous | categorical | binary | ordinal
    weight: float = 1.0     # learnable importance weight
    source: str = ""        # where the signal originates
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "data_type": self.data_type,
            "weight": self.weight,
            "source": self.source,
            "tags": self.tags,
        }


class AspectRegistry:
    """
    Generates and holds the full 5000+ aspect registry.

    Aspects are produced combinatorially from each category's axes, then
    given stable string IDs (e.g. ``content.topic.sports.video``) so they
    can be referenced, weighted and persisted.

    To keep the registry at the target size, each category contributes a
    capped number of combinations, and cross-platform interaction aspects
    fill the remainder.
    """

    def __init__(self, target: int = 5000):
        self.target = target
        self.aspects: Dict[str, Aspect] = {}
        self._by_category: Dict[str, List[Aspect]] = {}
        self._generate()

    # -- generation ---------------------------------------------------------

    def _generate(self) -> None:
        # First pass: generate base aspects per category, capped so the
        # total stays near target.  We allocate a budget per category.
        n_categories = len(ASPECT_CATEGORIES)
        budget_per_cat = self.target // n_categories

        for cat, spec in ASPECT_CATEGORIES.items():
            axes = spec["axes"]
            axis_names = list(axes.keys())
            axis_values = [axes[a] for a in axis_names]

            combos = self._product(axis_values)
            # cap combinations per category to stay near budget
            if len(combos) > budget_per_cat:
                combos = combos[:budget_per_cat]

            for combo in combos:
                parts = [cat] + [str(v) for v in combo]
                aid = ".".join(parts)
                name = " ".join(str(v) for v in combo).replace("_", " ").title()
                aspect = Aspect(
                    id=aid,
                    name=f"{cat.title()}: {name}",
                    category=cat,
                    description=f"{spec['description']} — "
                                f"{', '.join(f'{a}={v}' for a, v in zip(axis_names, combo))}",
                    data_type="categorical",
                    weight=1.0,
                    source=cat,
                    tags=[cat] + [str(v) for v in combo],
                )
                self.aspects[aid] = aspect
                self._by_category.setdefault(cat, []).append(aspect)

        # Fill remaining with cross-platform interaction aspects
        self._expand_to_target()

    def _expand_to_target(self) -> None:
        """Add cross-platform interaction aspects until target is reached."""
        platforms = ["facebook", "instagram", "snapchat", "youtube", "twitter",
                     "netflix", "tiktok", "twitch", "reddit", "linkedin",
                     "pinterest", "spotify", "discord"]
        base = list(self.aspects.values())
        idx = 0
        while len(self.aspects) < self.target and idx < len(base) * len(platforms):
            b = base[idx % len(base)]
            p = platforms[idx % len(platforms)]
            aid = f"xplatform.{p}.{b.id}"
            if aid not in self.aspects:
                self.aspects[aid] = Aspect(
                    id=aid,
                    name=f"[{p.title()}] {b.name}",
                    category="cross_platform",
                    description=f"Cross-platform signal: {b.description} (on {p})",
                    data_type=b.data_type,
                    weight=1.0,
                    source=p,
                    tags=["cross_platform", p, b.category],
                )
                self._by_category.setdefault("cross_platform", []).append(self.aspects[aid])
            idx += 1

    @staticmethod
    def _product(lists):
        """Cartesian product of a list of lists."""
        result = [[]]
        for pool in lists:
            result = [x + [y] for x in result for y in pool]
        return result

    # -- access -------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.aspects)

    def get(self, aid: str) -> Optional[Aspect]:
        return self.aspects.get(aid)

    def by_category(self, cat: str) -> List[Aspect]:
        return self._by_category.get(cat, [])

    def all_ids(self) -> List[str]:
        return list(self.aspects.keys())

    def summary(self) -> dict:
        return {
            "total_aspects": len(self.aspects),
            "target": self.target,
            "categories": {c: len(v) for c, v in self._by_category.items()},
        }

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({aid: a.to_dict() for aid, a in self.aspects.items()},
                      f, indent=2)

    def __repr__(self) -> str:
        return f"AspectRegistry({len(self.aspects)} aspects, {len(self._by_category)} categories)"


# Singleton-ish default registry
DEFAULT_REGISTRY = None


def get_registry(target: int = 5000) -> AspectRegistry:
    global DEFAULT_REGISTRY
    if DEFAULT_REGISTRY is None or len(DEFAULT_REGISTRY) < target:
        DEFAULT_REGISTRY = AspectRegistry(target)
    return DEFAULT_REGISTRY


if __name__ == "__main__":
    reg = AspectRegistry(5000)
    print(json.dumps(reg.summary(), indent=2))
    print("Sample aspects:")
    for aid in list(reg.aspects)[:5]:
        print(" ", reg.aspects[aid])
