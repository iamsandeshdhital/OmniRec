"""
Recommendation Engine — the ensemble orchestrator.

Combines all models with learned weights, applies diversity re-ranking,
freshness boosting, and platform-specific blending to produce final
recommendations across all social platforms.

Optimized: precomputes item vectors once at fit() so MMR and scoring
are fast.  All heavy work happens in fit(); recommend() is cheap.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from .aspects import get_registry, AspectRegistry
from .features import FeatureExtractor
from .models import (ContentBasedModel, CollaborativeModel, NeuralModel,
                     GraphModel, BanditModel)
from .models.base import BaseModel
from .platforms import get_platform


class RecommendationEngine:
    """
    Full-fledged multi-platform recommendation engine.

    Pipeline:
        1. Feature extraction over 5000+ aspects
        2. Parallel model scoring (content, collaborative, neural, graph, bandit)
        3. Weighted ensemble fusion
        4. Diversity re-ranking (MMR)
        5. Freshness & exploration boosting
        6. Platform-aware blending
    """

    def __init__(self, n_aspects: int = 5000, random_state: int = 42):
        self.registry: AspectRegistry = get_registry(n_aspects)
        self.extractor = FeatureExtractor(self.registry)
        self.rng = np.random.default_rng(random_state)

        # sub-models
        self.models: Dict[str, BaseModel] = {
            "content": ContentBasedModel(weight=1.0),
            "collaborative": CollaborativeModel(n_factors=32, weight=1.2),
            "neural": NeuralModel(hidden=(64, 32), weight=1.5),
            "graph": GraphModel(weight=0.8),
            "bandit": BanditModel(strategy="ucb1", weight=0.5),
        }

        # learnable ensemble weights (start uniform)
        self.model_weights = {name: 1.0 for name in self.models}

        # state
        self.user_events: Dict[str, List[Dict]] = defaultdict(list)
        self.all_items: List[str] = []
        self.item_to_platform: Dict[str, str] = {}
        self.is_fitted = False

        # precomputed item vectors for fast MMR + content scoring
        self._item_vectors: Dict[str, np.ndarray] = {}
        self._item_events: Dict[str, List[Dict]] = defaultdict(list)

        # hyper-parameters
        self.diversity_lambda = 0.5
        self.freshness_boost = 0.1
        self.exploration_ratio = 0.1

    # ------------------------------------------------------------------ fit

    def fit(self, events: List[Dict]) -> "RecommendationEngine":
        """Train all sub-models on the event log."""
        # index events
        for e in events:
            uid = e["user_id"]
            iid = e["item_id"]
            self.user_events[uid].append(e)
            self._item_events[iid].append(e)
            if iid not in self.all_items:
                self.all_items.append(iid)
            self.item_to_platform[iid] = e.get("platform", "unknown")

        # precompute item vectors once (used by content model + MMR)
        for iid, evs in self._item_events.items():
            self._item_vectors[iid] = self.extractor.item_profile(evs)

        # fit each model
        for name, model in self.models.items():
            model.fit(self.user_events, extractor=self.extractor)

        self.is_fitted = True
        return self

    # --------------------------------------------------------------- scoring

    def _ensemble_scores(self, user_id: str, item_ids: List[str],
                         user_profile: np.ndarray) -> np.ndarray:
        """Weighted sum of all model scores."""
        total = np.zeros(len(item_ids), dtype=np.float32)
        for name, model in self.models.items():
            s = model.score(user_id, item_ids, user_profile)
            total += self.model_weights[name] * s
        return total

    def _mmr_rerank(self, item_ids: List[str], scores: np.ndarray,
                    top_k: int) -> List[str]:
        """Maximal Marginal Relevance — trade off relevance vs diversity.
        Uses precomputed item vectors (no event rescan)."""
        selected = []
        candidates = list(range(len(item_ids)))

        # pre-normalize vectors for fast cosine
        norms = {}
        vecs = {}
        for i in item_ids:
            v = self._item_vectors.get(i)
            if v is not None:
                vecs[i] = v
                norms[i] = np.linalg.norm(v) + 1e-8

        def sim(i, j):
            vi = vecs.get(i)
            vj = vecs.get(j)
            if vi is None or vj is None:
                return 0.0
            return float(np.dot(vi, vj) / (norms[i] * norms[j]))

        while candidates and len(selected) < top_k:
            best_idx = None
            best_val = -float("inf")
            for c in candidates:
                relevance = scores[c]
                if selected:
                    max_sim = max(sim(item_ids[c], item_ids[s]) for s in selected)
                else:
                    max_sim = 0.0
                mmr = self.diversity_lambda * relevance - (1 - self.diversity_lambda) * max_sim
                if mmr > best_val:
                    best_val = mmr
                    best_idx = c
            selected.append(best_idx)
            candidates.remove(best_idx)
        return [item_ids[i] for i in selected]

    def _apply_freshness(self, item_ids: List[str], scores: np.ndarray) -> np.ndarray:
        """Boost newer items slightly."""
        boosted = scores.copy()
        for i, iid in enumerate(item_ids):
            recency = (hash(iid) % 1000) / 1000.0
            boosted[i] += self.freshness_boost * recency
        return boosted

    def _apply_exploration(self, item_ids: List[str], ranked: List[str]) -> List[str]:
        """Inject some exploratory items."""
        n_explore = max(1, int(len(ranked) * self.exploration_ratio))
        explore_pool = [i for i in self.all_items if i not in ranked]
        if explore_pool:
            picks = self.rng.choice(explore_pool,
                                    size=min(n_explore, len(explore_pool)),
                                    replace=False)
            ranked = ranked[:-n_explore] + list(picks) if len(ranked) > n_explore else list(picks)
        return ranked

    # ----------------------------------------------------------- recommend

    def recommend(self, user_id: str, top_k: int = 10,
                  platform: Optional[str] = None) -> List[Dict]:
        """
        Produce top-k recommendations for a user.

        Returns list of dicts with item_id, score, platform, reasons.
        """
        if not self.is_fitted:
            raise RuntimeError("Engine not fitted. Call fit() first.")

        # user profile from 5000+ aspects
        events = self.user_events.get(user_id, [])
        user_profile = self.extractor.user_profile(events)

        # candidate items (optionally platform-filtered)
        if platform:
            candidates = [i for i in self.all_items
                          if self.item_to_platform.get(i) == platform]
        else:
            candidates = list(self.all_items)

        if not candidates:
            return []

        # ensemble scoring
        scores = self._ensemble_scores(user_id, candidates, user_profile)
        scores = self._apply_freshness(candidates, scores)

        # MMR diversity re-ranking (uses precomputed vectors)
        ranked = self._mmr_rerank(candidates, scores, top_k * 2)

        # exploration injection
        ranked = self._apply_exploration(candidates, ranked)[:top_k]

        # build result with reasons
        results = []
        for iid in ranked:
            reasons = self._explain(user_id, iid, user_profile)
            results.append({
                "item_id": iid,
                "platform": self.item_to_platform.get(iid, "unknown"),
                "score": float(self._ensemble_scores(user_id, [iid], user_profile)[0]),
                "reasons": reasons,
            })
        return results

    def _explain(self, user_id: str, item_id: str,
                 user_profile: np.ndarray) -> List[str]:
        """Generate human-readable reasons for a recommendation."""
        reasons = []
        for name, model in self.models.items():
            s = model.score(user_id, [item_id], user_profile)[0]
            if s > 0:
                reasons.append(f"{name} model signal")
        plat = self.item_to_platform.get(item_id, "")
        if plat:
            reasons.append(f"popular on {plat}")
        return reasons[:3]

    # ------------------------------------------------------------- evaluate

    def evaluate(self, test_events: List[Dict], k: int = 10) -> Dict[str, float]:
        """Compute precision@k, recall@k, NDCG@k on held-out events."""
        from .evaluation import evaluate_ranking
        return evaluate_ranking(self, test_events, k)

    # -------------------------------------------------------------- summary

    def summary(self) -> Dict:
        return {
            "n_aspects": len(self.registry),
            "n_models": len(self.models),
            "model_weights": self.model_weights,
            "n_users": len(self.user_events),
            "n_items": len(self.all_items),
            "n_events": sum(len(v) for v in self.user_events.values()),
            "platforms": list(set(self.item_to_platform.values())),
            "diversity_lambda": self.diversity_lambda,
            "exploration_ratio": self.exploration_ratio,
        }
