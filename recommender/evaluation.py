"""Evaluation metrics for recommendation quality."""

from __future__ import annotations
import numpy as np
from typing import Dict, List
from collections import defaultdict


def precision_at_k(recommended: List[str], relevant: set, k: int) -> float:
    if k == 0:
        return 0.0
    rec = recommended[:k]
    return len(set(rec) & relevant) / k


def recall_at_k(recommended: List[str], relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    rec = recommended[:k]
    return len(set(rec) & relevant) / len(relevant)


def ndcg_at_k(recommended: List[str], relevant: set, k: int) -> float:
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)
    ideal = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_ranking(engine, test_events: List[Dict], k: int = 10) -> Dict[str, float]:
    """Evaluate the engine on held-out test events."""
    # ground truth: items each user engaged with in test set
    user_relevant = defaultdict(set)
    for e in test_events:
        if e.get("action") in ("like", "love", "save", "share", "follow"):
            user_relevant[e["user_id"]].add(e["item_id"])

    precisions, recalls, ndcgs = [], [], []
    for uid, relevant in user_relevant.items():
        recs = engine.recommend(uid, top_k=k)
        rec_ids = [r["item_id"] for r in recs]
        precisions.append(precision_at_k(rec_ids, relevant, k))
        recalls.append(recall_at_k(rec_ids, relevant, k))
        ndcgs.append(ndcg_at_k(rec_ids, relevant, k))

    n = max(len(precisions), 1)
    return {
        f"precision@{k}": float(np.mean(precisions)),
        f"recall@{k}": float(np.mean(recalls)),
        f"ndcg@{k}": float(np.mean(ndcgs)),
        "n_test_users": len(user_relevant),
    }
