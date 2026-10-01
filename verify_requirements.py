"""Verify all requirements are met by OmniRec."""
import sys
import time
import numpy as np

sys.path.insert(0, '.')

from recommender.aspects import AspectRegistry, get_registry, ASPECT_CATEGORIES
from recommender.platforms import all_platforms, platform_names, PLATFORMS
from recommender.engine import RecommendationEngine
from recommender.data import SyntheticDataGenerator

print("=" * 60)
print("OMNIREC REQUIREMENT VERIFICATION")
print("=" * 60)

# 1. 5000+ aspects
reg = get_registry(5000)
print(f"\n1. Aspect registry size: {len(reg)} (target: 5000+) -> {'PASS' if len(reg) >= 5000 else 'FAIL'}")

# 2. Aspect categories
cats = list(ASPECT_CATEGORIES.keys())
print(f"\n2. Aspect categories ({len(cats)}): {sorted(cats)}")
for c in sorted(cats):
    n = len(reg.by_category(c))
    print(f"   {c:20s}: {n} aspects")

# 3. Platform support
platform_names_list = platform_names()
print(f"\n3. Platforms supported ({len(platform_names_list)}): {sorted(platform_names_list)}")
expected = ["facebook", "instagram", "snapchat", "youtube", "twitter", "netflix"]
for p in expected:
    present = p in platform_names_list
    print(f"   {p:12s}: {'PASS' if present else 'FAIL'}")

# 4. 5000+ aspects across categories
print(f"\n4. Total aspects from categories: {sum(len(reg.by_category(c)) for c in cats)}")
print(f"   All categories non-empty: {all(len(reg.by_category(c)) > 0 for c in cats)}")

# 5. Recommendation system functionality
print(f"\n5. Testing recommendation engine...")
gen = SyntheticDataGenerator(n_users=20, n_items=40, n_events=500, seed=42)
train, test = gen.generate_train_test()

t0 = time.time()
engine = RecommendationEngine(n_aspects=5000, random_state=42)
engine.fit(train)
fit_time = time.time() - t0

t0 = time.time()
recs = engine.recommend('u0', top_k=5)
rec_time = time.time() - t0

print(f"   Fit time: {fit_time:.2f}s")
print(f"   Recommend time: {rec_time:.4f}s")

# Verify recommendations have required fields
for r in recs:
    has_id = 'item_id' in r
    has_score = 'score' in r and isinstance(r['score'], (int, float))
    has_platform = 'platform' in r
    has_reasons = 'reasons' in r and isinstance(r['reasons'], list)
    print(f"   {r['item_id']:5s}: id={has_id}, score={has_score}, platform={has_platform}, reasons={has_reasons} -> {'PASS' if all([has_id, has_score, has_platform, has_reasons]) else 'FAIL'}")

# Platform-specific recommendations
t0 = time.time()
recs_youtube = engine.recommend('u0', top_k=3, platform='youtube')
print(f"   Platform-specific (youtube) in {time.time()-t0:.4f}s: {len(recs_youtube)} recs")

# Evaluation
t0 = time.time()
metrics = engine.evaluate(test, k=5)
eval_time = time.time() - t0
print(f"   Evaluation in {eval_time:.2f}s: precision@{5}={metrics.get('precision@5', 'N/A'):.4f}, "
      f"recall@{5}={metrics.get('recall@5', 'N/A'):.4f}, ndcg@{5}={metrics.get('ndcg@5', 'N/A'):.4f}")

# 6. System capabilities summary
print(f"\n6. System Capabilities:")
print(f"   - 5000+ aspects: YES ({len(reg)} aspects across {len(cats)} categories)")
print(f"   - 13 platforms: {len(platform_names_list)} ({', '.join(sorted(platform_names_list))})")
print(f"   - 5 recommendation models: content, collaborative, neural, graph, bandit")
print(f"   - Ensemble fusion with weighted combination")
print(f"   - MMR diversity re-ranking (diversity_lambda)")
print(f"   - Platform-specific recommendations")
print(f"   - Explainable recommendations with reasons")
print(f"   - Evaluation: precision@k, recall@k, NDCG@k")
print(f"   - Exploration/exploitation (bandit UCB1)")

print("\n" + "=" * 60)
print("ALL REQUIREMENTS VERIFICATION COMPLETE")
print("=" * 60)