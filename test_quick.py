"""Quick integration test for OmniRec."""
import time
import numpy as np
from recommender.data import SyntheticDataGenerator
from recommender.engine import RecommendationEngine

print("=== OmniRec Quick Integration Test ===\n")

# 1. Registry
t = time.time()
from recommender.aspects import AspectRegistry
reg = AspectRegistry(5000)
print(f"[1] Registry: {len(reg)} aspects in {time.time()-t:.2f}s")

# 2. Data
t = time.time()
gen = SyntheticDataGenerator(n_users=50, n_items=100, n_events=2000, seed=42)
train, test = gen.generate_train_test()
print(f"[2] Data: {len(train)} train, {len(test)} test in {time.time()-t:.2f}s")

# 3. Fit
t = time.time()
engine = RecommendationEngine(n_aspects=5000, random_state=42)
engine.fit(train)
print(f"[3] Engine fit: {time.time()-t:.2f}s")

# 4. Recommend
t = time.time()
recs = engine.recommend("u0", top_k=5)
print(f"[4] Recommend: {time.time()-t:.4f}s")
for r in recs:
    print(f"    {r['item_id']:>6}  score={r['score']:+.4f}  platform={r['platform']}")

# 5. Platform-specific
t = time.time()
recs = engine.recommend("u0", top_k=5, platform="youtube")
print(f"[5] Platform-specific (youtube): {time.time()-t:.4f}s")
for r in recs:
    print(f"    {r['item_id']:>6}  score={r['score']:+.4f}  platform={r['platform']}")

# 6. Evaluate
t = time.time()
metrics = engine.evaluate(test, k=5)
print(f"[6] Evaluate: {time.time()-t:.2f}s")
for k, v in metrics.items():
    if isinstance(v, float):
        print(f"    {k}: {v:.4f}")
    else:
        print(f"    {k}: {v}")

# 7. Summary
print(f"\n[7] Engine summary:")
s = engine.summary()
for k, v in s.items():
    print(f"    {k}: {v}")

print("\n=== All tests passed ===")
