"""
OmniRec — Full-fledged multi-platform recommendation system demo.

Run:  python main.py
"""

import json
import time
import numpy as np

from recommender.aspects import AspectRegistry
from recommender.platforms import all_platforms, platform_names
from recommender.data import SyntheticDataGenerator
from recommender.engine import RecommendationEngine


def banner():
    print("=" * 70)
    print("  OmniRec — Multi-Platform Recommendation Engine")
    print("  5000+ aspects | 5 models | ensemble | diversity | exploration")
    print("=" * 70)


def main():
    banner()

    # 1. Aspect registry
    print("\n[1] Building 5000+ aspect registry...")
    reg = AspectRegistry(5000)
    summary = reg.summary()
    print(f"    Total aspects : {summary['total_aspects']}")
    print(f"    Categories    : {len(summary['categories'])}")
    for cat, count in sorted(summary["categories"].items()):
        print(f"      {cat:16} {count:>5}")

    # 2. Platforms
    print("\n[2] Supported platforms:")
    for p in all_platforms():
        print(f"    {p.display_name:14} modality={p.primary_modality:12} "
              f"types={len(p.content_types)} signals={len(p.signals)}")

    # 3. Generate data
    print("\n[3] Generating synthetic multi-platform interaction data...")
    t0 = time.time()
    gen = SyntheticDataGenerator(n_users=200, n_items=500, n_events=20000, seed=42)
    train, test = gen.generate_train_test(test_ratio=0.2)
    print(f"    Train events  : {len(train)}")
    print(f"    Test events   : {len(test)}")
    print(f"    Platforms     : {len(set(e['platform'] for e in train))}")
    print(f"    Generated in : {time.time()-t0:.2f}s")

    # 4. Build & fit engine
    print("\n[4] Training recommendation engine (5 models)...")
    t0 = time.time()
    engine = RecommendationEngine(n_aspects=5000, random_state=42)
    engine.fit(train)
    print(f"    Fitted in     : {time.time()-t0:.2f}s")

    # 5. Engine summary
    print("\n[5] Engine summary:")
    s = engine.summary()
    print(f"    Aspects       : {s['n_aspects']}")
    print(f"    Models        : {s['n_models']} ({', '.join(engine.models.keys())})")
    print(f"    Users         : {s['n_users']}")
    print(f"    Items         : {s['n_items']}")
    print(f"    Events        : {s['n_events']}")
    print(f"    Platforms     : {len(s['platforms'])}")

    # 6. Recommendations
    print("\n[6] Sample recommendations:")
    sample_users = list(engine.user_events.keys())[:3]
    for uid in sample_users:
        recs = engine.recommend(uid, top_k=5)
        print(f"\n    User {uid}:")
        for r in recs:
            print(f"      {r['item_id']:>6}  score={r['score']:+.4f}  "
                  f"platform={r['platform']:10}  reasons={', '.join(r['reasons'])}")

    # 7. Platform-specific
    print("\n[7] Platform-specific recommendations (user %s, YouTube):" % sample_users[0])
    recs = engine.recommend(sample_users[0], top_k=5, platform="youtube")
    for r in recs:
        print(f"      {r['item_id']:>6}  score={r['score']:+.4f}  platform={r['platform']}")

    # 8. Evaluation
    print("\n[8] Evaluation on held-out test set:")
    t0 = time.time()
    metrics = engine.evaluate(test, k=10)
    for k, v in metrics.items():
        print(f"    {k:16} {v:.4f}" if isinstance(v, float) else f"    {k:16} {v}")
    print(f"    Evaluated in  : {time.time()-t0:.2f}s")

    # 9. Feature vector demo
    print("\n[9] Feature extraction demo:")
    from recommender.features import FeatureExtractor
    ext = FeatureExtractor(reg)
    sample_event = train[0]
    vec = ext.transform(sample_event)
    print(f"    Event platform={sample_event['platform']} action={sample_event['action']}")
    print(f"    Vector dim    : {vec.shape[0]}")
    print(f"    Non-zero      : {np.count_nonzero(vec)}")
    print(f"    Sparsity      : {1 - np.count_nonzero(vec)/len(vec):.4f}")

    print("\n" + "=" * 70)
    print("  OmniRec demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
