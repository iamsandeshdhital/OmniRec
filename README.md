# OmniRec — Multi-Platform Recommendation Engine

A full-fledged recommendation system for **Facebook, Instagram, Snapchat,
YouTube, Twitter/X, Netflix, TikTok, Twitch, Reddit, LinkedIn, Pinterest,
Spotify, Discord** and any other social platform — built on **5000+ aspects
and variables**.

## What makes it different

| Capability | Details |
|---|---|
| **5000+ aspects** | Combinatorial registry across 12 categories: content, behavior, context, temporal, social-graph, psychographic, cross-platform, creator, engagement, monetization, privacy/safety |
| **5 models** | Content-based, Collaborative (MF), Neural (MLP), Graph (PPR), Bandit (UCB1) |
| **Ensemble fusion** | Weighted combination of all model scores |
| **Diversity** | MMR re-ranking to avoid filter bubbles |
| **Exploration** | Bandit-driven cold-start and serendipity |
| **Platform-aware** | Per-platform content types, surfaces, signals |
| **Explainable** | Every recommendation carries human-readable reasons |

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

## Architecture

```
events → FeatureExtractor (5000+ dim vectors)
       → 5 parallel models
       → weighted ensemble
       → MMR diversity re-rank
       → freshness + exploration boost
       → ranked recommendations
```

## Modules

- `recommender/aspects.py` — 5000+ aspect registry (combinatorial generation)
- `recommender/platforms.py` — 13 platform definitions
- `recommender/features.py` — event → dense aspect vector
- `recommender/models/` — content, collaborative, neural, graph, bandit
- `recommender/engine.py` — ensemble orchestrator
- `recommender/evaluation.py` — precision@k, recall@k, NDCG@k
- `recommender/data.py` — synthetic multi-platform data generator
- `main.py` — end-to-end demo

## API

```python
from recommender import RecommendationEngine

engine = RecommendationEngine(n_aspects=5000)
engine.fit(events)                          # train on event log
recs = engine.recommend("user_42", top_k=10)
# [{"item_id": "i123", "platform": "youtube", "score": 0.87, "reasons": [...]}]

metrics = engine.evaluate(test_events, k=10)
# {"precision@10": 0.32, "recall@10": 0.18, "ndcg@10": 0.41}
```

## Extending

Add a new platform in `platforms.py`, a new aspect category in `aspects.py`,
or a new model by subclassing `BaseModel` and registering it in `engine.py`.
