"""
GenRec — A multi-platform, multi-aspect recommendation engine.

Operates on 5000+ aspects/variables across content, behavior, context,
social-graph, temporal, psychographic and cross-platform signal domains.
"""

__version__ = "1.0.0"

from .engine import RecommendationEngine
from .platforms import Platform, get_platform
from .aspects import AspectRegistry, ASPECT_CATEGORIES

__all__ = [
    "RecommendationEngine",
    "Platform",
    "get_platform",
    "AspectRegistry",
    "ASPECT_CATEGORIES",
]
