"""Base class for all recommendation models."""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Optional


class BaseModel(ABC):
    """Every model scores items for a user. Higher score = better recommendation."""

    name = "base"

    def __init__(self, weight: float = 1.0):
        self.weight = weight
        self.is_fitted = False

    @abstractmethod
    def fit(self, user_events: Dict[str, List[Dict]], **kwargs) -> "BaseModel":
        """Train on a mapping of user_id -> list of event dicts."""
        ...

    @abstractmethod
    def score(self, user_id: str, item_ids: List[str],
              user_profile: Optional[np.ndarray] = None) -> np.ndarray:
        """Return a score for each item_id."""
        ...

    def rank(self, user_id: str, item_ids: List[str],
             user_profile: Optional[np.ndarray] = None,
             top_k: int = 10) -> List[str]:
        scores = self.score(user_id, item_ids, user_profile)
        order = np.argsort(-scores)[:top_k]
        return [item_ids[i] for i in order]
