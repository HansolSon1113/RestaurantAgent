from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class SearchQuery:
    where: str
    when: datetime | None = None
    category: str | None = None
    price_preference: str | None = None
    travel_plan: str | None = None
    party_size: int = 1


@dataclass
class Restaurant:
    restaurant_id: str
    name: str
    city: str
    category: str
    rating: float
    review_count: int
    price_level: str
    distance_km: float
    source: str
    url: str | None = None
    is_sponsored: bool = False
    last_updated: datetime | None = None
    recent_mentions_24h: int = 0
    negative_mentions_24h: int = 0
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class UserPreferenceProfile:
    user_id: str
    category_affinity: dict[str, int] = field(default_factory=dict)
    category_avoidance: dict[str, int] = field(default_factory=dict)
    price_affinity: dict[str, int] = field(default_factory=dict)


@dataclass
class FraudAssessment:
    risk_score: float
    warnings: list[str]


@dataclass
class ScoredRecommendation:
    restaurant: Restaurant
    base_score: float
    preference_score: float
    fraud_risk_score: float
    final_score: float
    warnings: list[str] = field(default_factory=list)
    reason: str = ""
