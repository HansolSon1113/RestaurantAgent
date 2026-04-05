from __future__ import annotations

from app.core.interfaces import FraudDetector
from app.core.models import FraudAssessment, Restaurant


class HeuristicFraudDetector(FraudDetector):
    def assess(self, restaurant: Restaurant) -> FraudAssessment:
        risk = 0.0
        warnings: list[str] = []

        if restaurant.rating >= 4.8 and restaurant.review_count < 25:
            risk += 0.35
            warnings.append("Unusually high rating with low review volume.")

        mention_delta = restaurant.recent_mentions_24h - restaurant.negative_mentions_24h
        if restaurant.recent_mentions_24h > 180 and mention_delta < 60:
            risk += 0.25
            warnings.append("Potential viral spike with uncertain sentiment.")

        if restaurant.recent_mentions_24h > 0:
            negative_ratio = restaurant.negative_mentions_24h / restaurant.recent_mentions_24h
            if negative_ratio > 0.4:
                risk += 0.35
                warnings.append("Recent negative mentions are high.")

        if restaurant.is_sponsored and restaurant.review_count < 30:
            risk += 0.15
            warnings.append("Sponsored listing with limited organic validation.")

        return FraudAssessment(risk_score=min(1.0, risk), warnings=warnings)
