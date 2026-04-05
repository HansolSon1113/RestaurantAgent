from __future__ import annotations

from app.core.models import Restaurant, UserPreferenceProfile


class ScoringService:
    def preference_score(self, restaurant: Restaurant, profile: UserPreferenceProfile) -> float:
        score = 0.0
        score += profile.category_affinity.get(restaurant.category, 0) * 0.12
        score -= profile.category_avoidance.get(restaurant.category, 0) * 0.2
        score += profile.price_affinity.get(restaurant.price_level, 0) * 0.08
        return max(-1.0, min(1.0, score))

    def base_score(self, restaurant: Restaurant) -> float:
        normalized_rating = restaurant.rating / 5.0
        review_confidence = min(1.0, restaurant.review_count / 250.0)
        distance_component = max(0.0, 1.0 - (restaurant.distance_km / 20.0))
        return (normalized_rating * 0.55) + (review_confidence * 0.25) + (distance_component * 0.2)
