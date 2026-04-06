from __future__ import annotations

from app.core.interfaces import AIReasoner, FraudDetector, PreferenceStore, RestaurantDataSource
from app.core.models import Restaurant, ScoredRecommendation, SearchQuery
from app.services.scoring import FRAUD_RISK_PENALTY, ScoringService


class RecommendationEngine:
    def __init__(
        self,
        sources: list[RestaurantDataSource],
        preference_store: PreferenceStore,
        fraud_detector: FraudDetector,
        ai_reasoner: AIReasoner,
        scoring_service: ScoringService,
    ) -> None:
        self._sources = sources
        self._preference_store = preference_store
        self._fraud_detector = fraud_detector
        self._ai_reasoner = ai_reasoner
        self._scoring_service = scoring_service

    def recommend(self, user_id: str, query: SearchQuery, top_n: int = 5) -> list[ScoredRecommendation]:
        profile = self._preference_store.get_profile(user_id)
        candidates = self._collect_candidates(query)

        recommendations: list[ScoredRecommendation] = []
        for restaurant in candidates:
            fraud_assessment = self._fraud_detector.assess(restaurant)
            base_score = self._scoring_service.base_score(restaurant)
            preference_score = self._scoring_service.preference_score(restaurant, profile)
            final_score = base_score + preference_score - (fraud_assessment.risk_score * FRAUD_RISK_PENALTY)
            recommendations.append(
                ScoredRecommendation(
                    restaurant=restaurant,
                    base_score=base_score,
                    preference_score=preference_score,
                    fraud_risk_score=fraud_assessment.risk_score,
                    final_score=final_score,
                    warnings=fraud_assessment.warnings.copy(),
                )
            )

        recommendations.sort(key=lambda x: x.final_score, reverse=True)
        recommendations = recommendations[: max(1, top_n)]
        return self._ai_reasoner.explain(query, recommendations)

    def record_feedback(
        self,
        user_id: str,
        accepted: bool,
        category: str | None,
        price_level: str | None,
    ) -> None:
        self._preference_store.record_feedback(
            user_id=user_id,
            accepted=accepted,
            category=category,
            price_level=price_level,
        )

    def _collect_candidates(self, query: SearchQuery):
        dedup: dict[str, Restaurant] = {}
        for source in self._sources:
            try:
                for candidate in source.search(query):
                    key = f"{candidate.name.strip().lower()}::{candidate.city.strip().lower()}"
                    if key not in dedup:
                        dedup[key] = candidate
            except Exception:
                continue
        return list(dedup.values())
