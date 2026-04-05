from __future__ import annotations

from abc import ABC, abstractmethod

from app.core.models import FraudAssessment, Restaurant, ScoredRecommendation, SearchQuery, UserPreferenceProfile


class RestaurantDataSource(ABC):
    @property
    @abstractmethod
    def source_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def search(self, query: SearchQuery) -> list[Restaurant]:
        raise NotImplementedError


class PreferenceStore(ABC):
    @abstractmethod
    def get_profile(self, user_id: str) -> UserPreferenceProfile:
        raise NotImplementedError

    @abstractmethod
    def record_feedback(
        self,
        user_id: str,
        accepted: bool,
        category: str | None,
        price_level: str | None,
    ) -> UserPreferenceProfile:
        raise NotImplementedError


class FraudDetector(ABC):
    @abstractmethod
    def assess(self, restaurant: Restaurant) -> FraudAssessment:
        raise NotImplementedError


class AIReasoner(ABC):
    @abstractmethod
    def explain(
        self,
        query: SearchQuery,
        recommendations: list[ScoredRecommendation],
    ) -> list[ScoredRecommendation]:
        raise NotImplementedError
