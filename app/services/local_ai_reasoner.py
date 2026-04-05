from __future__ import annotations

from app.core.interfaces import AIReasoner
from app.core.models import ScoredRecommendation, SearchQuery


class LocalAIReasoner(AIReasoner):
    def explain(
        self,
        query: SearchQuery,
        recommendations: list[ScoredRecommendation],
    ) -> list[ScoredRecommendation]:
        for rec in recommendations:
            warnings = ", ".join(rec.warnings) if rec.warnings else "no major risk signals"
            rec.reason = (
                f"Best fit for {query.where} because it matches {rec.restaurant.category}, "
                f"scores well on quality and distance, and has {warnings}."
            )
        return recommendations
