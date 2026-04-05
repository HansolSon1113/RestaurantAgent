from __future__ import annotations

import json

import requests

from app.config import Settings
from app.core.interfaces import AIReasoner
from app.core.models import ScoredRecommendation, SearchQuery


class ExternalAIReasoner(AIReasoner):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def explain(
        self,
        query: SearchQuery,
        recommendations: list[ScoredRecommendation],
    ) -> list[ScoredRecommendation]:
        if not recommendations:
            return recommendations

        if not self._settings.ai_api_key:
            return self._fallback_explanations(recommendations)

        try:
            prompt_payload = [
                {
                    "id": r.restaurant.restaurant_id,
                    "name": r.restaurant.name,
                    "category": r.restaurant.category,
                    "city": r.restaurant.city,
                    "rating": r.restaurant.rating,
                    "review_count": r.restaurant.review_count,
                    "warnings": r.warnings,
                    "score": round(r.final_score, 3),
                }
                for r in recommendations
            ]

            response = requests.post(
                f"{self._settings.ai_base_url.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._settings.ai_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._settings.ai_model,
                    "temperature": 0.2,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a restaurant recommendation reasoning assistant. "
                                "Return strict JSON with this schema: "
                                "{\"reasons\": [{\"id\": str, \"reason\": str, \"risk_note\": str}]}"
                            ),
                        },
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "query": {
                                        "where": query.where,
                                        "when": query.when.isoformat() if query.when else None,
                                        "category": query.category,
                                        "price_preference": query.price_preference,
                                        "travel_plan": query.travel_plan,
                                        "party_size": query.party_size,
                                    },
                                    "candidates": prompt_payload,
                                }
                            ),
                        },
                    ],
                },
                timeout=self._settings.request_timeout_seconds,
            )
            response.raise_for_status()
            content = (
                response.json()
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            parsed = json.loads(content)
            reason_map = {item["id"]: item for item in parsed.get("reasons", [])}

            for rec in recommendations:
                item = reason_map.get(rec.restaurant.restaurant_id)
                if not item:
                    continue
                rec.reason = item.get("reason", "")
                risk_note = item.get("risk_note", "")
                if risk_note:
                    rec.warnings.append(risk_note)
            return recommendations
        except Exception:
            return self._fallback_explanations(recommendations)

    def _fallback_explanations(self, recommendations: list[ScoredRecommendation]) -> list[ScoredRecommendation]:
        for rec in recommendations:
            category = rec.restaurant.category
            city = rec.restaurant.city
            rec.reason = (
                f"Strong match for {category} in {city} based on quality, review confidence, "
                "distance, and your historical preferences."
            )
        return recommendations
