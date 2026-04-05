from __future__ import annotations

import json
from dataclasses import replace

import app.services.ai_reasoner as ai_reasoner_module
from app.config import settings
from app.core.models import Restaurant, ScoredRecommendation, SearchQuery
from app.services.ai_reasoner import ExternalAIReasoner


class _MockResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_ai_reasoner_uses_mock_ai_api(monkeypatch):
    test_settings = replace(settings, ai_api_key="fake-key", ai_base_url="https://mock-ai.local")
    reasoner = ExternalAIReasoner(test_settings)

    rec = ScoredRecommendation(
        restaurant=Restaurant(
            restaurant_id="r-1",
            name="Mock House",
            city="Seoul",
            category="korean",
            rating=4.7,
            review_count=180,
            price_level="$$",
            distance_km=1.5,
            source="local-sample",
        ),
        base_score=0.8,
        preference_score=0.1,
        fraud_risk_score=0.05,
        final_score=0.88,
        warnings=[],
    )

    def _mock_post(url, headers, json, timeout):
        assert url.endswith("/chat/completions")
        assert headers["Authorization"] == "Bearer fake-key"
        assert json["model"] == test_settings.ai_model
        return _MockResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json_module.dumps(
                                {
                                    "reasons": [
                                        {
                                            "id": "r-1",
                                            "reason": "Great fit for your Korean dinner plan.",
                                            "risk_note": "Minor viral buzz, but sentiment remains stable.",
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ]
            }
        )

    json_module = json
    monkeypatch.setattr(ai_reasoner_module.requests, "post", _mock_post)

    out = reasoner.explain(SearchQuery(where="Seoul", category="korean"), [rec])

    assert out[0].reason == "Great fit for your Korean dinner plan."
    assert "Minor viral buzz" in out[0].warnings[0]


def test_ai_reasoner_falls_back_when_ai_fails(monkeypatch):
    test_settings = replace(settings, ai_api_key="fake-key", ai_base_url="https://mock-ai.local")
    reasoner = ExternalAIReasoner(test_settings)

    rec = ScoredRecommendation(
        restaurant=Restaurant(
            restaurant_id="r-2",
            name="Fallback Bistro",
            city="Tokyo",
            category="japanese",
            rating=4.5,
            review_count=120,
            price_level="$$",
            distance_km=3.0,
            source="local-sample",
        ),
        base_score=0.7,
        preference_score=0.05,
        fraud_risk_score=0.0,
        final_score=0.75,
        warnings=[],
    )

    def _explode(*args, **kwargs):
        raise RuntimeError("mocked network failure")

    monkeypatch.setattr(ai_reasoner_module.requests, "post", _explode)

    out = reasoner.explain(SearchQuery(where="Tokyo", category="japanese"), [rec])

    assert "Strong match for japanese in Tokyo" in out[0].reason
