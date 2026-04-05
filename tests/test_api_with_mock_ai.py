from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

import app.services.ai_reasoner as ai_reasoner_module
import app.main as main_module
from app.config import settings
from app.core.engine import RecommendationEngine
from app.services.ai_reasoner import ExternalAIReasoner
from app.services.fraud_detector import HeuristicFraudDetector
from app.services.preference_store import JsonPreferenceStore
from app.services.scoring import ScoringService
from app.sources.local_sample import LocalSampleSource


class _MockResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "reasons": [
                                    {
                                        "id": "local-seoul-1",
                                        "reason": "Balanced quality and distance for your Seoul plan.",
                                        "risk_note": "No critical risk signals.",
                                    }
                                ]
                            }
                        )
                    }
                }
            ]
        }


def test_recommend_endpoint_with_mock_ai(monkeypatch):
    test_engine = RecommendationEngine(
        sources=[LocalSampleSource()],
        preference_store=JsonPreferenceStore(str(Path("data") / "test_preferences.json")),
        fraud_detector=HeuristicFraudDetector(),
        ai_reasoner=ExternalAIReasoner(replace(settings, ai_api_key="fake-key", ai_base_url="https://mock-ai.local")),
        scoring_service=ScoringService(),
    )
    monkeypatch.setattr(main_module, "engine", test_engine)

    def _mock_post(*args, **kwargs):
        return _MockResponse()

    monkeypatch.setattr(ai_reasoner_module.requests, "post", _mock_post)

    client = TestClient(main_module.app)
    response = client.post(
        "/recommend",
        json={
            "user_id": "api-user-1",
            "where": "Seoul",
            "category": "korean",
            "top_n": 2,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["best_match"] is not None
    assert payload["recommendations"]
    assert payload["recommendations"][0]["reason"] != ""
