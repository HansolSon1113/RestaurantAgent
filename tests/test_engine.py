import app.services.ai_reasoner as ai_reasoner_module
from app.config import settings
from app.core.engine import RecommendationEngine
from app.core.models import SearchQuery
from app.services.ai_reasoner import ExternalAIReasoner
from app.services.fraud_detector import HeuristicFraudDetector
from app.services.preference_store import JsonPreferenceStore
from app.services.scoring import ScoringService
from app.sources.local_sample import LocalSampleSource


class _MockAIResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        import json
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"reasons": []}
                        )
                    }
                }
            ]
        }


def build_test_engine(tmp_path, monkeypatch):
    preference_file = tmp_path / "preferences.json"
    engine = RecommendationEngine(
        sources=[LocalSampleSource()],
        preference_store=JsonPreferenceStore(str(preference_file)),
        fraud_detector=HeuristicFraudDetector(),
        ai_reasoner=ExternalAIReasoner(
            __import__("dataclasses").replace(settings, ai_api_key="fake-key", ai_base_url="https://mock-ai.local")
        ),
        scoring_service=ScoringService(),
    )
    monkeypatch.setattr(ai_reasoner_module.requests, "post", lambda *a, **kw: _MockAIResponse())
    return engine


def test_recommendation_returns_results(tmp_path, monkeypatch):
    engine = build_test_engine(tmp_path, monkeypatch)
    query = SearchQuery(where="Seoul", category="korean")
    recs = engine.recommend(user_id="u1", query=query, top_n=3)

    assert recs
    assert recs[0].restaurant.city.lower() == "seoul"


def test_feedback_updates_preferences(tmp_path, monkeypatch):
    engine = build_test_engine(tmp_path, monkeypatch)
    engine.record_feedback("u2", accepted=True, category="korean", price_level="$$")

    query = SearchQuery(where="Seoul", category="korean")
    recs = engine.recommend(user_id="u2", query=query, top_n=3)
    assert recs
