from app.config import settings
from app.core.engine import RecommendationEngine
from app.core.models import SearchQuery
from app.services.ai_reasoner import ExternalAIReasoner
from app.services.fraud_detector import HeuristicFraudDetector
from app.services.preference_store import JsonPreferenceStore
from app.services.scoring import ScoringService
from app.sources.local_sample import LocalSampleSource


def build_test_engine(tmp_path):
    preference_file = tmp_path / "preferences.json"
    return RecommendationEngine(
        sources=[LocalSampleSource()],
        preference_store=JsonPreferenceStore(str(preference_file)),
        fraud_detector=HeuristicFraudDetector(),
        ai_reasoner=ExternalAIReasoner(settings),
        scoring_service=ScoringService(),
    )


def test_recommendation_returns_results(tmp_path):
    engine = build_test_engine(tmp_path)
    query = SearchQuery(where="Seoul", category="korean")
    recs = engine.recommend(user_id="u1", query=query, top_n=3)

    assert recs
    assert recs[0].restaurant.city.lower() == "seoul"


def test_feedback_updates_preferences(tmp_path):
    engine = build_test_engine(tmp_path)
    engine.record_feedback("u2", accepted=True, category="korean", price_level="$$")

    query = SearchQuery(where="Seoul", category="korean")
    recs = engine.recommend(user_id="u2", query=query, top_n=3)
    assert recs
