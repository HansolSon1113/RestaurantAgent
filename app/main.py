from __future__ import annotations

from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.config import settings
from app.core.agent import RestaurantAgent
from app.core.engine import RecommendationEngine
from app.core.models import SearchQuery
from app.services.ai_reasoner import ExternalAIReasoner
from app.services.llm_fraud_detector import LLMFraudDetector
from app.services.preference_store import JsonPreferenceStore
from app.services.scoring import ScoringService
from app.sources.foursquare_source import FoursquareSource
from app.sources.google_places_source import GooglePlacesSource
from app.sources.local_sample import LocalSampleSource
from app.sources.yelp_source import YelpSource


class RecommendationRequest(BaseModel):
    user_id: str = Field(min_length=1)
    where: str = Field(min_length=1)
    when: datetime | None = None
    category: str | None = None
    price_preference: str | None = None
    travel_plan: str | None = None
    party_size: int = Field(default=1, ge=1)
    top_n: int = Field(default=5, ge=1, le=20)


class FeedbackRequest(BaseModel):
    user_id: str = Field(min_length=1)
    accepted: bool
    category: str | None = None
    price_level: str | None = None


def _shared_sources():
    return [
        YelpSource(settings),
        GooglePlacesSource(settings),
        FoursquareSource(settings),
        LocalSampleSource(),
    ]


def build_engine() -> RecommendationEngine:
    return RecommendationEngine(
        sources=_shared_sources(),
        preference_store=JsonPreferenceStore(settings.preferences_file),
        fraud_detector=LLMFraudDetector(settings),
        ai_reasoner=ExternalAIReasoner(settings),
        scoring_service=ScoringService(),
    )


def build_agent() -> RestaurantAgent:
    return RestaurantAgent(
        sources=_shared_sources(),
        preference_store=JsonPreferenceStore(settings.preferences_file),
        fraud_detector=LLMFraudDetector(settings),
        scoring_service=ScoringService(),
        settings=settings,
    )


app = FastAPI(
    title="Restaurant Agent",
    description=(
        "AI-agent-powered restaurant recommendation engine with preference learning, "
        "multi-source retrieval, LLM-driven tool calling, and fraud/viral risk warnings."
    ),
    version="0.2.0",
)

engine = build_engine()
agent = build_agent()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/recommend")
def recommend(payload: RecommendationRequest) -> dict:
    query = SearchQuery(
        where=payload.where,
        when=payload.when,
        category=payload.category,
        price_preference=payload.price_preference,
        travel_plan=payload.travel_plan,
        party_size=payload.party_size,
    )
    recs = engine.recommend(user_id=payload.user_id, query=query, top_n=payload.top_n)
    return {
        "best_match": _to_output(recs[0]) if recs else None,
        "recommendations": [_to_output(rec) for rec in recs],
    }


@app.post("/feedback")
def feedback(payload: FeedbackRequest) -> dict[str, str]:
    engine.record_feedback(
        user_id=payload.user_id,
        accepted=payload.accepted,
        category=payload.category,
        price_level=payload.price_level,
    )
    return {"status": "saved"}


@app.post("/agent/recommend")
def agent_recommend(payload: RecommendationRequest) -> dict:
    """LLM-driven agentic recommendation endpoint.

    Uses ``RestaurantAgent`` which lets the LLM autonomously decide which
    tools to call (search, fraud-check, score-and-rank) and in what order.
    Falls back to the deterministic pipeline when no AI key is configured.
    """
    query = SearchQuery(
        where=payload.where,
        when=payload.when,
        category=payload.category,
        price_preference=payload.price_preference,
        travel_plan=payload.travel_plan,
        party_size=payload.party_size,
    )
    result = agent.run(user_id=payload.user_id, query=query, top_n=payload.top_n)
    return {
        "used_agent": result.used_agent,
        "agent_steps": [
            {
                "step": s.step,
                "tool": s.tool,
                "inputs": s.inputs,
                "result": s.result,
            }
            for s in result.steps
        ],
        "best_match": _to_output(result.recommendations[0]) if result.recommendations else None,
        "recommendations": [_to_output(rec) for rec in result.recommendations],
    }


def _to_output(rec):
    return {
        "restaurant_id": rec.restaurant.restaurant_id,
        "name": rec.restaurant.name,
        "city": rec.restaurant.city,
        "category": rec.restaurant.category,
        "rating": rec.restaurant.rating,
        "review_count": rec.restaurant.review_count,
        "price_level": rec.restaurant.price_level,
        "distance_km": rec.restaurant.distance_km,
        "source": rec.restaurant.source,
        "score": round(rec.final_score, 4),
        "fraud_risk": round(rec.fraud_risk_score, 4),
        "warnings": rec.warnings,
        "reason": rec.reason,
        "url": rec.restaurant.url,
    }
