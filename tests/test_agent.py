"""Tests for RestaurantAgent — the LLM-driven agentic recommendation path."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

import app.core.agent as agent_module
from app.config import settings
from app.core.agent import AgentResult, RestaurantAgent
from app.core.models import AgentStep
from app.services.fraud_detector import HeuristicFraudDetector
from app.services.preference_store import JsonPreferenceStore
from app.services.scoring import ScoringService
from app.sources.local_sample import LocalSampleSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_agent(tmp_path, extra_settings=None):
    pref_file = tmp_path / "preferences.json"
    s = replace(settings, preferences_file=str(pref_file))
    if extra_settings:
        s = replace(s, **extra_settings)
    return RestaurantAgent(
        sources=[LocalSampleSource()],
        preference_store=JsonPreferenceStore(str(pref_file)),
        fraud_detector=HeuristicFraudDetector(),
        scoring_service=ScoringService(),
        settings=s,
    )


class _MockResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


# ---------------------------------------------------------------------------
# Tests — no AI configured → RuntimeError, not a silent fallback
# ---------------------------------------------------------------------------

def test_agent_raises_without_ai_configured(tmp_path):
    ag = _build_agent(tmp_path, {"ai_api_key": "", "ai_base_url": ""})

    from app.core.models import SearchQuery

    with pytest.raises(RuntimeError, match="AI is not configured"):
        ag.run(user_id="u1", query=SearchQuery(where="Seoul", category="korean"), top_n=3)


# ---------------------------------------------------------------------------
# Tests — AI key present, LLM calls tools then returns final answer
# ---------------------------------------------------------------------------

def test_agent_executes_tool_calls_then_stops(monkeypatch, tmp_path):
    """Simulate LLM that calls search_restaurants, score_and_rank, then stops."""
    from app.core.models import SearchQuery

    ag = _build_agent(
        tmp_path,
        {"ai_api_key": "fake-key", "ai_base_url": "https://mock-ai.local"},
    )

    # Discover the restaurant_id we'll use in the final answer
    from app.sources.local_sample import LocalSampleSource
    from app.core.models import SearchQuery as SQ
    candidates = LocalSampleSource().search(SQ(where="Seoul", category="korean"))
    assert candidates, "LocalSampleSource must return at least one Seoul/korean restaurant"
    first = candidates[0]

    call_count = [0]
    import json as json_mod

    def _mock_post(url, headers, json, timeout):
        call_count[0] += 1

        if call_count[0] == 1:
            # First call: LLM searches for restaurants (populates the cache)
            return _MockResponse(
                {
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "search_restaurants",
                                            "arguments": json_mod.dumps(
                                                {"city": "Seoul", "category": "korean"}
                                            ),
                                        },
                                    }
                                ],
                            },
                        }
                    ]
                }
            )

        if call_count[0] == 2:
            # Second call: LLM scores the results
            return _MockResponse(
                {
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_2",
                                        "type": "function",
                                        "function": {
                                            "name": "score_and_rank",
                                            "arguments": json_mod.dumps(
                                                {
                                                    "user_id": "u_agent",
                                                    "restaurant_ids": [first.restaurant_id],
                                                    "top_n": 2,
                                                }
                                            ),
                                        },
                                    }
                                ],
                            },
                        }
                    ]
                }
            )

        # Third call: LLM returns final answer
        final = json_mod.dumps(
            [
                {
                    "restaurant_id": first.restaurant_id,
                    "name": first.name,
                    "city": first.city,
                    "category": first.category,
                    "rating": first.rating,
                    "price_level": first.price_level,
                    "reason": "Great Korean food with strong reviews.",
                    "warnings": [],
                }
            ]
        )
        return _MockResponse(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": final,
                        },
                    }
                ]
            }
        )

    monkeypatch.setattr(agent_module.requests, "post", _mock_post)

    result = ag.run(
        user_id="u_agent",
        query=SearchQuery(where="Seoul", category="korean"),
        top_n=2,
    )

    assert result.used_agent is True
    assert result.recommendations
    assert result.recommendations[0].reason  # LLM-provided reason must be non-empty
    # Two steps recorded: search_restaurants + score_and_rank
    assert len(result.steps) == 2
    assert result.steps[0].tool == "search_restaurants"
    assert result.steps[1].tool == "score_and_rank"


# ---------------------------------------------------------------------------
# Tests — AI key present, LLM API fails → RuntimeError, not a silent fallback
# ---------------------------------------------------------------------------

def test_agent_raises_on_api_error(monkeypatch, tmp_path):
    from app.core.models import SearchQuery

    ag = _build_agent(
        tmp_path,
        {"ai_api_key": "fake-key", "ai_base_url": "https://mock-ai.local"},
    )

    def _explode(*args, **kwargs):
        raise RuntimeError("mocked network failure")

    monkeypatch.setattr(agent_module.requests, "post", _explode)

    with pytest.raises(RuntimeError, match="AI API call failed"):
        ag.run(user_id="u3", query=SearchQuery(where="Seoul"), top_n=3)


# ---------------------------------------------------------------------------
# Tests — individual tools
# ---------------------------------------------------------------------------

def test_tool_search_restaurants_populates_cache(tmp_path):
    ag = _build_agent(tmp_path)
    results = ag._tool_search_restaurants(city="Seoul", category="korean")

    assert results
    assert all("restaurant_id" in r for r in results)
    # Cache should now contain every found restaurant
    assert len(ag._candidate_cache) == len(results)


def test_tool_get_user_preferences_returns_profile(tmp_path):
    ag = _build_agent(tmp_path)
    pref = ag._tool_get_user_preferences(user_id="new_user")

    assert pref["user_id"] == "new_user"
    assert "category_affinity" in pref
    assert "price_affinity" in pref


def test_tool_assess_fraud_risk_unknown_id(tmp_path):
    ag = _build_agent(tmp_path)
    result = ag._tool_assess_fraud_risk(restaurant_id="nonexistent")

    assert "error" in result


def test_tool_score_and_rank_returns_sorted_results(tmp_path):
    ag = _build_agent(tmp_path)
    # Populate the cache first via search
    found = ag._tool_search_restaurants(city="Seoul")
    ids = [r["restaurant_id"] for r in found]

    ranked = ag._tool_score_and_rank(user_id="u_rank", restaurant_ids=ids, top_n=2)

    assert len(ranked) <= 2
    if len(ranked) == 2:
        assert ranked[0]["final_score"] >= ranked[1]["final_score"]


# ---------------------------------------------------------------------------
# Tests — /agent/recommend API endpoint
# ---------------------------------------------------------------------------

def test_agent_recommend_endpoint_returns_503_without_ai(tmp_path, monkeypatch):
    """Integration test: /agent/recommend returns 503 when AI is not configured."""
    from fastapi.testclient import TestClient

    import app.main as main_module

    pref_file = tmp_path / "preferences.json"
    test_settings = replace(settings, ai_api_key="", ai_base_url="", preferences_file=str(pref_file))

    monkeypatch.setattr(main_module, "agent", RestaurantAgent(
        sources=[LocalSampleSource()],
        preference_store=JsonPreferenceStore(str(pref_file)),
        fraud_detector=HeuristicFraudDetector(),
        scoring_service=ScoringService(),
        settings=test_settings,
    ))

    client = TestClient(main_module.app, raise_server_exceptions=False)
    response = client.post(
        "/agent/recommend",
        json={"user_id": "u_api", "where": "Seoul", "category": "korean", "top_n": 3},
    )

    assert response.status_code == 503
    assert "AI is not configured" in response.json()["detail"]
