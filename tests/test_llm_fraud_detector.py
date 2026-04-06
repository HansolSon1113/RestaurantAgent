"""Tests for LLMFraudDetector — CoT-based fraud detection."""

from __future__ import annotations

import json
from dataclasses import replace

import app.services.llm_fraud_detector as llm_fd_module
from app.config import settings
from app.core.models import Restaurant
from app.services.llm_fraud_detector import LLMFraudDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_restaurant(**kwargs) -> Restaurant:
    defaults = dict(
        restaurant_id="r-test",
        name="Test Place",
        city="Seoul",
        category="korean",
        rating=4.5,
        review_count=100,
        price_level="$$",
        distance_km=1.0,
        source="local-sample",
        is_sponsored=False,
        recent_mentions_24h=0,
        negative_mentions_24h=0,
    )
    defaults.update(kwargs)
    return Restaurant(**defaults)


class _MockResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


# ---------------------------------------------------------------------------
# Tests — no API key: raises RuntimeError (no heuristic fallback)
# ---------------------------------------------------------------------------

def test_llm_fraud_detector_raises_without_ai_configured():
    import pytest
    test_settings = replace(settings, ai_api_key="", ai_base_url="")
    detector = LLMFraudDetector(test_settings)

    restaurant = _make_restaurant(rating=4.9, review_count=5)
    with pytest.raises(RuntimeError, match="AI is not configured"):
        detector.assess(restaurant)


# ---------------------------------------------------------------------------
# Tests — API key present, LLM returns CoT response
# ---------------------------------------------------------------------------

def test_llm_fraud_detector_uses_llm_when_api_key_set(monkeypatch):
    import json as json_module

    test_settings = replace(
        settings, ai_api_key="fake-key", ai_base_url="https://mock-ai.local"
    )
    detector = LLMFraudDetector(test_settings)
    restaurant = _make_restaurant(
        rating=4.9, review_count=5, recent_mentions_24h=200, negative_mentions_24h=90
    )

    llm_result = {
        "thinking": (
            "1. Rating 4.9 with only 5 reviews is suspicious. "
            "2. 200 recent mentions with 90 negative (45%) is a red flag. "
            "3. Not sponsored. Conclusion: high risk."
        ),
        "risk_score": 0.75,
        "warnings": [
            "Unusually high rating with very few reviews.",
            "High negative mention ratio in last 24 hours.",
        ],
    }

    def _mock_post(url, headers, json, timeout):
        assert "chat/completions" in url
        assert headers["Authorization"] == "Bearer fake-key"
        messages = json["messages"]
        # System prompt should mention chain-of-thought
        assert any("chain-of-thought" in m["content"].lower() for m in messages if m["role"] == "system")
        return _MockResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json_module.dumps(llm_result)
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(llm_fd_module.requests, "post", _mock_post)

    assessment = detector.assess(restaurant)

    assert abs(assessment.risk_score - 0.75) < 1e-6
    assert len(assessment.warnings) == 2
    assert "high rating" in assessment.warnings[0].lower()
    assert "negative mention" in assessment.warnings[1].lower()


# ---------------------------------------------------------------------------
# Tests — LLM call fails: raises RuntimeError (no heuristic fallback)
# ---------------------------------------------------------------------------

def test_llm_fraud_detector_raises_on_llm_error(monkeypatch):
    import pytest
    test_settings = replace(
        settings, ai_api_key="fake-key", ai_base_url="https://mock-ai.local"
    )
    detector = LLMFraudDetector(test_settings)

    def _explode(*args, **kwargs):
        raise RuntimeError("mocked network failure")

    monkeypatch.setattr(llm_fd_module.requests, "post", _explode)

    restaurant = _make_restaurant(rating=4.9, review_count=5)
    with pytest.raises(RuntimeError, match="LLM fraud detection failed"):
        detector.assess(restaurant)


# ---------------------------------------------------------------------------
# Tests — LLM returns bad JSON: raises RuntimeError (no heuristic fallback)
# ---------------------------------------------------------------------------

def test_llm_fraud_detector_raises_on_bad_json(monkeypatch):
    import pytest
    test_settings = replace(
        settings, ai_api_key="fake-key", ai_base_url="https://mock-ai.local"
    )
    detector = LLMFraudDetector(test_settings)
    restaurant = _make_restaurant(rating=4.9, review_count=5)

    def _mock_bad_json(url, headers, json, timeout):
        return _MockResponse(
            {
                "choices": [
                    {"message": {"content": "This is not valid JSON at all."}}
                ]
            }
        )

    monkeypatch.setattr(llm_fd_module.requests, "post", _mock_bad_json)

    with pytest.raises(RuntimeError, match="LLM fraud detection failed"):
        detector.assess(restaurant)


# ---------------------------------------------------------------------------
# Tests — risk_score is clamped to [0.0, 1.0]
# ---------------------------------------------------------------------------

def test_llm_fraud_detector_clamps_risk_score(monkeypatch):
    import json as json_module

    test_settings = replace(
        settings, ai_api_key="fake-key", ai_base_url="https://mock-ai.local"
    )
    detector = LLMFraudDetector(test_settings)
    restaurant = _make_restaurant()

    def _mock_extreme(url, headers, json, timeout):
        return _MockResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json_module.dumps(
                                {"thinking": "extreme", "risk_score": 99.9, "warnings": []}
                            )
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(llm_fd_module.requests, "post", _mock_extreme)

    assessment = detector.assess(restaurant)
    assert assessment.risk_score == 1.0
