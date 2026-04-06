"""LLM-based fraud detector using Chain-of-Thought (CoT) reasoning.

Unlike the rule-based ``HeuristicFraudDetector``, this implementation asks
an LLM to reason step-by-step through each fraud signal before producing a
final ``risk_score`` and a list of human-readable ``warnings``.

Falls back to ``HeuristicFraudDetector`` when:
- No ``ai_api_key`` is configured in ``Settings``.
- The LLM call fails for any reason (network error, bad JSON, etc.).
"""

from __future__ import annotations

import json

import requests

from app.config import Settings
from app.core.interfaces import FraudDetector
from app.core.models import FraudAssessment, Restaurant
from app.services.fraud_detector import HeuristicFraudDetector


class LLMFraudDetector(FraudDetector):
    """Fraud detector that uses LLM Chain-of-Thought (CoT) reasoning.

    The LLM is given a structured prompt that instructs it to reason through
    each potential fraud signal one at a time before arriving at a final
    ``risk_score`` (0.0 = clean, 1.0 = high risk) and a list of warning
    strings.  The ``thinking`` field in the JSON response captures the full
    step-by-step reasoning trace.
    """

    _SYSTEM_PROMPT = (
        "You are a restaurant fraud and authenticity analyst. "
        "Given restaurant metrics, reason step-by-step (chain-of-thought) through each "
        "potential fraud signal before concluding. "
        "Signals to consider in order:\n"
        "  1. Rating vs review volume — is the rating unusually high for the number of reviews?\n"
        "  2. Viral/mention spike — are recent_mentions_24h suspiciously high relative to negative_mentions_24h?\n"
        "  3. Negative sentiment ratio — what fraction of recent mentions are negative?\n"
        "  4. Sponsored status — is this a sponsored listing with few organic reviews?\n"
        "After reasoning through all signals, output ONLY valid JSON with this schema:\n"
        '{"thinking": "<your step-by-step reasoning>", '
        '"risk_score": <float 0.0-1.0>, '
        '"warnings": ["<human-readable warning>", ...]}\n'
        "Use an empty warnings list and risk_score of 0.0 when there are no concerns."
    )

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._fallback = HeuristicFraudDetector()

    def assess(self, restaurant: Restaurant) -> FraudAssessment:
        if not self._settings.ai_api_key:
            return self._fallback.assess(restaurant)

        payload = {
            "restaurant_id": restaurant.restaurant_id,
            "name": restaurant.name,
            "rating": restaurant.rating,
            "review_count": restaurant.review_count,
            "recent_mentions_24h": restaurant.recent_mentions_24h,
            "negative_mentions_24h": restaurant.negative_mentions_24h,
            "is_sponsored": restaurant.is_sponsored,
            "source": restaurant.source,
        }

        try:
            response = requests.post(
                f"{self._settings.ai_base_url.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._settings.ai_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._settings.ai_model,
                    "temperature": 0.1,
                    "messages": [
                        {"role": "system", "content": self._SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                "Analyze this restaurant for fraud/authenticity risks:\n"
                                + json.dumps(payload)
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
            risk_score = float(parsed.get("risk_score", 0.0))
            warnings = [str(w) for w in parsed.get("warnings", [])]
            return FraudAssessment(
                risk_score=min(1.0, max(0.0, risk_score)),
                warnings=warnings,
            )
        except Exception:
            return self._fallback.assess(restaurant)
