"""RestaurantAgent — a true AI agent that uses LLM-driven tool calling.

Unlike the pipeline-based RecommendationEngine (which always follows the same
hardcoded steps), the RestaurantAgent lets the LLM decide which tools to
invoke, in what order, and how many times.  Each tool call is recorded as an
AgentStep so callers can inspect the agent's full reasoning trace.

Agentic loop
------------
1. System prompt + user request sent to LLM together with tool schemas.
2. LLM responds with one or more ``tool_calls``.
3. Each tool is executed; observations are appended to the message history.
4. Steps 2–3 repeat (up to MAX_ITERATIONS) until the LLM produces a ``stop``
   finish reason and returns a final JSON answer.
5. If AI is not configured the call raises ``RuntimeError`` immediately.
   There is no fallback to a deterministic pipeline.

Available tools
---------------
- search_restaurants   — query all configured data sources
- get_user_preferences — retrieve persisted preference profile
- assess_fraud_risk    — run LLM fraud/viral-risk checks
- score_and_rank       — score candidates and return the top-N list
"""

from __future__ import annotations

import json
from typing import Any

import requests

from app.config import Settings
from app.core.interfaces import FraudDetector, PreferenceStore, RestaurantDataSource
from app.core.models import AgentStep, FraudAssessment, Restaurant, ScoredRecommendation, SearchQuery
from app.services.scoring import FRAUD_RISK_PENALTY, ScoringService
from app.utils import extract_json


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _Tool:
    """Wraps a callable with its OpenAI function-schema metadata."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self._handler = handler

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def execute(self, args: dict[str, Any]) -> Any:
        return self._handler(**args)


# ---------------------------------------------------------------------------
# RestaurantAgent
# ---------------------------------------------------------------------------

class RestaurantAgent:
    """AI agent that drives restaurant recommendations via LLM tool calling.

    The LLM autonomously decides which tools to call and in what order,
    enabling dynamic planning and multi-step reasoning rather than a fixed
    pipeline.  All tool invocations are recorded as ``AgentStep`` objects and
    exposed on the returned ``AgentResult`` for full transparency.
    """

    MAX_ITERATIONS: int = 8

    def __init__(
        self,
        sources: list[RestaurantDataSource],
        preference_store: PreferenceStore,
        fraud_detector: FraudDetector,
        scoring_service: ScoringService,
        settings: Settings,
    ) -> None:
        self._sources = sources
        self._preference_store = preference_store
        self._fraud_detector = fraud_detector
        self._scoring_service = scoring_service
        self._settings = settings

        # Ephemeral cache of restaurants discovered during a single run()
        self._candidate_cache: dict[str, Restaurant] = {}
        # Fraud assessments computed during a single run() — reused by score_and_rank
        # to avoid calling the fraud detector twice for the same restaurant.
        self._fraud_cache: dict[str, FraudAssessment] = {}

        self._tools: list[_Tool] = self._build_tools()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        user_id: str,
        query: SearchQuery,
        top_n: int = 5,
    ) -> "AgentResult":
        """Execute the agentic loop and return recommendations with trace.

        Raises:
            RuntimeError: If AI is not configured, the API call fails, the
                agent exhausts its iteration budget, or the LLM returns output
                that cannot be parsed into valid recommendations.
        """
        if not self._settings.ai_enabled:
            raise RuntimeError(
                "AI is not configured. Set AI_API_KEY or AI_BASE_URL to enable the agent."
            )

        self._candidate_cache = {}
        self._fraud_cache = {}
        steps: list[AgentStep] = []

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are a restaurant recommendation agent. "
                    "Use the available tools to discover, evaluate, and rank restaurants for the user. "
                    "Typical sequence: (1) get_user_preferences, (2) search_restaurants, "
                    "(3) assess_fraud_risk for promising candidates, (4) score_and_rank. "
                    "When you have enough information, respond with ONLY a JSON array "
                    "(no markdown fences) where each element has: "
                    "restaurant_id, name, city, category, rating, price_level, reason, warnings (list)."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "user_id": user_id,
                        "query": {
                            "where": query.where,
                            "when": query.when.isoformat() if query.when else None,
                            "category": query.category,
                            "price_preference": query.price_preference,
                            "travel_plan": query.travel_plan,
                            "party_size": query.party_size,
                        },
                        "top_n": top_n,
                    }
                ),
            },
        ]

        tool_schemas = [t.to_openai_schema() for t in self._tools]

        for iteration in range(self.MAX_ITERATIONS):
            try:
                headers: dict[str, str] = {"Content-Type": "application/json"}
                if self._settings.ai_api_key:
                    headers["Authorization"] = f"Bearer {self._settings.ai_api_key}"
                response = requests.post(
                    f"{self._settings.effective_ai_base_url.rstrip('/')}/chat/completions",
                    headers=headers,
                    json={
                        "model": self._settings.ai_model,
                        "temperature": 0.2,
                        "messages": messages,
                        "tools": tool_schemas,
                        "tool_choice": "auto",
                    },
                    timeout=self._settings.request_timeout_seconds,
                )
                response.raise_for_status()
            except Exception as exc:
                raise RuntimeError(f"AI API call failed: {exc}") from exc

            choice = response.json()["choices"][0]
            message: dict[str, Any] = choice["message"]
            finish_reason: str = choice.get("finish_reason", "")

            messages.append(message)

            if finish_reason == "tool_calls":
                for tool_call in message.get("tool_calls", []):
                    tool_name: str = tool_call["function"]["name"]
                    try:
                        tool_args: dict[str, Any] = json.loads(
                            tool_call["function"]["arguments"]
                        )
                    except (json.JSONDecodeError, KeyError):
                        tool_args = {}

                    tool = next(
                        (t for t in self._tools if t.name == tool_name), None
                    )
                    if tool is not None:
                        result = tool.execute(tool_args)
                    else:
                        result = {"error": f"Unknown tool: {tool_name}"}

                    steps.append(
                        AgentStep(
                            step=iteration,
                            tool=tool_name,
                            inputs=tool_args,
                            result=result,
                        )
                    )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": json.dumps(result),
                        }
                    )

            elif finish_reason == "stop":
                content = (message.get("content") or "").strip()
                recs = self._parse_llm_recommendations(content, user_id, query, top_n)
                return AgentResult(recommendations=recs, steps=steps, used_agent=True)

        # Exhausted iterations
        raise RuntimeError(
            f"Agent exhausted {self.MAX_ITERATIONS} iterations without producing a final answer."
        )

    # ------------------------------------------------------------------
    # Tool handlers (called by the LLM via tool_calls)
    # ------------------------------------------------------------------

    def _tool_search_restaurants(
        self,
        city: str,
        category: str | None = None,
        price_preference: str | None = None,
        party_size: int = 1,
    ) -> list[dict[str, Any]]:
        sub_query = SearchQuery(
            where=city,
            category=category,
            price_preference=price_preference,
            party_size=party_size,
        )
        dedup: dict[str, Restaurant] = {}
        for source in self._sources:
            try:
                for r in source.search(sub_query):
                    key = f"{r.name.strip().lower()}::{r.city.strip().lower()}"
                    if key not in dedup:
                        dedup[key] = r
            except Exception:
                continue

        results = []
        for r in dedup.values():
            self._candidate_cache[r.restaurant_id] = r
            results.append(
                {
                    "restaurant_id": r.restaurant_id,
                    "name": r.name,
                    "city": r.city,
                    "category": r.category,
                    "rating": r.rating,
                    "review_count": r.review_count,
                    "price_level": r.price_level,
                    "distance_km": r.distance_km,
                    "source": r.source,
                    "url": r.url,
                }
            )
        return results

    def _tool_get_user_preferences(self, user_id: str) -> dict[str, Any]:
        profile = self._preference_store.get_profile(user_id)
        return {
            "user_id": profile.user_id,
            "category_affinity": profile.category_affinity,
            "category_avoidance": profile.category_avoidance,
            "price_affinity": profile.price_affinity,
        }

    def _tool_assess_fraud_risk(self, restaurant_id: str) -> dict[str, Any]:
        restaurant = self._candidate_cache.get(restaurant_id)
        if restaurant is None:
            return {"error": f"Restaurant '{restaurant_id}' not found in search results."}
        assessment = self._fraud_detector.assess(restaurant)
        self._fraud_cache[restaurant_id] = assessment
        return {
            "restaurant_id": restaurant_id,
            "risk_score": assessment.risk_score,
            "warnings": assessment.warnings,
        }

    def _tool_score_and_rank(
        self,
        user_id: str,
        restaurant_ids: list[str],
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        profile = self._preference_store.get_profile(user_id)
        scored: list[tuple[float, dict[str, Any]]] = []

        for rid in restaurant_ids:
            restaurant = self._candidate_cache.get(rid)
            if restaurant is None:
                continue
            # Reuse a cached fraud assessment from a prior assess_fraud_risk call
            # so we never hit the fraud detector twice for the same restaurant.
            fraud = self._fraud_cache.get(rid) or self._fraud_detector.assess(restaurant)
            self._fraud_cache[rid] = fraud
            base = self._scoring_service.base_score(restaurant)
            pref = self._scoring_service.preference_score(restaurant, profile)
            final = base + pref - (fraud.risk_score * FRAUD_RISK_PENALTY)
            scored.append(
                (
                    final,
                    {
                        "restaurant_id": rid,
                        "name": restaurant.name,
                        "base_score": round(base, 4),
                        "preference_score": round(pref, 4),
                        "fraud_risk_score": round(fraud.risk_score, 4),
                        "final_score": round(final, 4),
                        "warnings": fraud.warnings,
                    },
                )
            )

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:top_n]]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_tools(self) -> list[_Tool]:
        return [
            _Tool(
                name="search_restaurants",
                description=(
                    "Search for restaurants in a city across all configured data sources "
                    "(Yelp, Google Places, Foursquare, local). Returns a list of candidates."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "City to search in (e.g. 'Seoul', 'Tokyo').",
                        },
                        "category": {
                            "type": "string",
                            "description": "Food category filter (e.g. 'korean', 'italian').",
                        },
                        "price_preference": {
                            "type": "string",
                            "description": "Desired price range (e.g. '$', '$$', '$$$').",
                        },
                        "party_size": {
                            "type": "integer",
                            "description": "Number of diners.",
                        },
                    },
                    "required": ["city"],
                },
                handler=self._tool_search_restaurants,
            ),
            _Tool(
                name="get_user_preferences",
                description=(
                    "Retrieve a user's persisted preference profile, including liked/disliked "
                    "food categories and preferred price levels."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "user_id": {
                            "type": "string",
                            "description": "Unique user identifier.",
                        }
                    },
                    "required": ["user_id"],
                },
                handler=self._tool_get_user_preferences,
            ),
            _Tool(
                name="assess_fraud_risk",
                description=(
                    "Run fraud and viral-risk heuristics on a specific restaurant "
                    "(identified by restaurant_id from a previous search_restaurants call). "
                    "Returns a risk_score (0–1) and a list of warning messages."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "restaurant_id": {
                            "type": "string",
                            "description": "restaurant_id returned by search_restaurants.",
                        }
                    },
                    "required": ["restaurant_id"],
                },
                handler=self._tool_assess_fraud_risk,
            ),
            _Tool(
                name="score_and_rank",
                description=(
                    "Score a list of restaurant candidates using base quality metrics, "
                    "user preference affinity, and fraud-risk penalties. "
                    "Reuses any fraud assessments already obtained via assess_fraud_risk "
                    "to avoid redundant computation. "
                    "Returns the top-N candidates sorted by final score."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "user_id": {
                            "type": "string",
                            "description": "User identifier (for preference-based scoring).",
                        },
                        "restaurant_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of restaurant_ids to score.",
                        },
                        "top_n": {
                            "type": "integer",
                            "description": "Maximum number of ranked results to return.",
                        },
                    },
                    "required": ["user_id", "restaurant_ids"],
                },
                handler=self._tool_score_and_rank,
            ),
        ]

    def _parse_llm_recommendations(
        self,
        content: str,
        user_id: str,
        query: SearchQuery,
        top_n: int,
    ) -> list[ScoredRecommendation]:
        """Parse the LLM's final JSON answer into ScoredRecommendations.

        Raises:
            RuntimeError: If the content cannot be parsed as JSON or yields no
                recognizable restaurant IDs from the current candidate cache.
        """
        try:
            items: list[dict[str, Any]] = json.loads(extract_json(content))
        except Exception as exc:
            raise RuntimeError(f"Agent returned invalid JSON: {exc}") from exc

        recs: list[ScoredRecommendation] = []
        for item in items[:top_n]:
            rid = item.get("restaurant_id", "")
            restaurant = self._candidate_cache.get(rid)
            if restaurant is None:
                continue
            fraud = self._fraud_cache.get(rid)
            if fraud is None:
                fraud = self._fraud_detector.assess(restaurant)
                self._fraud_cache[rid] = fraud
            base = self._scoring_service.base_score(restaurant)
            profile = self._preference_store.get_profile(user_id)
            pref = self._scoring_service.preference_score(restaurant, profile)
            final = base + pref - (fraud.risk_score * FRAUD_RISK_PENALTY)
            rec = ScoredRecommendation(
                restaurant=restaurant,
                base_score=base,
                preference_score=pref,
                fraud_risk_score=fraud.risk_score,
                final_score=final,
                warnings=list(item.get("warnings", fraud.warnings)),
                reason=item.get("reason", ""),
            )
            recs.append(rec)

        if not recs:
            raise RuntimeError(
                "Agent final answer contained no recognizable restaurant IDs."
            )
        return recs


# ---------------------------------------------------------------------------
# AgentResult
# ---------------------------------------------------------------------------

class AgentResult:
    """Wraps agent output together with its full reasoning trace."""

    def __init__(
        self,
        recommendations: list[ScoredRecommendation],
        steps: list[AgentStep],
        used_agent: bool,
    ) -> None:
        self.recommendations = recommendations
        self.steps = steps
        self.used_agent = used_agent
