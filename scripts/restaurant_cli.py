from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.engine import RecommendationEngine
from app.core.models import SearchQuery
from app.services.fraud_detector import HeuristicFraudDetector
from app.services.local_ai_reasoner import LocalAIReasoner
from app.services.preference_store import JsonPreferenceStore
from app.services.scoring import ScoringService
from app.sources.local_sample import LocalSampleSource


def ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    if value:
        return value
    return default or ""


def parse_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def build_local_engine() -> RecommendationEngine:
    return RecommendationEngine(
        sources=[LocalSampleSource()],
        preference_store=JsonPreferenceStore(str(ROOT / "data" / "preferences.json")),
        fraud_detector=HeuristicFraudDetector(),
        ai_reasoner=LocalAIReasoner(),
        scoring_service=ScoringService(),
    )


def collect_recommend_inputs() -> dict:
    return {
        "user_id": ask("User ID", "demo-user"),
        "where": ask("Where are you going?", "Seoul"),
        "when": ask("When? ISO format or blank", ""),
        "category": ask("Category", "korean"),
        "price_preference": ask("Price preference", "$$"),
        "travel_plan": ask("Travel plan / context", "casual dinner"),
        "party_size": ask("Party size", "2"),
        "top_n": ask("How many results?", "3"),
    }


def print_recommendations(payload: dict) -> None:
    best = payload.get("best_match")
    recommendations = payload.get("recommendations", [])

    print("\n=== Recommendation Result ===")
    if best:
        print(f"Best match: {best.get('name')} ({best.get('category')})")
        print(f"Score: {best.get('score')} | Fraud risk: {best.get('fraud_risk')}")
        print(f"Reason: {best.get('reason')}")
        warnings = best.get("warnings") or []
        if warnings:
            print(f"Warnings: {'; '.join(warnings)}")
    else:
        print("No recommendation found.")

    print("\nRanked options:")
    for index, item in enumerate(recommendations, start=1):
        warnings = item.get("warnings") or []
        warning_text = " | ".join(warnings) if warnings else "none"
        print(
            f"{index}. {item.get('name')} | score={item.get('score')} | "
            f"risk={item.get('fraud_risk')} | warnings={warning_text}"
        )


def local_recommend() -> None:
    values = collect_recommend_inputs()
    query = SearchQuery(
        where=values["where"],
        when=parse_datetime(values["when"]),
        category=values["category"] or None,
        price_preference=values["price_preference"] or None,
        travel_plan=values["travel_plan"] or None,
        party_size=max(1, int(values["party_size"])),
    )
    top_n = max(1, int(values["top_n"]))

    print("\nResolved inputs:")
    print(f"- user_id: {values['user_id']}")
    print(f"- where: {query.where}")
    print(f"- when: {query.when.isoformat() if query.when else 'unparsed / blank'}")
    print(f"- category: {query.category}")
    print(f"- price preference: {query.price_preference}")
    print(f"- travel plan: {query.travel_plan}")
    print(f"- party size: {query.party_size}")
    print(f"- top_n: {top_n}")

    engine = build_local_engine()
    recs = engine.recommend(user_id=values["user_id"], query=query, top_n=top_n)
    payload = {
        "best_match": _to_output(recs[0]) if recs else None,
        "recommendations": [_to_output(rec) for rec in recs],
    }
    print_recommendations(payload)


def local_feedback() -> None:
    user_id = ask("User ID", "demo-user")
    accepted_raw = ask("Accepted recommendation? (y/n)", "y").lower()
    category = ask("Category", "korean")
    price_level = ask("Price level", "$$")

    engine = build_local_engine()
    engine.record_feedback(
        user_id=user_id,
        accepted=accepted_raw in {"y", "yes", "true", "1"},
        category=category or None,
        price_level=price_level or None,
    )
    print("\nFeedback saved to local preference store.")


def api_recommend() -> None:
    base_url = ask("API base URL", "http://127.0.0.1:8000")
    values = collect_recommend_inputs()
    payload = {
        "user_id": values["user_id"],
        "where": values["where"],
        "when": values["when"] or None,
        "category": values["category"] or None,
        "price_preference": values["price_preference"] or None,
        "travel_plan": values["travel_plan"] or None,
        "party_size": max(1, int(values["party_size"])),
        "top_n": max(1, int(values["top_n"])),
    }

    response = requests.post(f"{base_url.rstrip('/')}/recommend", json=payload, timeout=15)
    response.raise_for_status()
    print_recommendations(response.json())


def api_feedback() -> None:
    base_url = ask("API base URL", "http://127.0.0.1:8000")
    user_id = ask("User ID", "demo-user")
    accepted_raw = ask("Accepted recommendation? (y/n)", "y").lower()
    category = ask("Category", "korean")
    price_level = ask("Price level", "$$")
    payload = {
        "user_id": user_id,
        "accepted": accepted_raw in {"y", "yes", "true", "1"},
        "category": category or None,
        "price_level": price_level or None,
    }

    response = requests.post(f"{base_url.rstrip('/')}/feedback", json=payload, timeout=15)
    response.raise_for_status()
    print("\nAPI feedback response:", response.json())


def _to_output(rec):
    return {
        "restaurant_id": rec.restaurant.restaurant_id,
        "name": rec.restaurant.name,
        "category": rec.restaurant.category,
        "score": round(rec.final_score, 4),
        "fraud_risk": round(rec.fraud_risk_score, 4),
        "warnings": rec.warnings,
        "reason": rec.reason,
    }


def main() -> int:
    print("Restaurant Agent CLI")
    print("1) Local recommend (mock AI backend)")
    print("2) Local feedback")
    print("3) API recommend")
    print("4) API feedback")
    choice = ask("Select mode", "1")

    try:
        if choice == "1":
            local_recommend()
        elif choice == "2":
            local_feedback()
        elif choice == "3":
            api_recommend()
        elif choice == "4":
            api_feedback()
        else:
            print("Unknown option.")
            return 1
        return 0
    except Exception as exc:
        print(f"\nError: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())