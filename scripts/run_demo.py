from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

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


def print_recommendations(payload: dict) -> None:
    best = payload["best_match"]
    recommendations = payload["recommendations"]

    print("\n=== Recommendation Result ===")
    if best:
        print(f"Best match: {best['name']} ({best['category']})")
        print(f"Score: {best['score']} | Fraud risk: {best['fraud_risk']}")
        print(f"Reason: {best['reason']}")
        if best["warnings"]:
            print(f"Warnings: {'; '.join(best['warnings'])}")
    else:
        print("No recommendation found.")

    print("\nRanked options:")
    for index, item in enumerate(recommendations, start=1):
        warning_text = " | ".join(item["warnings"]) if item["warnings"] else "none"
        print(
            f"{index}. {item['name']} | score={item['score']} | "
            f"risk={item['fraud_risk']} | warnings={warning_text}"
        )


def main() -> int:
    print("Restaurant Recommendation Demo")
    print("This run uses a local AI backend, so no external API is needed.")

    user_id = ask("User ID", "demo-user")
    where = ask("Where are you going?", "Seoul")
    when_text = ask("When? ISO format or blank", "")
    category = ask("Category", "korean")
    price_preference = ask("Price preference", "$$")
    travel_plan = ask("Travel plan / context", "casual dinner")
    party_size_text = ask("Party size", "2")
    top_n_text = ask("How many results?", "3")

    query = SearchQuery(
        where=where,
        when=parse_datetime(when_text),
        category=category or None,
        price_preference=price_preference or None,
        travel_plan=travel_plan or None,
        party_size=max(1, int(party_size_text)),
    )

    print("\nResolved inputs:")
    print(f"- user_id: {user_id}")
    print(f"- where: {query.where}")
    print(f"- when: {query.when.isoformat() if query.when else 'unparsed / blank'}")
    print(f"- category: {query.category}")
    print(f"- price preference: {query.price_preference}")
    print(f"- travel plan: {query.travel_plan}")
    print(f"- party size: {query.party_size}")
    print(f"- top_n: {max(1, int(top_n_text))}")

    engine = RecommendationEngine(
        sources=[LocalSampleSource()],
        preference_store=JsonPreferenceStore(str(ROOT / "data" / "preferences.json")),
        fraud_detector=HeuristicFraudDetector(),
        ai_reasoner=LocalAIReasoner(),
        scoring_service=ScoringService(),
    )

    recs = engine.recommend(user_id=user_id, query=query, top_n=max(1, int(top_n_text)))
    payload = {
        "best_match": None if not recs else _to_output(recs[0]),
        "recommendations": [_to_output(rec) for rec in recs],
    }
    print_recommendations(payload)
    return 0


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


if __name__ == "__main__":
    raise SystemExit(main())
