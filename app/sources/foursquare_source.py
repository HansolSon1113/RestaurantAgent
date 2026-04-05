from __future__ import annotations

import requests

from app.config import Settings
from app.core.interfaces import RestaurantDataSource
from app.core.models import Restaurant, SearchQuery


class FoursquareSource(RestaurantDataSource):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def source_name(self) -> str:
        return "foursquare"

    def search(self, query: SearchQuery) -> list[Restaurant]:
        if not self._settings.foursquare_api_key:
            return []

        response = requests.get(
            "https://api.foursquare.com/v3/places/search",
            headers={"Authorization": self._settings.foursquare_api_key},
            params={
                "query": query.category or "restaurant",
                "near": query.where,
                "limit": 20,
            },
            timeout=self._settings.request_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()

        restaurants: list[Restaurant] = []
        for place in payload.get("results", []):
            categories = place.get("categories") or [{"name": "restaurant"}]
            restaurants.append(
                Restaurant(
                    restaurant_id=f"foursquare-{place.get('fsq_id')}",
                    name=place.get("name", "unknown"),
                    city=query.where,
                    category=categories[0].get("name", "restaurant").lower(),
                    rating=float(place.get("rating", 0.0)),
                    review_count=int(place.get("stats", {}).get("total_ratings", 0)),
                    price_level="$$",
                    distance_km=float(place.get("distance", 0.0)) / 1000.0,
                    source=self.source_name,
                    recent_mentions_24h=0,
                    negative_mentions_24h=0,
                )
            )
        return restaurants
