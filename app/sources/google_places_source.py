from __future__ import annotations

import requests

from app.config import Settings
from app.core.interfaces import RestaurantDataSource
from app.core.models import Restaurant, SearchQuery


class GooglePlacesSource(RestaurantDataSource):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def source_name(self) -> str:
        return "google-places"

    def search(self, query: SearchQuery) -> list[Restaurant]:
        if not self._settings.google_places_api_key:
            return []

        response = requests.get(
            "https://maps.googleapis.com/maps/api/place/textsearch/json",
            params={
                "query": f"{query.category or 'restaurant'} in {query.where}",
                "key": self._settings.google_places_api_key,
            },
            timeout=self._settings.request_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()

        restaurants: list[Restaurant] = []
        for place in payload.get("results", [])[:20]:
            restaurants.append(
                Restaurant(
                    restaurant_id=f"gplaces-{place.get('place_id')}",
                    name=place.get("name", "unknown"),
                    city=query.where,
                    category=(query.category or "restaurant"),
                    rating=float(place.get("rating", 0.0)),
                    review_count=int(place.get("user_ratings_total", 0)),
                    price_level="$$",
                    distance_km=5.0,
                    source=self.source_name,
                    recent_mentions_24h=0,
                    negative_mentions_24h=0,
                )
            )
        return restaurants
