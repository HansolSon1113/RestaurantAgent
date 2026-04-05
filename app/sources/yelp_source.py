from __future__ import annotations

import requests

from app.config import Settings
from app.core.interfaces import RestaurantDataSource
from app.core.models import Restaurant, SearchQuery


class YelpSource(RestaurantDataSource):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def source_name(self) -> str:
        return "yelp"

    def search(self, query: SearchQuery) -> list[Restaurant]:
        if not self._settings.yelp_api_key:
            return []

        params = {
            "location": query.where,
            "categories": query.category or "restaurants",
            "sort_by": "best_match",
            "limit": 20,
        }
        response = requests.get(
            "https://api.yelp.com/v3/businesses/search",
            headers={"Authorization": f"Bearer {self._settings.yelp_api_key}"},
            params=params,
            timeout=self._settings.request_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()

        restaurants: list[Restaurant] = []
        for biz in payload.get("businesses", []):
            restaurants.append(
                Restaurant(
                    restaurant_id=f"yelp-{biz.get('id')}",
                    name=biz.get("name", "unknown"),
                    city=biz.get("location", {}).get("city", query.where),
                    category=(biz.get("categories") or [{"alias": "unknown"}])[0].get("alias", "unknown"),
                    rating=float(biz.get("rating", 0.0)),
                    review_count=int(biz.get("review_count", 0)),
                    price_level=biz.get("price", "$$"),
                    distance_km=float(biz.get("distance", 0.0)) / 1000.0,
                    source=self.source_name,
                    url=biz.get("url"),
                    recent_mentions_24h=0,
                    negative_mentions_24h=0,
                )
            )
        return restaurants
