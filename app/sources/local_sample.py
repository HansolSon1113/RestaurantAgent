from __future__ import annotations

from app.core.interfaces import RestaurantDataSource
from app.core.models import Restaurant, SearchQuery


class LocalSampleSource(RestaurantDataSource):
    @property
    def source_name(self) -> str:
        return "local-sample"

    def search(self, query: SearchQuery) -> list[Restaurant]:
        city = query.where.strip().lower()
        catalog = {
            "seoul": [
                Restaurant(
                    restaurant_id="local-seoul-1",
                    name="Han River Grill",
                    city="Seoul",
                    category="korean",
                    rating=4.7,
                    review_count=612,
                    price_level="$$",
                    distance_km=2.1,
                    source=self.source_name,
                    recent_mentions_24h=80,
                    negative_mentions_24h=8,
                ),
                Restaurant(
                    restaurant_id="local-seoul-2",
                    name="Maple Pasta Atelier",
                    city="Seoul",
                    category="italian",
                    rating=4.8,
                    review_count=42,
                    price_level="$$$",
                    distance_km=3.9,
                    source=self.source_name,
                    is_sponsored=True,
                    recent_mentions_24h=260,
                    negative_mentions_24h=140,
                ),
            ],
            "tokyo": [
                Restaurant(
                    restaurant_id="local-tokyo-1",
                    name="Sakura Sushi House",
                    city="Tokyo",
                    category="japanese",
                    rating=4.9,
                    review_count=314,
                    price_level="$$$",
                    distance_km=1.2,
                    source=self.source_name,
                    recent_mentions_24h=220,
                    negative_mentions_24h=30,
                ),
                Restaurant(
                    restaurant_id="local-tokyo-2",
                    name="Asakusa Curry Lab",
                    city="Tokyo",
                    category="japanese",
                    rating=4.6,
                    review_count=511,
                    price_level="$$",
                    distance_km=4.6,
                    source=self.source_name,
                    recent_mentions_24h=70,
                    negative_mentions_24h=5,
                ),
            ],
        }
        return catalog.get(city, [])
