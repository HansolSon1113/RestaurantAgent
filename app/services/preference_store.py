from __future__ import annotations

import json
from pathlib import Path

from app.core.interfaces import PreferenceStore
from app.core.models import UserPreferenceProfile


class JsonPreferenceStore(PreferenceStore):
    def __init__(self, file_path: str) -> None:
        self._path = Path(file_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.write_text("{}", encoding="utf-8")

    def _load_all(self) -> dict:
        return json.loads(self._path.read_text(encoding="utf-8"))

    def _save_all(self, data: dict) -> None:
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def get_profile(self, user_id: str) -> UserPreferenceProfile:
        data = self._load_all()
        raw = data.get(user_id, {})
        return UserPreferenceProfile(
            user_id=user_id,
            category_affinity=raw.get("category_affinity", {}),
            category_avoidance=raw.get("category_avoidance", {}),
            price_affinity=raw.get("price_affinity", {}),
        )

    def record_feedback(
        self,
        user_id: str,
        accepted: bool,
        category: str | None,
        price_level: str | None,
    ) -> UserPreferenceProfile:
        data = self._load_all()
        user = data.setdefault(
            user_id,
            {
                "category_affinity": {},
                "category_avoidance": {},
                "price_affinity": {},
            },
        )

        if category:
            target = "category_affinity" if accepted else "category_avoidance"
            user[target][category] = user[target].get(category, 0) + 1

        if accepted and price_level:
            user["price_affinity"][price_level] = user["price_affinity"].get(price_level, 0) + 1

        self._save_all(data)
        return self.get_profile(user_id)
