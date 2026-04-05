from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    ai_api_key: str = os.getenv("AI_API_KEY", "")
    ai_base_url: str = os.getenv("AI_BASE_URL", "https://api.openai.com/v1")
    ai_model: str = os.getenv("AI_MODEL", "gpt-4.1-mini")

    yelp_api_key: str = os.getenv("YELP_API_KEY", "")
    google_places_api_key: str = os.getenv("GOOGLE_PLACES_API_KEY", "")
    foursquare_api_key: str = os.getenv("FOURSQUARE_API_KEY", "")

    preferences_file: str = os.getenv("PREFERENCES_FILE", "data/preferences.json")
    request_timeout_seconds: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "10"))


settings = Settings()
