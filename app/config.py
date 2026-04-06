from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv


load_dotenv()

_OPENAI_DEFAULT_URL = "https://api.openai.com/v1"


@dataclass(frozen=True)
class Settings:
    ai_api_key: str = os.getenv("AI_API_KEY", "")
    # No hardcoded default: empty string means "not explicitly configured".
    # Use effective_ai_base_url when you need a URL to call.
    ai_base_url: str = os.getenv("AI_BASE_URL", "")
    ai_model: str = os.getenv("AI_MODEL", "gpt-4.1-mini")

    yelp_api_key: str = os.getenv("YELP_API_KEY", "")
    google_places_api_key: str = os.getenv("GOOGLE_PLACES_API_KEY", "")
    foursquare_api_key: str = os.getenv("FOURSQUARE_API_KEY", "")

    preferences_file: str = os.getenv("PREFERENCES_FILE", "data/preferences.json")
    request_timeout_seconds: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "10"))

    @property
    def ai_enabled(self) -> bool:
        """True when an AI API key or a custom local AI base URL is configured."""
        return bool(self.ai_api_key or self.ai_base_url)

    @property
    def effective_ai_base_url(self) -> str:
        """The base URL to use for AI API calls.

        Falls back to the public OpenAI endpoint only when no explicit URL has
        been configured, which is only reached when ``ai_api_key`` is set (and
        therefore the OpenAI default is appropriate).
        """
        return self.ai_base_url or _OPENAI_DEFAULT_URL


settings = Settings()
