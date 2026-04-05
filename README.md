# Restaurant Agent

Object-oriented Python AI agent that automatically searches restaurants from broad data sources, ranks by user fit, learns from accept/deny feedback, and warns about suspicious or viral-risk candidates.

## Features

- Multi-source retrieval adapters:
  - Yelp API
  - Google Places API
  - Foursquare Places API
  - Local sample source (fallback and dev mode)
- OOP architecture with explicit interfaces for source, preference store, fraud detector, and AI reasoner
- Preference learning persisted in `data/preferences.json`
- Fraud/viral-risk detection with warning signals for low-trust listings
- External AI API integration (OpenAI-compatible `/v1/chat/completions`)
- FastAPI service with deploy-ready Docker files

## Project Structure

- `app/core/models.py`: Domain models
- `app/core/interfaces.py`: Abstract interfaces for extensibility
- `app/core/engine.py`: Main orchestration engine
- `app/sources/*`: Data source adapters
- `app/services/*`: Scoring, fraud detection, preference persistence, AI explanations
- `app/main.py`: API entrypoint

## Quick Start (Local)

1. Create and activate virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy environment template and fill keys:

   ```bash
   cp .env.example .env
   ```

4. Run API server:

   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

5. Open docs:

   - http://localhost:8000/docs

## Quick Start (Docker)

```bash
docker compose up --build
```

## Interactive Test Runner

Run tests with an interactive CLI that asks what to execute and prints a live summary:

```bash
python scripts/run_tests.py
```

Options include running all tests, only the mock AI tests, only the API mock test, or a custom pytest pattern.

## Interactive Recommendation Demo

Run an interactive prompt that collects user input, executes the recommender, and prints the result using a local AI backend:

```bash
python scripts/run_demo.py
```

This mode uses mock/local data sources and a local AI reasoner, so it behaves like the AI backend without requiring external API calls.

## Unified CLI Interface

Use the unified CLI for local mode or live API mode:

```bash
python scripts/restaurant_cli.py
```

Menu options:

- Local recommend (mock AI backend)
- Local feedback
- API recommend
- API feedback

## API

### POST `/recommend`

Example body:

```json
{
  "user_id": "user-123",
  "where": "Seoul",
  "when": "2026-04-05T19:00:00",
  "category": "korean",
  "price_preference": "$$",
  "travel_plan": "Gangnam walk",
  "party_size": 2,
  "top_n": 5
}
```

Response includes:

- `best_match`
- `recommendations[]`
- per item: score, fraud risk, warnings, and AI reason

### POST `/feedback`

Example body:

```json
{
  "user_id": "user-123",
  "accepted": true,
  "category": "korean",
  "price_level": "$$"
}
```

This updates the user profile so later recommendations are personalized.

## Notes

- If external data-source API keys are not set, local sample data is used.
- If `AI_API_KEY` is not set or AI API fails, heuristic explanations are used.
- Fraud/viral detection is heuristic by default; replace with ML classifier if desired.
