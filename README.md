---
title: Conversation Intelligence
emoji: 💬
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
---

# AI Conversation Intelligence Platform

Production-ready FastAPI platform for conversation ingestion, semantic intelligence, analytics dashboards, historical knowledge growth, and retrieval-first AI assistance.

> **Deployed on Hugging Face Spaces?** The free CPU tier's storage is ephemeral — the SQLite database (including any registered account and uploaded data) resets whenever the Space restarts, sleeps from inactivity, or rebuilds. Fine for trying it out; for anything you want to keep, either enable HF's paid Persistent Storage add-on for this Space, or self-host instead (Docker/Oracle Cloud Free Tier — see Docker section below).

## Project Overview

This platform ingests customer-support Excel exports, reconstructs conversations, classifies and clusters issues, builds a persistent knowledge base with embeddings, and exposes operational dashboards and APIs for analytics teams.

It is designed for deterministic analytics and retrieval integrity: no fine-tuning, no hidden model training, no black-box state.

## Core Features

- Excel ingestion and validation (Tickets + Conversations sheets)
- Conversation reconstruction by Ticket ID
- Semantic duplicate/similar/unique classification
- Deterministic cluster naming
- Business insights engine (recurring issues, automation opportunities, emerging issues)
- Persistent historical knowledge base across uploads
- Knowledge search and nearest-conversation retrieval
- Retrieval-first AI analyst assistant (`POST /ai/query`) with source citations
- Modern SaaS-style web dashboard with Chart.js visualizations
- Export endpoints (CSV/Excel)
- Structured JSON logging
- Dockerized runtime with persistent SQLite volume
- CI pipeline with lint, formatting, and test gates

## Architecture

### Processing Pipeline

1. Upload Excel
2. Validate sheets and columns
3. Reconstruct conversation threads
4. Classify and cluster conversations
5. Generate deterministic cluster labels
6. Persist active analytics dataset
7. Append to historical knowledge base
8. Serve analytics, insights, search, and AI retrieval APIs

### Runtime Components

- API layer: `app/main.py`
- Services layer: ingestion, analytics, insights, knowledge, AI retrieval
- Persistence: SQLite (`data/knowledge.db`)
- Frontend: static dashboard (`app/static/index.html`)

## Installation

### Requirements

- Python 3.11+
- pip

### Local Setup

```bash
git clone <repo-url>
cd Qs
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run Locally

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open:

- Swagger: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- UI: `http://127.0.0.1:8000/ui`

## Authentication

Every API route except the ones below requires a valid `Authorization: Bearer <token>` header. This is a **single-admin-account** system: the first `POST /auth/register` call creates the one account and registration then closes itself (subsequent attempts return `403`).

Public (no token required): `GET /`, `GET /health`, `GET /ui`, `GET /docs`, `GET /redoc`, `GET /openapi.json`, `POST /auth/register`, `POST /auth/login`, `GET /auth/status`.

- `GET /auth/status` — `{"registered": bool}`, tells you whether to register or log in
- `POST /auth/register` — `{"username": "...", "password": "..."}` (password ≥ 8 chars); only works once
- `POST /auth/login` — same body shape; returns `{"access_token": "...", "token_type": "bearer"}`
- `GET /auth/me` — returns the current user (requires the token)

The `/ui` dashboard handles this flow automatically (shows a registration form on first run, a sign-in form afterward). For direct API access, pass the returned token as `Authorization: Bearer <access_token>` on every subsequent request.

Tokens are JWTs signed with `SECRET_KEY` (see Configuration below) and expire after `ACCESS_TOKEN_EXPIRE_MINUTES` (default 30).

## API Documentation

### Health & Root

- `GET /`
- `GET /health`
- `GET /ui`

### Ingestion

- `POST /admin/conversations/upload`

Example request (multipart form):
- field: `file`
- value: `.xlsx` or `.xls` file

### Analytics

- `GET /analytics/overview`
- `GET /analytics/charts`
- `GET /analytics/conversations`
- `GET /analytics/conversations/{conversation_id}`
- `GET /analytics/insights`
- `GET /analytics/export/csv`
- `GET /analytics/export/excel`

### Knowledge Base

- `GET /knowledge/dashboard`
- `GET /knowledge/search`
- `GET /knowledge/similar/{conversation_id}`

### AI Analyst Assistant

- `POST /ai/query`

Example body:

```json
{
  "question": "What are customers complaining about most?",
  "top_k": 10
}
```

Response includes:

- `answer`
- `sources`
- `matching_conversations`
- `matching_clusters`
- `confidence_score`
- `generation_mode`

## Folder Structure

```text
app/
  main.py
  api/
    auth.py
  core/
    config.py
    loader.py
    logging.py
    security.py
  models/
    database.py
    schemas.py
  search/
    engine.py
    fuzzy.py
    semantic.py
    hybrid.py
  services/
    ai_assistant.py
    auth.py
    cluster_labeling.py
    conversation_analytics.py
    conversation_classification.py
    conversation_clustering.py
    conversation_ingestion.py
    conversation_insights.py
    conversation_summary.py
    database.py
    knowledge_base.py
  static/
    index.html
data/
  knowledge.db
  knowledge_base.xlsx
tests/
Dockerfile
docker-compose.yml
requirements.txt
```

## Database Schema (High-Level)

- `knowledge`: base Q&A records
- `conversation_threads`: active analytics snapshot
- `knowledge_conversations`: historical cumulative knowledge records

Stored attributes include ticket metadata, reconstructed text, embedding vector JSON, cluster/classification fields, upload batch, and timestamps.

## Docker

### Build and Run

```bash
SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))") docker compose up --build
```

- API port: `8000`
- SQLite persistence via Docker volume mounted at `/app/data` (a fresh, empty database is created automatically on first run — no local dev/test data is baked into the image; see `.dockerignore`)
- **Set `SECRET_KEY` explicitly for any real deployment.** If it's not provided, the app generates a random one at process startup and logs a warning — every login session is invalidated whenever the process restarts. Generate one once and store it in your deployment platform's secrets (not in `docker-compose.yml`, which is committed to git).

## Configuration

Environment-configurable values (see `app/core/config.py`):

- `SECRET_KEY`, `ALGORITHM` (default `HS256`), `ACCESS_TOKEN_EXPIRE_MINUTES` (default `30`) — JWT signing for authentication
- docs URLs (`DOCS_URL`, `REDOC_URL`, `OPENAPI_URL`)
- upload size limits
- pagination limits
- chart top-N
- embedding model name
- knowledge thresholds
- log level and JSON logging

## Screenshots (Placeholders)

- `docs/screenshots/dashboard-home.png`
- `docs/screenshots/analytics-charts.png`
- `docs/screenshots/knowledge-search.png`
- `docs/screenshots/ask-ai-panel.png`

## Testing

Run all tests:

```bash
pytest -v
```

Fast tests only:

```bash
pytest -m "not slow" -v
```

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs on push/PR:

- dependency install
- formatting checks
- lint checks
- unit/integration tests

## Future Roadmap

- Role-based access and tenant isolation
- Streaming exports for very large datasets
- Incremental background indexing workers
- Time-series anomaly detection for emerging issues
- Admin audit reports and SLO dashboards

