# AI Conversation Intelligence Platform

Production-ready FastAPI platform for conversation ingestion, semantic intelligence, analytics dashboards, historical knowledge growth, and retrieval-first AI assistance.

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
  core/
    config.py
    loader.py
    logging.py
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
    cluster_labeling.py
    conversation_analytics.py
    conversation_classification.py
    conversation_clustering.py
    conversation_ingestion.py
    conversation_insights.py
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
docker compose up --build
```

- API port: `8000`
- SQLite persistence via Docker volume mounted at `/app/data`

## Configuration

Environment-configurable values (see `app/core/config.py`):

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

