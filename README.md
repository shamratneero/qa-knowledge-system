# Q&A Knowledge System

An Excel-powered question-answering API that combines **keyword**, **fuzzy**, **semantic**, and **hybrid** retrieval to return ranked answers with confidence scores.

Built with FastAPI, sentence-transformers, and RapidFuzz.

## Why This Project

This project is designed to demonstrate practical backend engineering and applied AI retrieval in a single system:

- API design and schema validation with FastAPI + Pydantic
- retrieval quality improvements from exact matching to semantic search
- measurable quality via benchmark metrics and automated tests
- production readiness with Docker and CI

---

## Features

- **Multi-strategy search** — keyword, fuzzy (typo-tolerant), semantic (embedding-based), and hybrid (weighted combination)
- **Confidence scores** — every result includes a 0–1 confidence score
- **Structured knowledge base** — loads Q&A pairs from Excel (`id`, `question`, `answer`, `category`, `keywords`)
- **Auto-generated API docs** — interactive Swagger UI at `/docs`
- **Structured logging** — query, method, score, and latency logged on every request
- **Docker-ready** — run with a single `docker compose up`
- **CI pipeline** — GitHub Actions runs tests on every push

## Quick Demo

1. Start the API:

```bash
bash scripts/run/start_local.sh
```

2. Open interactive docs:

```text
http://127.0.0.1:8000/docs
```

Or open the minimal frontend:

```text
http://127.0.0.1:8000/ui
```

3. Ask a question:

```json
{
  "question": "What is AI?",
  "method": "hybrid",
  "top_n": 3
}
```

Optional smoke test (in another terminal):

```bash
bash scripts/run/smoke_test.sh
```

---

## Architecture

```mermaid
flowchart LR
    Client["Client / Frontend"] --> API["FastAPI\nPOST /ask"]
    API --> Hybrid["Hybrid Search"]
    Hybrid --> Keyword["Keyword Search"]
    Hybrid --> Fuzzy["Fuzzy Search\n(RapidFuzz)"]
    Hybrid --> Semantic["Semantic Search\n(sentence-transformers)"]
    Keyword --> KB["Excel Knowledge Base\ndata/knowledge_base.xlsx"]
    Fuzzy --> KB
    Semantic --> KB
    Semantic --> Cache["Embedding Cache\n.cache/embeddings.pkl"]
    API --> Logger["Python logging"]
```

### Search pipeline

```mermaid
sequenceDiagram
    participant U as User
    participant A as FastAPI
    participant H as Hybrid Engine
    participant K as Keyword
    participant F as Fuzzy
    participant S as Semantic
    participant D as Knowledge Base

    U->>A: POST /ask {question, method}
    A->>H: search_hybrid(query)
    H->>K: score by substring match
    H->>F: score by fuzzy ratio
    H->>S: score by cosine similarity
    K->>D: read rows
    F->>D: read rows
    S->>D: read rows + embeddings
    H-->>A: ranked results + confidence
    A-->>U: {answer, score, method, results}
```

## Benchmark Snapshot

Use the built-in harness to evaluate retrieval quality and response latency:

```bash
python scripts/benchmark.py
```

The script reports:

- Top-1 accuracy
- Top-3 accuracy
- Average latency (ms)
- JSON artifact at `artifacts/benchmark_results.json`

---

## Project Structure

```
Qs/
├── app/
│   ├── main.py              # FastAPI entrypoint
│   ├── api/                 # (reserved)
│   ├── core/
│   │   ├── config.py        # Settings via environment variables
│   │   ├── loader.py        # Excel knowledge base loader
│   │   └── logging.py       # Logging configuration
│   ├── models/
│   │   └── schemas.py       # Pydantic request/response models
│   ├── search/
│   │   ├── engine.py        # Keyword search
│   │   ├── fuzzy.py         # Fuzzy search
│   │   ├── semantic.py      # Semantic (embedding) search
│   │   └── hybrid.py        # Hybrid combiner
│   └── services/
│       └── database.py      # SQLite migration (optional)
├── data/
│   └── knowledge_base.xlsx  # Default knowledge base
├── tests/
│   ├── test_loader.py
│   ├── test_search.py
│   └── test_api.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .github/workflows/ci.yml
```

---

## Installation

### Prerequisites

- Python 3.11+
- pip

### Local setup

```bash
git clone https://github.com/shamratneero/qa-knowledge-system.git
cd qa-knowledge-system

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Run the API

```bash
uvicorn app.main:app --reload
```

Open **http://127.0.0.1:8000/docs** for Swagger UI.

### Docker

```bash
docker compose up --build
```

API available at **http://localhost:8000**.

Verify container health:

```bash
curl http://localhost:8000/health
```

## Deployment Guide

### Option A: Render (recommended for first public demo)

1. Push this repository to GitHub.
2. In Render, create a new Web Service from the repo (or use `render.yaml`).
3. Configure:
  - Runtime: `Docker`
  - Branch: your deployment branch
  - Port: `8000`
4. Deploy and verify:

```bash
curl https://<your-render-service>/health
```

### Option B: Railway

1. Create a new Railway project from this GitHub repo.
2. Use Docker-based deployment (auto-detected from `Dockerfile` or `railway.toml`).
3. Set environment variables if needed (`LOG_LEVEL`, etc.).
4. Verify:

```bash
curl https://<your-railway-service>/health
```

Deployment manifests included:

- `render.yaml`
- `railway.toml`

### Local Docker note

If `docker build` fails with daemon connection errors, start Docker Desktop first and retry:

```bash
docker build -t qa-knowledge-system:local .
docker compose up --build
```

## Release Checklist

Before creating a release tag:

1. Run tests

```bash
pytest -m "not slow" -v
```

2. Run benchmark and verify artifact

```bash
python scripts/benchmark.py
cat artifacts/benchmark_results.json
```

3. Create and push a version tag

```bash
git tag v1.0.0
git push origin v1.0.0
```

Tag pushes trigger the release workflow in `.github/workflows/release.yml`.

Optional helper script:

```bash
bash scripts/release/prepare_release.sh v1.0.1
```

This script runs fast tests, runs benchmark, checks for clean git state, and creates an annotated local tag.

Project release notes are tracked in `CHANGELOG.md`.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | API info |
| `GET` | `/ui` | Minimal browser UI |
| `GET` | `/health` | Health check |
| `POST` | `/ask` | Search the knowledge base |

### `POST /ask`

**Request body:**

```json
{
  "question": "What is machine learning?",
  "method": "hybrid",
  "top_n": 5
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `question` | string | required | User question (1–500 chars) |
| `method` | string | `"hybrid"` | `hybrid`, `keyword`, `fuzzy`, or `semantic` |
| `top_n` | int | `5` | Number of results (1–50) |

**Success response (200):**

```json
{
  "found": true,
  "query": "What is machine learning?",
  "method": "hybrid",
  "answer": "Machine Learning is a subset of AI that learns patterns from data.",
  "score": 0.87,
  "results": [
    {
      "id": 2,
      "question": "What is Machine Learning?",
      "answer": "Machine Learning is a subset of AI that learns patterns from data.",
      "category": "ML",
      "keywords": "Machine Learning,ML",
      "score": 0.87,
      "confidence": 0.87
    }
  ]
}
```

**Error responses:**

| Status | When |
|--------|------|
| `404` | No matching answer found |
| `422` | Invalid request body (empty question, bad method) |
| `500` | Internal search error |

### Example requests

```bash
# Health check
curl http://127.0.0.1:8000/health

# Keyword search
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is AI?", "method": "keyword"}'

# Hybrid search (default)
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Tell me about artificial intelligence", "method": "hybrid"}'

# Fuzzy search (typo-tolerant)
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Machne Learning?", "method": "fuzzy"}'
```

### Swagger UI

Start the server and visit **http://127.0.0.1:8000/docs**:

Use the live `/docs` page for interactive testing.

---

## Knowledge Base Format

Place an Excel file at `data/knowledge_base.xlsx` with these columns:

| Column | Required | Description |
|--------|----------|-------------|
| `id` | yes | Unique identifier |
| `question` | yes | The question text |
| `answer` | yes | The answer text |
| `category` | yes | Category label |
| `keywords` | yes | Comma-separated keywords |

---

## Testing

```bash
# Fast tests (keyword, fuzzy, API, loader)
pytest -m "not slow" -v

# All tests including semantic/hybrid (downloads ML model on first run)
pytest -v

# Run benchmark harness
python scripts/benchmark.py
```

Expected benchmark output includes:
- Top-1 accuracy
- Top-3 accuracy
- Average latency (ms)

---

## Configuration

Set via environment variables or a `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `HOST` | `127.0.0.1` | Server host |
| `PORT` | `8000` | Server port |
| `DEBUG` | `false` | Debug mode |

---

## Future Improvements

- [ ] **Benchmark suite** — 100-question eval set with precision, recall, top-1/top-3 accuracy
- [ ] **RAG upgrade** — retrieve top-5 results, pass to LLM for natural language answers
- [ ] **Vector index** — replace brute-force cosine similarity with FAISS for larger knowledge bases
- [ ] **Frontend** — simple React/Next.js question-answer UI
- [ ] **Live deployment** — deploy to Railway, Render, or Azure
- [ ] **SQLite integration** — wire database layer into search for dynamic KB updates

## Recruiter Highlights

- clear API contracts (`/ask` with validated request/response models)
- multiple retrieval strategies with confidence scoring
- automated test coverage for loader, search, API, and edge cases
- CI workflow and containerized runtime for reproducibility

---

## License

MIT
