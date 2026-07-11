#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -d ".venv" ]]; then
  echo "Error: .venv not found. Create it with: python3 -m venv .venv"
  exit 1
fi

source .venv/bin/activate

echo "Starting API on http://127.0.0.1:8000"
exec uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
