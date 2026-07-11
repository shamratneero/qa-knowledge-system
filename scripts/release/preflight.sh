#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -d ".venv" ]]; then
  echo "Error: .venv not found. Create with: python3 -m venv .venv"
  exit 1
fi

source .venv/bin/activate

echo "[1/6] Running fast tests..."
pytest -m "not slow" -q

echo "[2/6] Running benchmark..."
python scripts/benchmark.py >/tmp/qa_preflight_benchmark.out

echo "[3/6] Checking benchmark artifact..."
if [[ ! -f artifacts/benchmark_results.json ]]; then
  echo "Error: artifacts/benchmark_results.json not found"
  exit 1
fi

echo "[4/6] Validating docker-compose config..."
docker compose config >/tmp/qa_preflight_compose.out

echo "[5/6] Checking release workflow files..."
[[ -f .github/workflows/ci.yml ]] || { echo "Missing CI workflow"; exit 1; }
[[ -f .github/workflows/release.yml ]] || { echo "Missing release workflow"; exit 1; }

echo "[6/6] Verifying key docs..."
[[ -f README.md ]] || { echo "Missing README.md"; exit 1; }
[[ -f CHANGELOG.md ]] || { echo "Missing CHANGELOG.md"; exit 1; }

echo "Preflight checks passed. Ready to tag/release."
