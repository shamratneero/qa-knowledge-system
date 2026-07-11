#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:8000}"

echo "Smoke test target: $BASE_URL"

echo "[1/4] Health"
curl -sS "$BASE_URL/health" | cat

echo "\n[2/4] Root"
curl -sS "$BASE_URL/" | cat

echo "\n[3/4] Ask (hybrid)"
curl -sS -X POST "$BASE_URL/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is AI?","method":"hybrid","top_n":3}' | cat

echo "\n[4/4] UI check"
status_code="$(curl -sS -o /tmp/qa_ui_body.txt -w "%{http_code}" "$BASE_URL/ui")"
echo "UI status: $status_code"

echo "\nSmoke test complete."
