#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:8000}"

echo "Smoke test target: $BASE_URL"

echo "[1/5] Health"
curl -sS "$BASE_URL/health" | cat

echo "\n[2/5] Root"
curl -sS "$BASE_URL/" | cat

echo "\n[3/5] Ask requires authentication (expect 401 without a token)"
status_code="$(curl -sS -o /tmp/qa_ask_body.txt -w "%{http_code}" -X POST "$BASE_URL/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is AI?","method":"hybrid","top_n":3}')"
if [[ "$status_code" != "401" ]]; then
  echo "Error: expected /ask to return 401 without a token, got $status_code"
  cat /tmp/qa_ask_body.txt
  exit 1
fi
echo "Ask auth boundary OK (401 as expected)"

echo "\n[4/5] Auth status endpoint (public)"
curl -sS "$BASE_URL/auth/status" | cat

echo "\n[5/5] UI check"
status_code="$(curl -sS -o /tmp/qa_ui_body.txt -w "%{http_code}" "$BASE_URL/ui")"
echo "UI status: $status_code"

echo "\nSmoke test complete."
