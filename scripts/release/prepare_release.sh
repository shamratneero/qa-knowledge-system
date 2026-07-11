#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <version>"
  echo "Example: $0 v1.0.1"
  exit 1
fi

VERSION="$1"

if [[ ! "$VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Error: version must match vMAJOR.MINOR.PATCH (e.g. v1.0.1)"
  exit 1
fi

echo "[1/5] Running fast tests..."
pytest -m "not slow" -q

echo "[2/5] Running benchmark..."
python scripts/benchmark.py > /tmp/qa_release_benchmark.out

echo "[3/5] Ensuring git workspace is clean..."
dirty="$(git status --porcelain | grep -v 'artifacts/benchmark_results.json' || true)"
if [[ -n "$dirty" ]]; then
  echo "Error: workspace is not clean. Commit or stash changes first."
  git status --short
  exit 1
fi

echo "[4/5] Creating annotated tag $VERSION"
git tag -a "$VERSION" -m "Release $VERSION"

echo "[5/5] Release tag created locally."
echo "To publish: git push origin $VERSION"
