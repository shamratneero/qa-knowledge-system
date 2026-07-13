#!/usr/bin/env bash
# Stages a clean, dereferenced copy of the embedding model for offline
# bundling into the desktop app (see desktop_launcher.py / desktop_app.spec).
#
# Hugging Face's local cache stores model files as symlinks into a
# content-addressed blob store, which PyInstaller can mishandle -- this
# copies the real files instead, matching EMBEDDING_MODEL_NAME's default
# (sentence-transformers/all-MiniLM-L6-v2).
#
# Requires the model to already be present in the local HF cache (i.e. the
# app has been run at least once locally with network access beforehand).
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_CACHE_DIR="${HOME}/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2"
DEST_DIR="${PROJECT_ROOT}/packaging/bundled_model/all-MiniLM-L6-v2"

SNAPSHOT_DIR="$(find "$MODEL_CACHE_DIR/snapshots" -mindepth 1 -maxdepth 1 -type d | head -1)"
if [[ -z "$SNAPSHOT_DIR" ]]; then
  echo "Error: no cached snapshot found under $MODEL_CACHE_DIR/snapshots" >&2
  echo "Run the app locally once (with network access) so it downloads the model, then retry." >&2
  exit 1
fi

mkdir -p "$DEST_DIR"
rsync -aL --delete "$SNAPSHOT_DIR/" "$DEST_DIR/"

echo "Staged model at: $DEST_DIR"
du -sh "$DEST_DIR"
