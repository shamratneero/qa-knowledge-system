#!/usr/bin/env bash
# Wraps dist/ConversationIntelligence.app into a distributable .dmg with the
# standard drag-to-Applications layout. Run after:
#   pyinstaller packaging/desktop_app.spec --noconfirm
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

APP_PATH="dist/ConversationIntelligence.app"
DMG_NAME="ConversationIntelligence"
STAGING_DIR="dist/dmg_staging"
OUTPUT_DMG="dist/${DMG_NAME}.dmg"

if [[ ! -d "$APP_PATH" ]]; then
  echo "Error: $APP_PATH not found. Run 'pyinstaller packaging/desktop_app.spec --noconfirm' first." >&2
  exit 1
fi

rm -rf "$STAGING_DIR" "$OUTPUT_DMG"
mkdir -p "$STAGING_DIR"

cp -R "$APP_PATH" "$STAGING_DIR/"
ln -s /Applications "$STAGING_DIR/Applications"

hdiutil create \
  -volname "Conversation Intelligence" \
  -srcfolder "$STAGING_DIR" \
  -ov -format UDZO \
  "$OUTPUT_DMG"

rm -rf "$STAGING_DIR"

echo "Built: $OUTPUT_DMG"
du -h "$OUTPUT_DMG"
