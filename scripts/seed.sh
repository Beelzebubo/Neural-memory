#!/bin/bash
# Lino — Seed Script
# Imports Obsidian vault markdown files into Lino
# Uses the CLI tool to import each .md file

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

[ -f .env ] && set -a && source .env && set +a

VAULT_PATH="${OBSIDIAN_VAULT_PATH:-$HOME/Documents/AI_MEMORIES}"
HOST="${LINO_HOST:-127.0.0.1}"
PORT="${LINO_PORT:-8210}"
API_BASE="http://${HOST}:${PORT}/api"

echo "[LINO] Seeding from vault: $VAULT_PATH"

if [ ! -d "$VAULT_PATH" ]; then
    echo "[LINO] Error: vault path not found: $VAULT_PATH"
    echo "Set OBSIDIAN_VAULT_PATH in .env"
    exit 1
fi

# Check server is running
if ! curl -sf "$API_BASE/memories/stats" > /dev/null 2>&1; then
    echo "[LINO] Server not running at $API_BASE"
    echo "Start it first: ./scripts/start.sh prod"
    exit 1
fi

COUNT=0
SKIP=0

while IFS= read -r -d '' file; do
    REL_PATH="${file#$VAULT_PATH/}"
    CONTENT=$(cat "$file")

    # Skip binary or empty files
    [ -z "$CONTENT" ] && { SKIP=$((SKIP + 1)); continue; }

    # Check if already imported by path
    EXISTING=$(curl -sf "$API_BASE/memories?source=$REL_PATH" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d))" 2>/dev/null || echo "0")

    if [ "$EXISTING" -gt 0 ]; then
        SKIP=$((SKIP + 1))
        continue
    fi

    TITLE=$(head -1 "$file" | sed 's/^# //' | tr -dc '[:print:]' || echo "$REL_PATH")

    curl -sf -X POST "$API_BASE/memories" \
        -H "Content-Type: application/json" \
        -d "{\"text\": $(python3 -c "import json; print(json.dumps('$TITLE\n\n$CONTENT'))"), \"source\": \"$REL_PATH\", \"importance\": 1.0}" \
        > /dev/null 2>&1 && COUNT=$((COUNT + 1)) || SKIP=$((SKIP + 1))

    # Brief delay to avoid hammering
    [ $((COUNT % 5)) -eq 0 ] && sleep 0.5

done < <(find "$VAULT_PATH" -name "*.md" -print0)

echo "[LINO] Seeded $COUNT new memories ($SKIP skipped)"
