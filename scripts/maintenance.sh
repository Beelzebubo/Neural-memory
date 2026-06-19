#!/bin/bash
# Lino nightly maintenance: compress + prune old memories
# Runs via cron at 3 AM daily

set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$DIR/logs/maintenance.log"
mkdir -p "$DIR/logs"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting maintenance..." >> "$LOG"

# Phase 1: Compress old/low-importance memories (if LLM API key available)
if [ -n "${GROQ_API_KEY:-}" ] || [ -n "${OPENAI_API_KEY:-}" ] || [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    echo "  Running rip-and-compress..." >> "$LOG"
    python "$DIR/scripts/rip_and_compress.py" --min-age 48 >> "$LOG" 2>&1
else
    echo "  Skipping compress (no LLM API key)" >> "$LOG"
fi

# Phase 2: Prune low-importance memories
echo "  Running prune..." >> "$LOG"
curl -s -X POST http://127.0.0.1:8210/api/prune \
    -H "Content-Type: application/json" \
    -d '{"strategy":"hybrid","max_items":5000}' >> "$LOG" 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Maintenance complete." >> "$LOG"
