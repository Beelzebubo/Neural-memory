#!/bin/bash
# Lino — Start Script
# Usage: ./scripts/start.sh [--dev|--prod]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Load .env
[ -f .env ] && set -a && source .env && set +a

MODE="${1:-prod}"
HOST="${LINO_HOST:-127.0.0.1}"
PORT="${LINO_PORT:-8210}"
WORKERS="${LINO_WORKERS:-2}"

# Ensure venv
if [ ! -d .venv ]; then
    echo "[LINO] Creating virtual environment..."
    python3 -m venv .venv
    .venv/bin/pip install -q -r requirements.txt
fi

case "$MODE" in
    dev)
        echo "[LINO] Starting in DEV mode on $HOST:$PORT"
        exec .venv/bin/python -m uvicorn ui.app:app --host "$HOST" --port "$PORT" --reload
        ;;
    prod)
        echo "[LINO] Starting in PROD mode on $HOST:$PORT ($WORKERS workers)"
        mkdir -p logs
        exec .venv/bin/python -m uvicorn ui.app:app --host "$HOST" --port "$PORT" --workers "$WORKERS" --log-file logs/lino.log
        ;;
    *)
        echo "Usage: $0 [dev|prod]"
        exit 1
        ;;
esac
