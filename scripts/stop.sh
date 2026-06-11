#!/bin/bash
# Lino — Stop Script
# Stops the lino server gracefully

set -euo pipefail

PID_FILE="${TMPDIR:-/tmp}/lino.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "[LINO] Stopping PID $PID..."
        kill -TERM "$PID"
        sleep 2
        if kill -0 "$PID" 2>/dev/null; then
            echo "[LINO] Force stopping..."
            kill -KILL "$PID" 2>/dev/null || true
        fi
        rm -f "$PID_FILE"
        echo "[LINO] Stopped."
    else
        echo "[LINO] PID $PID not running. Cleaning up."
        rm -f "$PID_FILE"
    fi
else
    # Try pkill fallback
    if pkill -f "uvicorn ui.app:app" 2>/dev/null; then
        echo "[LINO] Stopped via pkill."
    else
        echo "[LINO] No running Lino server found."
    fi
fi
