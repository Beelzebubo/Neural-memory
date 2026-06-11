#!/bin/bash
# Lino — Health Check Script
# Returns 0 if server is healthy, 1 otherwise

set -euo pipefail

HOST="${LINO_HOST:-127.0.0.1}"
PORT="${LINO_PORT:-8210}"
URL="http://${HOST}:${PORT}/ui/"

# Check if process is responding
if ! curl -sf -o /dev/null "$URL" 2>/dev/null; then
    echo "[LINO] UNHEALTHY — Server not responding on $URL"
    exit 1
fi

# Check API endpoints
if ! curl -sf -o /dev/null "http://${HOST}:${PORT}/api/memories/stats" 2>/dev/null; then
    echo "[LINO] DEGRADED — Web UI up but API down"
    exit 1
fi

echo "[LINO] HEALTHY — All endpoints up"
exit 0
