#!/usr/bin/env bash
# Lino Neural Memory Server — auto-start helper
# Called by opencode session start and Hermes CLI init.
# Starts the server via systemd if not already running, then waits for readiness.

set -e

HOST=${LINO_HOST:-127.0.0.1}
PORT=${LINO_PORT:-8210}
TIMEOUT=${LINO_TIMEOUT:-30}

# 1. Check if already listening
if ss -tlnp "sport = :${PORT}" 2>/dev/null | grep -q LISTEN; then
  exit 0
fi

# 2. Try systemd user service (preferred)
if systemctl --user is-enabled lino-server.service &>/dev/null; then
  systemctl --user start lino-server.service
else
  # Fallback: start directly
  cd /home/Beelzebub/Documents/neural-memory.bak
  nohup python3 -m uvicorn ui.app:app --host 127.0.0.1 --port 8210 > /tmp/lino-server.log 2>&1 &
fi

# 3. Wait for readiness
for i in $(seq 1 "$TIMEOUT"); do
  if ss -tlnp "sport = :${PORT}" 2>/dev/null | grep -q LISTEN; then
    exit 0
  fi
  sleep 1
done

echo "Lino server failed to start within ${TIMEOUT}s" >&2
exit 1
