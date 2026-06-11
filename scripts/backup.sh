#!/bin/bash
# Lino — Backup Script
# Creates timestamped backup of memory store + config

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BACKUP_DIR="$PROJECT_DIR/backups"
STORE_PATH="$PROJECT_DIR/data/memory_store.pkl"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/lino_memory_${TIMESTAMP}.tar.gz"

mkdir -p "$BACKUP_DIR"

echo "[LINO] Backing up..."

# Collect files to backup
FILES=()
[ -f "$STORE_PATH" ] && FILES+=("data/memory_store.pkl")
[ -f "$PROJECT_DIR/config/config.yaml" ] && FILES+=("config/config.yaml")
[ -f "$PROJECT_DIR/.env" ] && FILES+=(".env")

if [ ${#FILES[@]} -eq 0 ]; then
    echo "[LINO] Nothing to backup — no data found."
    exit 0
fi

cd "$PROJECT_DIR"
tar -czf "$BACKUP_FILE" "${FILES[@]}"

# Keep only last 10 backups
ls -t "$BACKUP_DIR"/*.tar.gz 2>/dev/null | tail -n +11 | xargs -r rm

SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
echo "[LINO] Backup saved: $BACKUP_FILE ($SIZE)"
echo "[LINO] Backups retained: $(ls "$BACKUP_DIR"/*.tar.gz 2>/dev/null | wc -l)"
