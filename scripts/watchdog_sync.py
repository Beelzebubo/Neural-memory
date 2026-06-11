#!/usr/bin/env python3
"""Real-time file watcher for Obsidian vault → neural-memory sync.

Uses the watchdog library to monitor the vault directory for .md file
changes. On create/modify, parses YAML frontmatter and syncs to the
vector store. On delete, removes the memory. Applies a 2-second debounce
to coalesce rapid save events.

Usage:
  python3 watchdog_sync.py
  python3 watchdog_sync.py --vault /path/to/vault
  python3 watchdog_sync.py --store /path/to/store.pkl
"""

import os
import sys
import time
import json
import glob
import logging
import argparse
from pathlib import Path
from collections import defaultdict

NEURAL_DIR = os.path.expanduser("~/Documents/neural-memory")
sys.path.insert(0, NEURAL_DIR)

from src import TextEmbedder, VectorMemoryStore

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    print("watchdog is required. Install with: pip install watchdog")
    sys.exit(1)

VAULT = os.path.expanduser("~/Documents/Hermes_memory")
STORE_PATH = os.path.expanduser("~/.neural_memory/store.pkl")
MAX_WORDS_PER_CHUNK = 800
DEBOUNCE_SECONDS = 2


def parse_frontmatter(content: str):
    """Parse YAML-like frontmatter from markdown content."""
    content = content.lstrip("\n")
    if not content.startswith("---"):
        return {}, content

    end_idx = content.find("---", 3)
    if end_idx == -1:
        return {}, content

    fm_block = content[3:end_idx].strip()
    body = content[end_idx + 3:].lstrip("\n")

    fm = {}
    for line in fm_block.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if value.startswith("[") and value.endswith("]"):
                try:
                    value = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    value = [v.strip().strip("\"'") for v in value[1:-1].split(",") if v.strip()]
            elif value.isdigit():
                value = int(value)
            else:
                try:
                    value = float(value)
                except ValueError:
                    value = value.strip("\"'")
            fm[key] = value

    return fm, body


def safe_id_from_path(fpath: str) -> str:
    fname = os.path.basename(fpath)
    return "vault_" + fname.replace(" ", "_").replace(".md", "").lower()


def sync_file(fpath: str, embedder: TextEmbedder, store: VectorMemoryStore):
    """Sync a single vault file to the vector store. Returns True if changed."""
    if not os.path.exists(fpath):
        return False

    fname = os.path.basename(fpath)
    mtime = os.path.getmtime(fpath)
    safe_id = safe_id_from_path(fpath)

    with open(fpath) as f:
        raw = f.read()

    fm, body = parse_frontmatter(raw)
    if not fm:
        logging.info("No frontmatter in %s, using defaults", fname)
        fm = {
            "memory_id": safe_id,
            "priority": 0.8,
            "tags": ["vault"],
            "tier": "active",
        }
        body = raw

    content = body.strip()
    if not content:
        logging.warning("Empty content in %s, skipping", fname)
        return False

    old_ids = [i for i in store.list_all() if i.startswith(safe_id)]
    for oid in old_ids:
        store.delete(oid)

    chunk_size = MAX_WORDS_PER_CHUNK * 5
    chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]

    priority = float(fm.get("priority", 0.8))
    tags = list(fm.get("tags", ["vault"]))
    tier = str(fm.get("tier", "active"))

    for ci, chunk in enumerate(chunks):
        chunk_id = safe_id if len(chunks) == 1 else f"{safe_id}_p{ci+1}"
        vec = embedder.embed(chunk)
        store.store(chunk_id, vec, {
            "text": chunk,
            "source": "obsidian-vault",
            "vault_file": fname,
            "importance_score": priority,
            "protected": True,
            "_vault_mtime": mtime,
            "tags": tags + ["vault", "obsidian-vault", fname.replace(".md", "").replace(" ", "-").lower()],
            "original_file": fpath,
            "chunk": ci + 1,
            "total_chunks": len(chunks),
            "tier": tier,
            "yaml_frontmatter": dict(fm),
        })

    return True


def remove_file(fpath: str, store: VectorMemoryStore):
    """Remove all chunks for a vault file from the store."""
    safe_id = safe_id_from_path(fpath)
    ids_to_remove = [i for i in store.list_all() if i.startswith(safe_id)]
    for oid in ids_to_remove:
        store.delete(oid)
    return len(ids_to_remove)


class VaultEventHandler(FileSystemEventHandler):
    def __init__(self, embedder, store, store_path):
        self.embedder = embedder
        self.store = store
        self.store_path = store_path
        self._debounce_timers = {}
        self._last_events = defaultdict(float)

    def on_modified(self, event):
        if event.is_directory:
            return
        if not event.src_path.endswith(".md"):
            return
        self._debounce(event.src_path, "modified")

    def on_created(self, event):
        if event.is_directory:
            return
        if not event.src_path.endswith(".md"):
            return
        self._debounce(event.src_path, "created")

    def on_deleted(self, event):
        if event.is_directory:
            return
        if not event.src_path.endswith(".md"):
            return
        self._process_delete(event.src_path)

    def on_moved(self, event):
        if event.is_directory:
            return
        if event.dest_path.endswith(".md"):
            self._debounce(event.dest_path, "created")
        if event.src_path.endswith(".md"):
            self._process_delete(event.src_path)

    def _debounce(self, fpath, event_type):
        now = time.time()
        key = fpath
        if now - self._last_events[key] < DEBOUNCE_SECONDS:
            logging.debug("Debounced %s for %s", event_type, os.path.basename(fpath))
        self._last_events[key] = now

        if key in self._debounce_timers:
            self._debounce_timers[key].cancel()

        import threading
        timer = threading.Timer(DEBOUNCE_SECONDS, self._process_file, args=[fpath])
        timer.daemon = True
        timer.start()
        self._debounce_timers[key] = timer

    def _process_file(self, fpath):
        fname = os.path.basename(fpath)
        logging.info("Processing %s", fname)
        try:
            changed = sync_file(fpath, self.embedder, self.store)
            if changed:
                self.store.save(str(self.store_path))
                logging.info("  Saved %s to store", fname)
        except Exception as e:
            logging.error("  Error processing %s: %s", fname, e)

    def _process_delete(self, fpath):
        fname = os.path.basename(fpath)
        logging.info("Removing %s from store", fname)
        try:
            removed = remove_file(fpath, self.store)
            if removed:
                self.store.save(str(self.store_path))
                logging.info("  Removed %d chunks for %s", removed, fname)
        except Exception as e:
            logging.error("  Error removing %s: %s", fname, e)


def main():
    parser = argparse.ArgumentParser(description="Watchdog sync for Obsidian vault → neural-memory")
    parser.add_argument("--vault", default=VAULT, help="Path to Obsidian vault directory")
    parser.add_argument("--store", default=STORE_PATH, help="Path to store.pkl")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    vault_path = os.path.expanduser(args.vault)
    store_path = Path(os.path.expanduser(args.store))
    store_path.parent.mkdir(parents=True, exist_ok=True)

    if not os.path.isdir(vault_path):
        logging.error("Vault directory does not exist: %s", vault_path)
        sys.exit(1)

    embedder = TextEmbedder()
    store = VectorMemoryStore()
    if store_path.exists():
        try:
            store.load(str(store_path))
            logging.info("Loaded store with %d entries", len(store))
        except Exception as e:
            logging.warning("Could not load store: %s", e)
    else:
        logging.info("No existing store, starting fresh")

    logging.info("WARNING: If the UI server is running, it may have the store in memory.")
    logging.info("Changes from watchdog will persist to disk but the server will need restart.")
    logging.info("Watching %s for .md changes...", vault_path)

    event_handler = VaultEventHandler(embedder, store, store_path)
    observer = Observer()
    observer.schedule(event_handler, vault_path, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Shutting down...")
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
