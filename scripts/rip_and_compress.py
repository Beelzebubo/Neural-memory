#!/usr/bin/env python3
"""Priority-tier rip-and-compress pipeline for neural-memory.

Queries the vector store for entries where tier=="transient" OR
importance_score < 0.5 AND age > 24 hours. Sends each qualifying entry
to an LLM for factual compression, re-embeds the compressed text, and
updates both the pickle store and the Obsidian vault file.

Usage:
  python3 rip_and_compress.py                  # Normal run
  python3 rip_and_compress.py --dry-run        # Show what would be compressed
  python3 rip_and_compress.py --min-age 48     # Only compress entries older than 48h
  python3 rip_and_compress.py --provider openai # Use OpenAI instead of Groq
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

NEURAL_DIR = os.path.expanduser("~/Documents/Lino")
sys.path.insert(0, NEURAL_DIR)

from src import TextEmbedder, VectorMemoryStore

LLM_CONFIG = {
    "provider": "groq",
    "model": "gemini-2.5-flash",
    "api_key": os.environ.get("GROQ_API_KEY", ""),
    "endpoint": "https://api.groq.com/openai/v1/chat/completions",
}

PROVIDER_MAP = {
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "openai": "https://api.openai.com/v1/chat/completions",
    "anthropic": "https://api.anthropic.com/v1/messages",
}

VAULT = os.path.expanduser("~/Documents/AI_MEMORIES")
STORE_PATH = os.path.expanduser("~/.neural_memory/store.pkl")
DEFAULT_MIN_AGE_HOURS = 24
COMPRESS_PROMPT = (
    "Extract only the core dense factual information from this text. "
    "Remove all conversational fluff, repetitions, and low-value filler. "
    "Return just the facts.\n\n"
)


def get_llm_endpoint(provider: str) -> str:
    return PROVIDER_MAP.get(provider, LLM_CONFIG["endpoint"])


def call_llm(text: str, config: dict) -> str:
    """Send text to LLM for compression. Returns compressed text."""
    import requests

    provider = config.get("provider", "groq")
    endpoint = config.get("endpoint", get_llm_endpoint(provider))
    api_key = config.get("api_key", "")
    model = config.get("model", "gemini-2.5-flash")

    if not api_key:
        raise ValueError(f"No API key configured for provider '{provider}'. Set the appropriate env var.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": COMPRESS_PROMPT + text}
        ],
        "temperature": 0.3,
        "max_tokens": 2048,
    }

    resp = requests.post(endpoint, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    if provider == "anthropic":
        return data.get("content", [{}])[0].get("text", "")
    return data.get("choices", [{}])[0].get("message", {}).get("content", "")


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


def build_frontmatter(fm: dict) -> str:
    """Build YAML frontmatter string from dictionary."""
    lines = ["---"]
    for key in ("memory_id", "priority", "tags", "tier", "last_accessed", "related_memories"):
        if key in fm:
            value = fm[key]
            if isinstance(value, list):
                items = []
                for v in value:
                    if isinstance(v, str) and v.startswith("[["):
                        items.append(v)
                    elif isinstance(v, str):
                        items.append(f"\"{v}\"")
                    else:
                        items.append(str(v))
                lines.append(f"{key}: [{', '.join(items)}]")
            elif isinstance(value, float):
                lines.append(f"{key}: {value}")
            elif isinstance(value, bool):
                lines.append(f"{key}: {'true' if value else 'false'}")
            elif isinstance(value, int):
                lines.append(f"{key}: {value}")
            else:
                lines.append(f"{key}: \"{value}\"")
    lines.append("---")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Priority-tier rip-and-compress pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be compressed without doing it")
    parser.add_argument("--min-age", type=int, default=DEFAULT_MIN_AGE_HOURS, help=f"Minimum age in hours (default: {DEFAULT_MIN_AGE_HOURS})")
    parser.add_argument("--provider", choices=list(PROVIDER_MAP.keys()) + ["custom"], default="groq", help="LLM provider")
    parser.add_argument("--model", default=LLM_CONFIG["model"], help="LLM model name")
    parser.add_argument("--endpoint", default="", help="Custom API endpoint URL")
    args = parser.parse_args()

    config = dict(LLM_CONFIG)
    config["provider"] = args.provider
    config["model"] = args.model
    if args.endpoint:
        config["endpoint"] = args.endpoint
    else:
        config["endpoint"] = get_llm_endpoint(args.provider) if args.provider != "custom" else ""

    embedder = TextEmbedder()
    store = VectorMemoryStore()
    store_path = Path(os.path.expanduser(STORE_PATH))
    if store_path.exists():
        store.load(str(store_path))
    else:
        print("No store found at", STORE_PATH)
        return 1

    vault = os.path.expanduser(VAULT)
    now = time.time()
    min_age_seconds = args.min_age * 3600

    candidates = []
    for mid in store.list_all():
        meta = store.get_metadata(mid) or {}
        tier = meta.get("tier", "active")
        importance = meta.get("importance_score", 0.5)
        timestamp = meta.get("timestamp", now)
        age = now - timestamp

        if tier == "transient" or (importance < 0.5 and age > min_age_seconds):
            text = meta.get("text", "")
            if text:
                candidates.append((mid, meta, text))

    if not candidates:
        print("No entries qualify for compression.")
        return 0

    if args.dry_run:
        print(f"DRY RUN — {len(candidates)} entries would be compressed:\n")
        for mid, meta, text in candidates:
            fname = meta.get("vault_file", "unknown")
            tier = meta.get("tier", "active")
            imp = meta.get("importance_score", 0.5)
            words = len(text.split())
            print(f"  {mid} ({fname}) — tier={tier}, importance={imp}, {words} words")
        print(f"\nWould compress: {len(candidates)}")
        return 0

    if not config["api_key"]:
        provider = config["provider"]
        key_var = f"{provider.upper()}_API_KEY"
        print(f"Error: No API key for '{provider}'. Set {key_var} environment variable.")
        return 1

    print(f"Compressing {len(candidates)} entries with {config['provider']}/{config['model']}...\n")

    compressed_count = 0
    skipped_count = 0
    error_count = 0

    for mid, meta, text in candidates:
        fname = meta.get("vault_file", "unknown")
        words = len(text.split())
        print(f"  Processing {mid} ({fname}, {words} words)...", end=" ")

        try:
            compressed = call_llm(text, config)
            if not compressed.strip():
                print("SKIPPED (empty response)")
                skipped_count += 1
                continue
        except Exception as e:
            print(f"ERROR: {e}")
            error_count += 1
            continue

        compressed_words = len(compressed.split())
        ratio = compressed_words / max(words, 1)
        print(f"compressed {words}→{compressed_words} words ({ratio:.0%})")

        # Re-embed the compressed text
        try:
            new_emb = embedder.embed(compressed)
            idx = store.get_index_of(mid)
            if idx is None:
                print(f"ERROR: memory {mid} not found in store")
                error_count += 1
                continue
            store.set_embedding(idx, np.array(new_emb, dtype=np.float32))
            store.update_metadata_value(mid, "text", compressed)
            store.update_metadata_value(mid, "tier", "compressed")
            old_imp = (store.get_metadata(mid) or {}).get("importance_score", 0.5)
            store.update_metadata_value(mid, "importance_score", min(old_imp + 0.1, 0.9))
            store.update_metadata_value(mid, "_compressed_at", now)
        except Exception as e:
            print(f"  ERROR updating store: {e}")
            error_count += 1
            continue

        # Update the vault file if applicable
        fpath = meta.get("original_file", "")
        if fpath and os.path.exists(fpath):
            try:
                with open(fpath) as f:
                    raw = f.read()
                fm, _ = parse_frontmatter(raw)
                fm["tier"] = "compressed"
                updated_meta = store.get_metadata(mid) or {}
                fm["priority"] = updated_meta.get("importance_score", 0.9)
                fm["last_accessed"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                new_content = build_frontmatter(fm) + "\n" + compressed
                with open(fpath, "w") as f:
                    f.write(new_content)
            except Exception as e:
                print(f"  WARNING: could not update vault file: {e}")

        compressed_count += 1

    store.save(str(store_path))
    print(f"\nCompressed: {compressed_count}, Skipped: {skipped_count}, Errors: {error_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
