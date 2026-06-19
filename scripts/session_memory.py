#!/usr/bin/env python3
"""
session_memory.py — Session summarization & context retrieval for Lino.

Designed to be called by both opencode and Hermes at session boundaries.

Commands:
  summarize  — Send session log to LLM, extract key facts, store as memories
  store-fact — Quick single-fact storage with importance
  context    — Retrieve relevant past memories for a project/goal

Usage:
  python scripts/session_memory.py summarize \
    --text "session conversation log..." \
    --project "NYUAD" \
    --goal "admission planning" \
    --importance 0.9

  python scripts/session_memory.py store-fact \
    --text "Decided to use FastAPI for the backend" \
    --tag "decision" \
    --importance 0.85

  python scripts/session_memory.py context \
    --project "Lino" \
    --k 5
"""

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

LINO_API = os.environ.get("LINO_API_URL", "http://127.0.0.1:8210")
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-2.5-flash")
LLM_API_KEY = os.environ.get(f"{LLM_PROVIDER.upper()}_API_KEY", os.environ.get("GROQ_API_KEY", ""))

PROVIDER_ENDPOINTS = {
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "openai": "https://api.openai.com/v1/chat/completions",
    "anthropic": "https://api.anthropic.com/v1/messages",
}

SUMMARY_PROMPT = """Extract the key information from this conversation session.

Return a JSON object with these fields:
- "project": the main project or topic being worked on (string)
- "goal": the primary goal or objective (string)
- "key_decisions": list of important decisions made (array of strings)
- "key_changes": list of code/config changes made (array of strings)
- "bugs_fixed": list of bugs resolved (array of strings)
- "important_facts": list of important facts learned or established (array of strings)
- "action_items": list of things to do next (array of strings)
- "overall_summary": a 2-3 sentence summary of what was accomplished (string)

Only include items that are genuinely important. Omit trivial conversation, greetings, small talk, and repetitive content.

If a category has no items, use an empty array [].
If the session has no meaningful work, set overall_summary to an empty string.

Return ONLY the JSON object, no other text."""


def call_llm(text: str) -> dict:
    """Send session text to LLM for fact extraction."""
    if not LLM_API_KEY:
        print("Warning: No LLM API key set. Using heuristic extraction.", file=sys.stderr)
        return _heuristic_extract(text)

    endpoint = PROVIDER_ENDPOINTS.get(LLM_PROVIDER, PROVIDER_ENDPOINTS["groq"])
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": text[-8000:]},
        ],
        "temperature": 0.3,
        "max_tokens": 2048,
        "response_format": {"type": "json_object"},
    }

    try:
        import urllib.request
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "overall_summary" in parsed:
            return parsed
    except Exception as e:
        print(f"LLM call failed: {e}", file=sys.stderr)

    return _heuristic_extract(text)


def _heuristic_extract(text: str) -> dict:
    """Fallback: simple heuristic extraction when no LLM is available."""
    lines = text.split("\n")
    key_lines = [l.strip() for l in lines if l.strip() and len(l.strip()) > 40]
    return {
        "project": "unknown",
        "goal": "unknown",
        "key_decisions": key_lines[:3] if key_lines else [],
        "key_changes": [],
        "bugs_fixed": [],
        "important_facts": key_lines[:5] if key_lines else [],
        "action_items": [],
        "overall_summary": text[:500] if text else "",
    }


def store_memory(text: str, source: str, tags: list, importance: float) -> bool:
    """Store a memory via Lino REST API."""
    payload = json.dumps({
        "text": text,
        "source": source,
        "importance": importance,
        "tags": tags,
    }).encode()
    try:
        import urllib.request
        req = urllib.request.Request(
            f"{LINO_API}/api/memories",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"Failed to store memory: {e}", file=sys.stderr)
        return False


def search_memories(query: str, k: int = 5) -> list:
    """Search memories via Lino REST API."""
    payload = json.dumps({"query": query, "k": k}).encode()
    try:
        import urllib.request
        req = urllib.request.Request(
            f"{LINO_API}/api/search",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        return data.get("results", [])
    except Exception as e:
        print(f"Search failed: {e}", file=sys.stderr)
        return []


def cmd_summarize(args):
    """Extract key facts from a session log and store them."""
    text = args.text or sys.stdin.read()
    if not text.strip():
        print("No text provided")
        return 1

    facts = call_llm(text)

    if not facts.get("overall_summary"):
        print("No meaningful content extracted — nothing stored")
        return 0

    facts_text = json.dumps(facts, indent=2)
    summary = facts.get("overall_summary", "")
    project = facts.get("project", args.project or "unknown")
    base_tags = args.tags + ["session-summary", f"project:{project}"]

    stored_count = 0
    importance = args.importance

    # Store the overall summary at highest importance
    if summary:
        tags = base_tags + ["summary"]
        if store_memory(
            f"[{project}] Session summary: {summary}",
            source=f"session-summary:{project}",
            tags=tags,
            importance=min(importance + 0.05, 1.0),
        ):
            stored_count += 1

    # Store individual key facts at slightly lower importance
    for category, label in [
        ("key_decisions", "decision"),
        ("key_changes", "change"),
        ("bugs_fixed", "bugfix"),
        ("important_facts", "fact"),
        ("action_items", "action-item"),
    ]:
        items = facts.get(category, [])
        for item in items:
            tags = base_tags + [label, f"category:{category}"]
            if store_memory(
                f"[{project}] {item}",
                source=f"session-fact:{project}",
                tags=tags,
                importance=importance - 0.05,
            ):
                stored_count += 1

    print(json.dumps({
        "status": "stored",
        "stored_count": stored_count,
        "project": project,
        "summary": summary,
        "facts": facts,
    }, indent=2))
    return 0


def cmd_store_fact(args):
    """Store a single fact."""
    if not args.text:
        print("No text provided")
        return 1

    tags = args.tags
    if args.tag:
        tags = tags + [args.tag]

    ok = store_memory(args.text, source=args.source, tags=tags, importance=args.importance)
    if ok:
        print(json.dumps({"status": "stored", "text": args.text, "importance": args.importance}, indent=2))
        return 0
    print("Failed to store fact")
    return 1


def cmd_context(args):
    """Retrieve relevant past memories."""
    if not args.project:
        print("No project specified — searching without filter")
    query = args.query or args.project or ""

    results = search_memories(query, k=args.k)
    print(json.dumps({"results": results, "query": query, "k": args.k}, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description="Lino Session Memory Tool")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sum = sub.add_parser("summarize", help="Extract + store key facts from a session log")
    p_sum.add_argument("--text", help="Session conversation text (omit to read stdin)")
    p_sum.add_argument("--project", default="", help="Project name")
    p_sum.add_argument("--goal", default="", help="Goal description")
    p_sum.add_argument("--importance", type=float, default=0.85, help="Importance 0.0-1.0")
    p_sum.add_argument("--tags", nargs="*", default=[], help="Extra tags")

    p_fact = sub.add_parser("store-fact", help="Store a single fact quickly")
    p_fact.add_argument("--text", required=True, help="Fact text")
    p_fact.add_argument("--tag", help="Single category tag (e.g. decision, bugfix)")
    p_fact.add_argument("--source", default="session-memory", help="Source identifier")
    p_fact.add_argument("--importance", type=float, default=0.8, help="Importance 0.0-1.0")
    p_fact.add_argument("--tags", nargs="*", default=[], help="Extra tags")

    p_ctx = sub.add_parser("context", help="Retrieve context for a project")
    p_ctx.add_argument("--project", default="", help="Project name")
    p_ctx.add_argument("--query", default="", help="Custom search query (overrides --project)")
    p_ctx.add_argument("-k", type=int, default=5, help="Number of results")

    args = parser.parse_args()

    if args.command == "summarize":
        return cmd_summarize(args)
    elif args.command == "store-fact":
        return cmd_store_fact(args)
    elif args.command == "context":
        return cmd_context(args)

    return 1


if __name__ == "__main__":
    sys.exit(main())
