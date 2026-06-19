#!/usr/bin/env python3
"""CLI tool for the neural-memory plugin (standalone)."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from integration.hermes_plugin import MemoryPlugin


def main():
    parser = argparse.ArgumentParser(description="Neural Memory CLI")
    parser.add_argument("--store-path", help="Path to the memory store pickle file")
    sub = parser.add_subparsers(dest="command", required=True)

    # store
    p_store = sub.add_parser("store", help="Store a new memory")
    p_store.add_argument("text", help="Memory text to store")
    p_store.add_argument("--source", default="cli", help="Source tag")
    p_store.add_argument("--importance", type=float, default=0.5, help="Importance 0.0-1.0")
    p_store.add_argument("--tags", nargs="*", default=[], help="Tags")

    # search
    p_search = sub.add_parser("search", help="Semantic search")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("-k", type=int, default=5, help="Number of results")
    p_search.add_argument("--threshold", type=float, default=0.0, help="Similarity threshold")

    # get
    p_get = sub.add_parser("get", help="Get memory by ID")
    p_get.add_argument("id", help="Memory ID")

    # list
    p_list = sub.add_parser("list", help="List memories")
    p_list.add_argument("--source", help="Filter by source")
    p_list.add_argument("--tags", nargs="*", default=[], help="Filter by tags")
    p_list.add_argument("--limit", type=int, default=50, help="Max results")
    p_list.add_argument("--offset", type=int, default=0, help="Pagination offset")

    # stats
    sub.add_parser("stats", help="System statistics")

    # priority-update
    p_pri = sub.add_parser("priority-update", help="Update a memory's importance/priority score")
    p_pri.add_argument("memory_id", help="Memory ID to update")
    p_pri.add_argument("--priority", type=float, required=True, help="New priority 0.0-1.0")

    # sync
    p_sync = sub.add_parser("sync", help="Run vault-to-neural-memory sync")
    p_sync.add_argument("--no-link", action="store_true", help="Skip wikilink generation")
    p_sync.add_argument("--max-related", type=int, default=5, help="Max related memories per entry")

    # compress
    p_comp = sub.add_parser("compress", help="Run rip-and-compress pipeline")
    p_comp.add_argument("--dry-run", action="store_true", help="Show what would be compressed without doing it")
    p_comp.add_argument("--min-age", type=int, default=24, help="Minimum age in hours")
    p_comp.add_argument("--provider", default="groq", help="LLM provider (groq, openai, anthropic)")
    p_comp.add_argument("--model", default="gemini-2.5-flash", help="LLM model name")

    # session-done
    p_sd = sub.add_parser("session-done", help="Store session summary with key facts")
    p_sd.add_argument("summary", help="Session summary text")
    p_sd.add_argument("--project", default="", help="Project name")
    p_sd.add_argument("--goal", default="", help="Goal description")
    p_sd.add_argument("--importance", type=float, default=0.85, help="Importance 0.0-1.0")
    p_sd.add_argument("--decisions", nargs="*", default=[], help="Key decisions")
    p_sd.add_argument("--changes", nargs="*", default=[], help="Key changes")
    p_sd.add_argument("--facts", nargs="*", default=[], help="Important facts")

    # link
    p_link = sub.add_parser("link", help="Find similar memories and create bidirectional links")
    p_link.add_argument("memory_id", nargs="?", default=None, help="Memory ID to find connections for")
    p_link.add_argument("--all", action="store_true", dest="link_all", help="Link all memories")
    p_link.add_argument("--max-links", type=int, default=5, help="Maximum links to create")
    p_link.add_argument("--threshold", type=float, default=0.6, help="Similarity threshold")

    # watchdog
    p_watch = sub.add_parser("watchdog", help="Manage vault file watcher daemon")
    p_watch.add_argument("action", choices=["start", "stop", "status"], help="Action")

    args = parser.parse_args()
    plugin = MemoryPlugin(store_path=args.store_path)

    if args.command == "store":
        result = plugin.cmd_store(args.text, source=args.source, importance=args.importance, tags=args.tags or None)
        print(json.dumps(result, indent=2))

    elif args.command == "search":
        result = plugin.cmd_search(args.query, k=args.k, threshold=args.threshold)
        print(json.dumps(result, indent=2))

    elif args.command == "get":
        result = plugin.cmd_recall(args.id)
        print(json.dumps(result, indent=2))

    elif args.command == "list":
        result = plugin.cmd_list(source=args.source, tags=args.tags or None, limit=args.limit, offset=args.offset)
        print(json.dumps(result, indent=2))

    elif args.command == "stats":
        result = plugin.cmd_stats()
        print(json.dumps(result, indent=2))

    elif args.command == "priority-update":
        result = plugin.cmd_update_priority(args.memory_id, args.priority)
        print(json.dumps(result, indent=2))

    elif args.command == "sync":
        result = plugin.cmd_run_sync(no_link=args.no_link, max_related=args.max_related)
        print(json.dumps(result, indent=2))

    elif args.command == "compress":
        result = plugin.cmd_run_compress(
            dry_run=args.dry_run,
            min_age=args.min_age,
            provider=args.provider,
            model=args.model,
        )
        print(json.dumps(result, indent=2))

    elif args.command == "session-done":
        result = plugin.cmd_session_done(
            summary=args.summary,
            project=args.project,
            goal=args.goal,
            importance=args.importance,
            decisions=args.decisions or None,
            changes=args.changes or None,
            facts=args.facts or None,
        )
        print(json.dumps(result, indent=2))

    elif args.command == "link":
        if args.link_all:
            all_ids = plugin.cmd_list(limit=10000)["memories"]
            total = len(all_ids)
            linked = 0
            errors = 0
            print(f"Linking {total} memories...")
            for i, mem in enumerate(all_ids):
                try:
                    res = plugin.cmd_link(mem["id"], max_links=args.max_links, threshold=args.threshold)
                    if res.get("status") == "linked":
                        linked += res.get("links_created", 0)
                    print(f"  [{i+1}/{total}] {mem['id'][:12]}... → {res.get('links_created', 0)} links", flush=True)
                except Exception as e:
                    errors += 1
                    print(f"  [{i+1}/{total}] {mem['id'][:12]}... ERROR: {e}", flush=True)
            print(json.dumps({"total": total, "total_links_created": linked, "errors": errors}, indent=2))
        else:
            result = plugin.cmd_link(
                memory_id=args.memory_id,
                max_links=args.max_links,
                threshold=args.threshold,
            )
            print(json.dumps(result, indent=2))

    elif args.command == "watchdog":
        result = plugin.cmd_watchdog(action=args.action)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
