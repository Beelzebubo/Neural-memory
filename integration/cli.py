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

    p_store = sub.add_parser("store", help="Store a new memory")
    p_store.add_argument("text", help="Memory text to store")
    p_store.add_argument("--source", default="cli", help="Source tag")
    p_store.add_argument("--importance", type=float, default=0.5, help="Importance 0.0-1.0")
    p_store.add_argument("--tags", nargs="*", default=[], help="Tags")

    p_search = sub.add_parser("search", help="Semantic search")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("-k", type=int, default=5, help="Number of results")
    p_search.add_argument("--threshold", type=float, default=0.0, help="Similarity threshold")

    p_get = sub.add_parser("get", help="Get memory by ID")
    p_get.add_argument("id", help="Memory ID")

    p_list = sub.add_parser("list", help="List memories")
    p_list.add_argument("--source", help="Filter by source")
    p_list.add_argument("--tags", nargs="*", default=[], help="Filter by tags")
    p_list.add_argument("--limit", type=int, default=50, help="Max results")
    p_list.add_argument("--offset", type=int, default=0, help="Pagination offset")

    sub.add_parser("stats", help="System statistics")

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


if __name__ == "__main__":
    main()
