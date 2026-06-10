#!/usr/bin/env python3
"""
Neural Memory MCP Server — stdio JSON-RPC interface for Hermes.
Registers tools: read, write, search, list, delete, stats, prune, backup.

Usage (registration):
  hermes mcp add neural-memory --command 'python3 /path/to/mcp_server.py'

Then in any Hermes session, neural_memory_* tools appear automatically.
"""

import json
import sys
import os
import traceback

# Ensure we can import from project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from integration.hermes_plugin import MemoryPlugin


plugin = None  # Lazy initialized


def get_plugin():
    global plugin
    if plugin is None:
        plugin = MemoryPlugin()
    return plugin


def handle_request(req: dict) -> dict:
    method = req.get("method", "")
    params = req.get("params", {})
    req_id = req.get("id")

    if method == "initialize":
        client_version = params.get("protocolVersion", "2024-11-05")
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": client_version,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "neural-memory", "version": "1.0.0"},
            },
        }

    if method == "notifications/initialized":
        return None  # No response for notifications

    if method == "tools/list":
        tools = [
            {
                "name": "neural_memory_store",
                "description": "Store a new memory with text and optional metadata",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Memory text content"},
                        "source": {"type": "string", "description": "Source identifier (default: manual)"},
                        "importance": {"type": "number", "description": "Importance score 0.0-1.0 (default: 0.5)"},
                        "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags"},
                    },
                    "required": ["text"],
                },
            },
            {
                "name": "neural_memory_search",
                "description": "Semantic search across memories",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query text"},
                        "k": {"type": "number", "description": "Number of results (default: 10)"},
                        "threshold": {"type": "number", "description": "Minimum similarity score 0.0-1.0 (default: 0.0)"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "neural_memory_list",
                "description": "List memories with optional source/tag filtering",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "number", "description": "Max results (default: 50)"},
                        "offset": {"type": "number", "description": "Pagination offset (default: 0)"},
                        "source": {"type": "string", "description": "Filter by source"},
                        "tags": {"type": "array", "items": {"type": "string"}, "description": "Filter by tags (AND)"},
                    },
                },
            },
            {
                "name": "neural_memory_get",
                "description": "Get a single memory by ID",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string", "description": "Memory UUID"},
                    },
                    "required": ["memory_id"],
                },
            },
            {
                "name": "neural_memory_delete",
                "description": "Delete a memory by ID",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string", "description": "Memory UUID"},
                    },
                    "required": ["memory_id"],
                },
            },
            {
                "name": "neural_memory_stats",
                "description": "Get memory system statistics",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "neural_memory_prune",
                "description": "Prune memories by strategy",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "strategy": {"type": "string", "description": "Prune strategy: hybrid (default), importance, recency, random"},
                        "max_items": {"type": "number", "description": "Maximum memories after pruning (default: 10000)"},
                    },
                },
            },
        ]
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools}}

    if method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        try:
            p = get_plugin()
            result = p.execute(tool_name, arguments)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result, indent=2, default=str)}]
                },
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32000, "message": str(e), "data": traceback.format_exc()},
            }

    # Unknown method
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}


def main():
    """Stdio JSON-RPC loop."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            resp = {"jsonrpc": "2.0", "error": {"code": -32700, "message": f"Parse error: {e}"}, "id": None}
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
            continue

        resp = handle_request(req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
