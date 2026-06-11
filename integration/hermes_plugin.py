import inspect
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from src import TextEmbedder, VectorMemoryStore, MemoryRetriever, MemoryConsolidator

TOOL_SCHEMAS = {
    "neural_memory_store": {
        "description": "Store a text memory with metadata. Text is embedded and stored for later semantic retrieval.",
        "parameters": {
            "text": {"type": "string", "description": "The memory text to store"},
            "source": {"type": "string", "description": "Optional source tag", "nullable": True},
            "importance": {"type": "number", "description": "Importance 0.0-1.0 (default 0.5)", "nullable": True},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional list of tags", "nullable": True},
        },
        "required": ["text"],
    },
    "neural_memory_search": {
        "description": "Semantically search stored memories by query text.",
        "parameters": {
            "query": {"type": "string", "description": "The search query"},
            "k": {"type": "integer", "description": "Number of results (default 5)", "nullable": True},
            "threshold": {"type": "number", "description": "Similarity threshold (default 0.0)", "nullable": True},
        },
        "required": ["query"],
    },
    "neural_memory_recall": {
        "description": "Retrieve a specific memory by its ID.",
        "parameters": {
            "id": {"type": "string", "description": "Memory ID to retrieve"},
        },
        "required": ["id"],
    },
    "neural_memory_list": {
        "description": "List all stored memories with optional filters.",
        "parameters": {
            "source": {"type": "string", "description": "Filter by source", "nullable": True},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "Filter by tags", "nullable": True},
            "limit": {"type": "integer", "description": "Max results (default 50)", "nullable": True},
            "offset": {"type": "integer", "description": "Pagination offset", "nullable": True},
        },
    },
    "neural_memory_stats": {
        "description": "Get neural memory system statistics.",
        "parameters": {},
    },
    "neural_memory_update_priority": {
        "description": "Update a memory's importance/priority score in-place without re-embedding.",
        "parameters": {
            "memory_id": {"type": "string", "description": "Memory ID to update"},
            "priority": {"type": "number", "description": "New priority value 0.0-1.0"},
        },
        "required": ["memory_id", "priority"],
    },
    "neural_memory_run_sync": {
        "description": "Run an immediate vault-to-neural-memory sync with optional wikilink generation.",
        "parameters": {
            "no_link": {"type": "boolean", "description": "Skip wikilink generation (default: false)", "nullable": True},
            "max_related": {"type": "integer", "description": "Max related memories per entry (default: 5)", "nullable": True},
        },
    },
    "neural_memory_run_compress": {
        "description": "Run the priority-tier rip-and-compress pipeline on transient/low-priority memories.",
        "parameters": {
            "dry_run": {"type": "boolean", "description": "Show what would be compressed without doing it (default: false)", "nullable": True},
            "min_age": {"type": "integer", "description": "Minimum age in hours (default: 24)", "nullable": True},
            "provider": {"type": "string", "description": "LLM provider: groq, openai, anthropic (default: groq)", "nullable": True},
            "model": {"type": "string", "description": "LLM model name (default: gemini-2.5-flash)", "nullable": True},
        },
    },
    "neural_memory_watchdog": {
        "description": "Manage the real-time vault file watcher daemon (start/stop/status).",
        "parameters": {
            "action": {"type": "string", "description": "Action: start, stop, or status (default: status)"},
        },
        "required": ["action"],
    },
}

DEFAULT_STORE_PATH = Path.home() / ".neural_memory" / "store.pkl"


class MemoryPlugin:
    def __init__(self, store_path: Optional[str] = None):
        raw = store_path or os.environ.get("NEURAL_MEMORY_PATH", "")
        if raw:
            self.store_path = Path(raw).resolve()
            # Restrict store path to home directory or a known safe location
            home = Path.home().resolve()
            if not str(self.store_path).startswith(str(home)):
                self.store_path = DEFAULT_STORE_PATH
        else:
            self.store_path = DEFAULT_STORE_PATH
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.store = VectorMemoryStore()
        self.consolidator = MemoryConsolidator()
        self.embedder = None
        self.retriever = None
        self._init_embedder()
        self._load_store()
        self._init_retriever()

    def _init_embedder(self):
        try:
            self.embedder = TextEmbedder()
            self.embedder_online = True
        except Exception:
            self.embedder = None
            self.embedder_online = False

    def _init_retriever(self):
        if self.embedder and self.store:
            self.retriever = MemoryRetriever(self.embedder, self.store)
        else:
            self.retriever = None

    def _load_store(self):
        if self.store_path.exists():
            try:
                self.store.load(str(self.store_path))
            except Exception:
                pass

    def _save_store(self):
        self.store.save(str(self.store_path))

    def _get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        ids = self.store.list_all()
        if memory_id not in ids:
            return None
        idx = ids.index(memory_id)
        meta = dict(self.store._metadata.get(memory_id, {}))
        emb = self.store._embeddings[idx] if idx < len(self.store._embeddings) else None
        return {
            "id": memory_id,
            "text": meta.pop("text", ""),
            "embedding": emb.tolist() if emb is not None else None,
            "metadata": meta,
        }

    def _list_memories(self, limit: int = 50, offset: int = 0) -> tuple:
        ids = self.store.list_all()
        total = len(ids)
        page = ids[offset:offset + limit]
        results = []
        for mid in page:
            idx = ids.index(mid)
            meta = dict(self.store._metadata.get(mid, {}))
            emb = self.store._embeddings[idx] if idx < len(self.store._embeddings) else None
            results.append({
                "id": mid,
                "text": meta.pop("text", ""),
                "metadata": meta,
            })
        return results, total

    def tools(self) -> List[Dict[str, Any]]:
        return [
            {"name": name, "schema": schema}
            for name, schema in TOOL_SCHEMAS.items()
        ]

    def execute(self, tool_name: str, arguments: dict) -> Dict[str, Any]:
        handler = getattr(self, f"cmd_{tool_name.replace('neural_memory_', '')}", None)
        if handler is None:
            return {"error": f"Unknown tool: {tool_name}"}
        sig = inspect.signature(handler)
        allowed = set(sig.parameters.keys()) - {"self"}
        extra = set(arguments.keys()) - allowed
        if extra:
            return {"error": f"Unexpected arguments: {', '.join(sorted(extra))}"}
        try:
            return handler(**{
                k: v for k, v in arguments.items() if k in allowed
            })
        except Exception as e:
            return {"error": str(e)}

    def cmd_store(self, text: str, source: str = "hermes", importance: float = 0.5, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        memory_id = str(uuid.uuid4())
        metadata = {
            "text": text,
            "source": source,
            "importance_score": importance,
            "access_count": 0,
            "timestamp": time.time(),
        }
        if tags:
            metadata["tags"] = tags
        if self.embedder_online and self.embedder:
            emb = self.embedder.embed(text)
        else:
            import numpy as np
            emb = np.zeros(384).tolist()
        self.store.store(memory_id, emb, metadata)
        self._save_store()
        return {
            "status": "stored",
            "id": memory_id,
            "text": text,
            "source": source,
            "importance": importance,
            "tags": tags or [],
        }

    def cmd_search(self, query: str, k: int = 5, threshold: float = 0.0) -> Dict[str, Any]:
        if not self.retriever:
            return {"results": [], "note": "Embedder offline; no semantic search available"}
        results = self.retriever.retrieve(query, k=k, min_score=threshold)
        enriched = []
        for r in results:
            meta = dict(r.get("metadata", {}))
            text = meta.pop("text", "")
            enriched.append({
                "id": r["id"],
                "score": r["score"],
                "text": text,
                "metadata": meta,
            })
        return {"results": enriched, "query": query, "k": k}

    def cmd_recall(self, id: str) -> Dict[str, Any]:
        mem = self._get_memory(id)
        if mem is None:
            return {"error": f"Memory not found: {id}"}
        return mem

    def cmd_list(self, source: Optional[str] = None, tags: Optional[List[str]] = None, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        all_results, total = self._list_memories(limit=limit, offset=offset)
        if source:
            all_results = [m for m in all_results if m.get("metadata", {}).get("source") == source]
        if tags:
            tag_set = set(tags)
            all_results = [
                m for m in all_results
                if tag_set.intersection(set(m.get("metadata", {}).get("tags", [])))
            ]
        return {"memories": all_results, "total": len(all_results)}

    def cmd_stats(self) -> Dict[str, Any]:
        total = len(self.store)
        dim = self.store._dim if hasattr(self.store, "_dim") else 0
        return {
            "total_memories": total,
            "dimension": dim,
            "embedder_online": self.embedder_online,
            "store_path": str(self.store_path),
            "status": "online" if self.embedder_online else "degraded",
        }

    def cmd_update_priority(self, memory_id: str, priority: float) -> Dict[str, Any]:
        """Update a memory's importance/priority score in-place without re-embedding."""
        ids = self.store.list_all()
        if memory_id not in ids:
            return {"error": f"Memory not found: {memory_id}"}
        priority = max(0.0, min(1.0, float(priority)))
        self.store._metadata[memory_id]["importance_score"] = priority
        self._save_store()
        return {"status": "updated", "memory_id": memory_id, "priority": priority}

    def cmd_run_sync(self, no_link: bool = False, max_related: int = 5) -> Dict[str, Any]:
        """Run the vault-to-neural-memory sync script with optional wikilinks."""
        import subprocess
        sync_script = str(Path(__file__).resolve().parent.parent.parent / ".hermes" / "scripts" / "sync_vault_to_neural_memory.py")
        if not os.path.exists(sync_script):
            sync_script = str(Path.home() / ".hermes" / "scripts" / "sync_vault_to_neural_memory.py")
        cmd = [sys.executable, sync_script]
        if no_link:
            cmd.append("--no-link")
        if max_related != 5:
            cmd.append(f"--max-related={max_related}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            return {
                "status": "completed" if result.returncode == 0 else "failed",
                "exit_code": result.returncode,
                "output": result.stdout,
                "errors": result.stderr or None,
            }
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "output": "Sync script exceeded 5-minute timeout"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def cmd_run_compress(self, dry_run: bool = False, min_age: int = 24,
                         provider: str = "groq", model: str = "gemini-2.5-flash") -> Dict[str, Any]:
        """Run the priority-tier rip-and-compress pipeline."""
        import subprocess
        compress_script = str(Path(__file__).resolve().parent.parent / "scripts" / "rip_and_compress.py")
        cmd = [sys.executable, compress_script]
        if dry_run:
            cmd.append("--dry-run")
        if min_age != 24:
            cmd.append(f"--min-age={min_age}")
        if provider != "groq":
            cmd.extend(["--provider", provider])
        if model != "gemini-2.5-flash":
            cmd.extend(["--model", model])
        try:
            env = os.environ.copy()
            # Pass through provider API keys from the Hermes environment
            for key_var in (f"{provider.upper()}_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
                if key_var in env:
                    break
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
            return {
                "status": "completed" if result.returncode == 0 else "failed",
                "exit_code": result.returncode,
                "output": result.stdout,
                "errors": result.stderr or None,
            }
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "output": "Compress script exceeded 5-minute timeout"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def cmd_watchdog(self, action: str = "status") -> Dict[str, Any]:
        """Manage the real-time vault file watcher daemon."""
        import subprocess
        pid_file = str(Path.home() / ".neural_memory" / "watchdog.pid")
        watchdog_script = str(Path(__file__).resolve().parent.parent / "scripts" / "watchdog_sync.py")

        if action == "start":
            if os.path.exists(pid_file):
                try:
                    with open(pid_file) as f:
                        pid = int(f.read().strip())
                    os.kill(pid, 0)  # Check if alive
                    return {"status": "already_running", "pid": pid}
                except (OSError, ValueError):
                    os.remove(pid_file)
            # Start the watchdog
            try:
                proc = subprocess.Popen(
                    [sys.executable, watchdog_script],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
                Path(pid_file).parent.mkdir(parents=True, exist_ok=True)
                with open(pid_file, "w") as f:
                    f.write(str(proc.pid))
                return {"status": "started", "pid": proc.pid}
            except Exception as e:
                return {"status": "error", "error": str(e)}

        elif action == "stop":
            if not os.path.exists(pid_file):
                return {"status": "not_running"}
            try:
                with open(pid_file) as f:
                    pid = int(f.read().strip())
                os.kill(pid, 15)  # SIGTERM
                os.remove(pid_file)
                return {"status": "stopped", "pid": pid}
            except ProcessLookupError:
                os.remove(pid_file)
                return {"status": "stopped", "pid": None, "note": "Process was already dead"}
            except Exception as e:
                return {"status": "error", "error": str(e)}

        else:  # status
            if not os.path.exists(pid_file):
                return {"status": "stopped"}
            try:
                with open(pid_file) as f:
                    pid = int(f.read().strip())
                os.kill(pid, 0)
                return {"status": "running", "pid": pid}
            except (OSError, ValueError):
                if os.path.exists(pid_file):
                    os.remove(pid_file)
                return {"status": "stopped", "note": "Stale PID file cleaned up"}
            return {"status": "stopped"}
