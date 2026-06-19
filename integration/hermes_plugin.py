import inspect
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from src import TextEmbedder, VectorMemoryStore, MemoryRetriever, MemoryConsolidator

logger = logging.getLogger(__name__)

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
    "neural_memory_session_done": {
        "description": "Call at session end. Stores a structured session summary with key facts, decisions, and actions. Runs session_memory.py under the hood.",
        "parameters": {
            "summary": {"type": "string", "description": "Brief summary of what was accomplished this session"},
            "project": {"type": "string", "description": "Project name (optional)", "nullable": True},
            "goal": {"type": "string", "description": "Goal description (optional)", "nullable": True},
            "importance": {"type": "number", "description": "Importance 0.0-1.0 (default 0.85)", "nullable": True},
            "decisions": {"type": "array", "items": {"type": "string"}, "description": "Key decisions made this session", "nullable": True},
            "changes": {"type": "array", "items": {"type": "string"}, "description": "Code/config changes made this session", "nullable": True},
            "facts": {"type": "array", "items": {"type": "string"}, "description": "Important facts learned this session", "nullable": True},
        },
        "required": ["summary"],
    },
    "neural_memory_link": {
        "description": "Find semantically similar memories and create bidirectional wikilink connections. Run after storing a new memory to connect it to related context.",
        "parameters": {
            "memory_id": {"type": "string", "description": "Memory ID to find connections for"},
            "max_links": {"type": "integer", "description": "Maximum links to create (default 5)", "nullable": True},
            "threshold": {"type": "number", "description": "Similarity threshold 0.0-1.0 (default 0.6)", "nullable": True},
        },
        "required": ["memory_id"],
    },
    "neural_memory_get_profile": {
        "description": "Get the user's identity profile and preferences. Returns name, role, bio, learning goals, preferences, and learned preferences. Call at session start to personalize responses.",
        "parameters": {},
    },
    "neural_memory_observe_preference": {
        "description": "Log a user preference observation (e.g., they asked for code examples, or gave explicit instruction). Used by the AI to learn user preferences over time.",
        "parameters": {
            "type": {"type": "string", "description": "Observation type: code_style, answer_verbosity, explicit_preference, always_keyword, never_keyword"},
            "signal": {"type": "string", "description": "What the user said or did (e.g. 'show me the code', 'be more concise')", "nullable": True},
            "explicit": {"type": "boolean", "description": "Whether this was an explicit preference statement (default: false)", "nullable": True},
            "source": {"type": "string", "description": "Source context (e.g. 'session:ui-redesign')", "nullable": True},
        },
        "required": ["type"],
    },
    "neural_memory_log_skill": {
        "description": "Log that a skill was used during a task. The system uses this to learn which skills to auto-invoke for which task categories.",
        "parameters": {
            "skill_path": {"type": "string", "description": "File path to the skill file", "nullable": True},
            "skill_name": {"type": "string", "description": "Name of the skill used"},
            "task_category": {"type": "string", "description": "Category of the task (e.g. 'ui/ux design', 'backend', 'testing')"},
            "task_description": {"type": "string", "description": "Brief description of what the task was", "nullable": True},
        },
        "required": ["skill_name", "task_category"],
    },
    "neural_memory_consolidate_preferences": {
        "description": "Run preference consolidation: analyze all observations, update learned preferences, and regenerate the identity document. Call periodically to keep learned preferences up to date.",
        "parameters": {},
    },
    "neural_memory_log_session": {
        "description": "Log an entry to the active session in real time. Call after each meaningful exchange to prevent context loss. Types: message, decision, fact, change, fix.",
        "parameters": {
            "type": {"type": "string", "description": "Entry type: message, decision, fact, change, fix (default: message)", "nullable": True},
            "content": {"type": "string", "description": "The content of the log entry"},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags for this entry", "nullable": True},
            "importance": {"type": "number", "description": "Importance 0.0-1.0 (default 0.5)", "nullable": True},
        },
        "required": ["content"],
    },
    "neural_memory_dream": {
        "description": "Run the Dream Cycle — 22-phase overnight maintenance pipeline (cluster, merge, compress, prune, recalibrate, backup, etc.). Typically runs daily via systemd timer.",
        "parameters": {
            "max_memories": {"type": "integer", "description": "Maximum memories after pruning (default 5000)", "nullable": True},
            "fail_fast": {"type": "boolean", "description": "Stop on first phase failure (default false)", "nullable": True},
        },
    },
    "neural_memory_job_queue_status": {
        "description": "Get the job queue status and recent job history.",
        "parameters": {
            "limit": {"type": "integer", "description": "Max recent jobs to return (default 50)", "nullable": True},
        },
    },
    "neural_memory_job_enqueue": {
        "description": "Enqueue a new job for async background processing.",
        "parameters": {
            "job_type": {"type": "string", "description": "Job type name (e.g. 'sync-vault', 'compress', 'dream')"},
            "params": {"type": "object", "description": "Optional JSON params for the job", "nullable": True},
            "priority": {"type": "integer", "description": "Priority -10 to 10 (higher = more urgent, default 0)", "nullable": True},
        },
        "required": ["job_type"],
    },
    "neural_memory_job_stats": {
        "description": "Get job queue statistics (totals by status and type).",
        "parameters": {},
    },
}

DEFAULT_STORE_PATH = Path.home() / ".neural_memory" / "store.pkl"


def _ensure_lino_server():
    import subprocess
    import socket
    host = os.environ.get("LINO_HOST", "127.0.0.1")
    port = int(os.environ.get("LINO_PORT", "8210"))
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((host, port))
        sock.close()
        return
    except ConnectionRefusedError:
        pass
    finally:
        sock.close()
    script = os.path.join(os.path.dirname(__file__), "..", "bin", "lino-server.sh")
    script = os.path.abspath(script)
    if os.path.exists(script):
        subprocess.check_call(["bash", script])


class MemoryPlugin:
    def __init__(self, store_path: Optional[str] = None):
        _ensure_lino_server()
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
        idx = self.store.get_index_of(memory_id)
        if idx is None:
            return None
        meta = self.store.get_metadata(memory_id)
        if meta is None:
            return None
        emb = self.store.get_embedding(idx)
        text = meta.pop("text", "")
        return {
            "id": memory_id,
            "text": text,
            "embedding": emb.tolist() if emb is not None else None,
            "metadata": meta,
        }

    def _list_memories(self, limit: int = 50, offset: int = 0) -> tuple:
        ids = self.store.list_all()
        total = len(ids)
        page = ids[offset:offset + limit]
        results = []
        for mid in page:
            idx = self.store.get_index_of(mid)
            meta = self.store.get_metadata(mid) or {}
            emb = self.store.get_embedding(idx) if idx is not None else None
            text = meta.pop("text", "")
            results.append({
                "id": mid,
                "text": text,
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
        dim = self.store.get_dimension()
        return {
            "total_memories": total,
            "dimension": dim,
            "embedder_online": self.embedder_online,
            "store_path": str(self.store_path),
            "status": "online" if self.embedder_online else "degraded",
        }

    def cmd_update_priority(self, memory_id: str, priority: float) -> Dict[str, Any]:
        """Update a memory's importance/priority score in-place without re-embedding."""
        if self.store.get_index_of(memory_id) is None:
            return {"error": f"Memory not found: {memory_id}"}
        priority = max(0.0, min(1.0, float(priority)))
        self.store.update_metadata_value(memory_id, "importance_score", priority)
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

    def cmd_session_done(self, summary: str, project: str = "", goal: str = "",
                         importance: float = 0.85, decisions: Optional[List[str]] = None,
                         changes: Optional[List[str]] = None,
                         facts: Optional[List[str]] = None) -> Dict[str, Any]:
        """Call at session end. Stores a structured session summary."""
        # Close the active session first
        try:
            from app import memory_system
            memory_system.close_active_session(summary=summary)
        except Exception:
            pass

        now_ts = time.time()
        ts = datetime.fromtimestamp(now_ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        tags = ["session-summary"]
        if project:
            tags.append(f"project:{project}")
        stored_count = 0

        parts = [f"Timestamp: {ts}"]
        if project:
            parts.append(f"Project: {project}")
        if goal:
            parts.append(f"Goal: {goal}")
        parts.append(f"\n{summary}")

        if decisions:
            parts.append(f"\n### Decisions")
            for d in decisions:
                parts.append(f"- {d}")
        if changes:
            parts.append(f"\n### Changes")
            for c in changes:
                parts.append(f"- {c}")
        if facts:
            parts.append(f"\n### Facts")
            for f in facts:
                parts.append(f"- {f}")

        full_text = "\n".join(parts)
        source = f"session:{project}" if project else "session"

        memory_id = str(uuid.uuid4())
        metadata = {
            "text": full_text,
            "source": source,
            "importance_score": importance,
            "access_count": 0,
            "timestamp": now_ts,
            "tags": tags,
        }
        if self.embedder_online and self.embedder:
            emb = self.embedder.embed(full_text)
        else:
            emb = np.zeros(384).tolist()
        self.store.store(memory_id, emb, metadata)
        stored_count += 1

        if decisions:
            for d in decisions:
                did = str(uuid.uuid4())
                dmeta = {
                    "text": d,
                    "source": f"session-decision:{project}" if project else "session-decision",
                    "importance_score": max(importance - 0.05, 0.0),
                    "access_count": 0,
                    "timestamp": now_ts,
                    "tags": tags + ["decision"],
                    "related_memories": [memory_id],
                }
                if self.embedder_online and self.embedder:
                    demb = self.embedder.embed(d)
                else:
                    demb = np.zeros(384).tolist()
                self.store.store(did, demb, dmeta)
                stored_count += 1

        if facts:
            for f in facts:
                fid = str(uuid.uuid4())
                fmeta = {
                    "text": f,
                    "source": f"session-fact:{project}" if project else "session-fact",
                    "importance_score": max(importance - 0.1, 0.0),
                    "access_count": 0,
                    "timestamp": now_ts,
                    "tags": tags + ["fact"],
                    "related_memories": [memory_id],
                }
                if self.embedder_online and self.embedder:
                    femb = self.embedder.embed(f)
                else:
                    femb = np.zeros(384).tolist()
                self.store.store(fid, femb, fmeta)
                stored_count += 1

        self._save_store()

        # Auto-link the summary with existing memories
        link_result = self._link_memory(memory_id, max_links=5, threshold=0.6)

        # Auto-consolidate preferences
        try:
            consolidation = memory_system.consolidate_preferences()
        except Exception:
            consolidation = {"skipped": True}

        return {
            "status": "stored",
            "memory_id": memory_id,
            "stored_count": stored_count,
            "links_created": link_result["links_created"],
            "importance": importance,
            "project": project or "",
            "timestamp_utc": ts,
            "consolidation": consolidation,
        }

    def _extract_text(self, memory_id: str) -> Optional[str]:
        meta = self.store.get_metadata(memory_id)
        if meta is None:
            return None
        return meta.get("text", "")

    def _link_memory(self, memory_id: str, max_links: int = 5,
                     threshold: float = 0.6) -> Dict[str, Any]:
        """Find semantically similar memories and create bidirectional links."""
        target_text = self._extract_text(memory_id)
        if not target_text:
            return {"status": "skipped", "reason": "memory not found", "links_created": 0}

        if not self.retriever:
            self._init_retriever()
        if not self.retriever:
            return {"status": "skipped", "reason": "retriever unavailable", "links_created": 0}

        results = self.retriever.retrieve(target_text, k=max_links + 1)
        linked = []
        for r in results:
            mid = r.get("id", "")
            if mid == memory_id:
                continue
            score = r.get("score", 0)
            if score < threshold:
                continue
            existing = self.store.get_metadata(memory_id)
            if existing is None:
                continue
            related = existing.get("related_memories", []) or []
            if mid not in related and f"[[{mid}]]" not in related:
                related.append(mid)
                self.store.update_metadata_value(memory_id, "related_memories", related)
            other_meta = self.store.get_metadata(mid)
            if other_meta is not None:
                other_related = other_meta.get("related_memories", []) or []
                if memory_id not in other_related and f"[[{memory_id}]]" not in other_related:
                    other_related.append(memory_id)
                    self.store.update_metadata_value(mid, "related_memories", other_related)
            linked.append({"memory_id": mid, "similarity": score})

        self._save_store()
        return {"status": "linked", "links_created": len(linked), "links": linked}

    def cmd_link(self, memory_id: str, max_links: int = 5,
                 threshold: float = 0.6) -> Dict[str, Any]:
        """Find semantically similar memories and create bidirectional links."""
        return self._link_memory(memory_id, max_links, threshold)

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

    def _load_observations(self) -> list:
        p = Path.home() / ".neural_memory" / "preference_observations.json"
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                return []
        return []

    def _save_observations(self, obs: list):
        p = Path.home() / ".neural_memory" / "preference_observations.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        if len(obs) > 10000:
            obs = obs[-10000:]
        p.write_text(json.dumps(obs, indent=2))

    def cmd_observe_preference(self, type: str, signal: str = "", explicit: bool = False, source: str = "") -> Dict[str, Any]:
        obs = self._load_observations()
        obs.append({
            "id": str(uuid.uuid4()),
            "type": type,
            "signal": signal,
            "explicit": explicit,
            "source": source,
            "timestamp": time.time(),
        })
        self._save_observations(obs)
        return {"observed": True, "type": type, "total": len(obs)}

    def cmd_log_skill(self, skill_name: str, task_category: str, skill_path: str = "", task_description: str = "") -> Dict[str, Any]:
        obs = self._load_observations()
        obs.append({
            "id": str(uuid.uuid4()),
            "type": "skill_usage",
            "skill_path": skill_path,
            "skill_name": skill_name,
            "task_category": task_category,
            "task_description": task_description,
            "explicit": False,
            "timestamp": time.time(),
        })
        self._save_observations(obs)
        return {"logged": True, "skill": skill_name, "category": task_category, "total": len(obs)}

    def cmd_consolidate_preferences(self) -> Dict[str, Any]:
        from app import memory_system
        result = memory_system.consolidate_preferences()
        return result

    def cmd_log_session(self, type: str = "message", content: str = "", tags: Optional[List[str]] = None, importance: float = 0.5) -> Dict[str, Any]:
        """Log an entry to the active session in real time."""
        try:
            from app import memory_system
            result = memory_system.append_to_session(
                entry_type=type,
                content=content,
                tags=tags,
                importance=importance,
            )
            return result
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def cmd_dream(self, max_memories: int = 5000, fail_fast: bool = False) -> Dict[str, Any]:
        from scripts.dream_cycle import run_dream_cycle
        try:
            result = run_dream_cycle({"max_memories": max_memories, "fail_fast": fail_fast})
            errors = [n for n, p in result.get("phases", {}).items() if p.get("status") == "error"]
            return {
                "status": "ok" if not errors else "partial",
                "total_elapsed": result.get("total_elapsed", 0),
                "phases_ok": sum(1 for p in result.get("phases", {}).values() if p.get("status") == "ok"),
                "phases_error": len(errors),
                "error_phases": errors,
                "report_path": result.get("phases", {}).get("dream_report", {}).get("result", {}).get("report_path"),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def cmd_job_queue_status(self, limit: int = 50) -> Dict[str, Any]:
        from scripts.job_queue import job_queue
        try:
            return job_queue.status(limit=limit)
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def cmd_job_enqueue(self, job_type: str, params: Optional[Dict] = None, priority: int = 0) -> Dict[str, Any]:
        from scripts.job_queue import job_queue
        try:
            job = job_queue.enqueue(job_type=job_type, params=params or {}, priority=priority)
            return {"status": "enqueued", "job": job.to_dict()}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def cmd_job_stats(self) -> Dict[str, Any]:
        from scripts.job_queue import job_queue
        try:
            return job_queue.stats()
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def cmd_get_profile(self) -> Dict[str, Any]:
        """Return the user's identity profile and preferences for Hermes to read at session start."""
        try:
            PROC_UUID = str(uuid.uuid5(uuid.NAMESPACE_DNS, "lino-profile"))
            mem = self._get_memory(PROC_UUID)
            if not mem:
                return {"status": "no_profile", "message": "No profile exists yet. Create one via the Lino UI."}
            meta = mem.get("metadata", {}) or {}
            profile_data = meta.get("profile_data", {})
            related = meta.get("related_memories", []) or []
            linked_entities = []
            for rid in related:
                rm = self._get_memory(rid)
                if rm:
                    rmeta = rm.get("metadata", {}) or {}
                    rtags = rmeta.get("tags", []) or []
                    if "type:preferences" in rtags:
                        profile_data["preferences"] = rmeta.get("preference_data", {})
                    if "type:entity" in rtags:
                        linked_entities.append({
                            "id": rid,
                            "text": rmeta.get("text", ""),
                            "tags": rtags,
                        })
            # Include learned preferences
            id_path = Path.home() / ".neural_memory" / "lino-identity.md"
            if id_path.exists():
                profile_data["identity_doc"] = id_path.read_text()
            return {
                "status": "ok",
                "profile": profile_data,
                "linked_entities": linked_entities,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
