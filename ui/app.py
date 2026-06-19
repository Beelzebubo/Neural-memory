import json
import logging
import os
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml
from fastapi import FastAPI, Form, HTTPException, Query, Request, UploadFile
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src import TextEmbedder, VectorMemoryStore, MemoryRetriever, MemoryConsolidator

# ── Load .env ──
ENV_PATH = Path(__file__).parent.parent / ".env"
if ENV_PATH.exists():
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"
DEFAULT_STORE_PATH = Path.home() / ".neural_memory" / "store.pkl"

@asynccontextmanager
async def lifespan(app):
    if memory_system.store and memory_system.embedder:
        try:
            result = memory_system.batch_relink(skip_semantic=False)
            logger.info(f"Auto-relink on startup: {result['changed']} memories updated")
        except Exception:
            logger.exception("Auto-relink failed")
    yield

app = FastAPI(title="Neural Memory — Lino", lifespan=lifespan)

# ── Profile constants ──
PROFILE_UUID = str(uuid.uuid5(uuid.NAMESPACE_DNS, "lino-profile"))
PREFERENCES_UUID = str(uuid.uuid5(uuid.NAMESPACE_DNS, "lino-preferences"))
OBSERVATIONS_PATH = Path.home() / ".neural_memory" / "preference_observations.json"
IDENTITY_PATH = Path.home() / ".neural_memory" / "lino-identity.md"

# Thresholds for learned preferences
OBSERVATION_THRESHOLDS = {
    "code_style": 5,
    "answer_verbosity": 5,
    "skill_usage": 3,
    "explicit_preference": 1,
    "always_keyword": 1,
    "never_keyword": 1,
}

# ── Rate limiter ──
limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── API Key from environment ──
LINO_API_KEY = os.environ.get("LINO_API_KEY", "")

# ── CORS origins from environment ──
raw_origins = os.environ.get("LINO_CORS_ORIGINS", "http://127.0.0.1:8210")
CORS_ORIGINS = [o.strip() for o in raw_origins.split(",") if o.strip()]


# ── Security Headers Middleware ──

from starlette.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    # API key check for non-GET, non-UI routes
    if LINO_API_KEY and request.method not in ("GET", "OPTIONS"):
        path = request.url.path
        if path.startswith("/api/") and not path.startswith("/api/stats"):
            auth = request.headers.get("X-API-Key", "")
            if auth != LINO_API_KEY:
                return JSONResponse(status_code=401, content={"error": "Unauthorized — missing or invalid X-API-Key"})

    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' https://unpkg.com https://d3js.org https://cdn.jsdelivr.net; "
        "style-src 'self' https://fonts.googleapis.com https://unpkg.com 'unsafe-inline'; "
        "font-src 'self' https://fonts.gstatic.com https://unpkg.com; "
        "img-src 'self' data: blob:; "
        "connect-src 'self'; "
        "worker-src 'self' blob:; "
        "frame-src 'none'"
    )
    return response

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
app.mount("/ui/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


class MemorySystem:
    def __init__(self):
        self.store = VectorMemoryStore()
        self.embedder = None
        self.retriever = None
        self.consolidator = MemoryConsolidator()
        self.config = self._load_config()
        self.store_path = Path(os.environ.get("NEURAL_MEMORY_PATH", DEFAULT_STORE_PATH))
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_embedder()
        self._load_store()
        self._init_retriever()

    def _load_config(self) -> dict:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH) as f:
                return yaml.safe_load(f) or {}
        return {}

    def _init_embedder(self):
        try:
            cfg = self.config.get("embedder", {})
            model = cfg.get("model_name", "all-MiniLM-L6-v2")
            self.embedder = TextEmbedder(model_name=model)
        except Exception:
            self.embedder = None

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

    def save_store(self):
        self.store.save(str(self.store_path))

    def get_memory_by_id(self, memory_id: str) -> Optional[dict]:
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

    def list_memories(self, limit: int = 50, offset: int = 0, reverse: bool = False, sort: str = "newest") -> tuple:
        all_ids = list(self.store.list_all())
        total = len(all_ids)

        if sort == "importance":
            scored = []
            for mid in all_ids:
                meta = self.store.get_metadata(mid) or {}
                imp = meta.get("importance_score", 0.5)
                scored.append((imp, mid))
            scored.sort(key=lambda x: -x[0])
            all_ids = [mid for _, mid in scored]
        elif reverse:
            all_ids.reverse()

        page = all_ids[offset:offset + limit]
        results = []
        for mid in page:
            meta = self.store.get_metadata(mid) or {}
            text = meta.pop("text", "")
            results.append({
                "id": mid,
                "text": text,
                "metadata": meta,
            })
        return results, total

    def _ensure_profile_node(self) -> Optional[str]:
        for mid in self.store.list_all():
            meta = self.store.get_metadata(mid) or {}
            tags = meta.get("tags", []) or []
            if "type:profile" in tags:
                return mid
        emb = [0.0] * 384
        meta = {
            "text": "Your Identity",
            "source": "system",
            "importance_score": 1.0,
            "access_count": 0,
            "timestamp": time.time(),
            "tags": ["type:profile", "protected"],
            "protected": True,
            "related_memories": [],
            "profile_data": {
                "name": "",
                "role": "",
                "bio": "",
                "learning_goals": [],
                "preferences": {
                    "answer_style": "concise",
                    "code_examples": "when_relevant",
                    "skills_for_tasks": {},
                    "extra": {},
                },
            },
        }
        self.store.store(PROFILE_UUID, emb, meta)
        return PROFILE_UUID

    def _ensure_entity_node(self, name: str, entity_type: str, importance: float = 0.8) -> str:
        for mid in self.store.list_all():
            meta = self.store.get_metadata(mid) or {}
            tags = meta.get("tags", []) or []
            if f"type:entity" in tags and f"entity_type:{entity_type}" in tags and meta.get(f"entity_{entity_type}_name", "").lower() == name.lower():
                return mid
        eid = str(uuid.uuid4())
        emb = self.embedder.embed(name) if self.embedder else np.zeros(384).tolist()
        meta = {
            "text": f"{entity_type.title()}: {name}",
            "source": "system",
            "importance_score": importance,
            "access_count": 0,
            "timestamp": time.time(),
            "tags": ["type:entity", f"entity_type:{entity_type}", f"entity_{entity_type}_name:{name}", "protected"],
            "protected": True,
            "related_memories": [],
        }
        self.store.store(eid, emb, meta)
        return eid

    def _ensure_preferences_node(self) -> str:
        for mid in self.store.list_all():
            meta = self.store.get_metadata(mid) or {}
            tags = meta.get("tags", []) or []
            if "type:preferences" in tags:
                return mid
        emb = [0.0] * 384
        meta = {
            "text": "User Preferences",
            "source": "system",
            "importance_score": 0.8,
            "access_count": 0,
            "timestamp": time.time(),
            "tags": ["type:preferences", "protected"],
            "protected": True,
            "related_memories": [],
            "preference_data": {
                "answer_style": "concise",
                "code_examples": "when_relevant",
                "skills_for_tasks": {},
                "extra": {},
            },
        }
        self.store.store(PREFERENCES_UUID, emb, meta)
        return PREFERENCES_UUID

    def _link_memories(self, a_id: str, b_id: str, link_type: str = "related"):
        for mid in (a_id, b_id):
            if mid not in self.store._metadata:
                self.store._metadata[mid] = {}
            meta = self.store._metadata[mid]
            related = meta.get("related_memories", [])
            if isinstance(related, list):
                related = list(related)
            else:
                related = []
            other = b_id if mid == a_id else a_id
            if other not in related:
                related.append(other)
                meta["related_memories"] = related
            related_types = meta.get("related_types", {})
            if not isinstance(related_types, dict):
                related_types = {}
            if other not in related_types:
                related_types[other] = link_type
            meta["related_types"] = related_types

    WIKILINK_RE = re.compile(r'\[\[([^\]|]+)(?:\|([^\]|]+))?\]\]')

    TYPED_LINK_RE = re.compile(
        r'(?:(?P<type>works at|employed by|part of|member of|founded|created|related to|connected to|mentions)\s+)?'
        r'\[\[(?P<target>[^\]|]+)(?:\|[^\]|]+)?\]\]',
        re.IGNORECASE
    )

    def _extract_wikilinks(self, text: str) -> List[str]:
        linked = []
        for match in self.WIKILINK_RE.finditer(text):
            target = (match.group(2) or match.group(1)).strip().lower()
            for mid in self.store.list_all():
                meta = self.store.get_metadata(mid) or {}
                mem_text = (meta.get("text", "") or "").lower()
                if target in mem_text:
                    linked.append(mid)
                    break
        return linked

    def _extract_typed_links(self, text: str) -> List[tuple]:
        """Extract typed wikilinks. Returns list of (memory_id, link_type)."""
        results = []
        for match in self.TYPED_LINK_RE.finditer(text):
            rel_type = match.group("type")
            if not rel_type:
                continue  # bare wikilink without type prefix → handled by _extract_wikilinks
            rel_type = rel_type.strip().lower().replace(" ", "_")
            target = match.group("target").strip().lower()
            for mid in self.store.list_all():
                meta = self.store.get_metadata(mid) or {}
                mem_text = (meta.get("text", "") or "").lower()
                if target in mem_text:
                    results.append((mid, rel_type))
                    break
        return results

    # ── Session Management ──

    def _ensure_active_session(self) -> str:
        for mid in self.store.list_all():
            meta = self.store.get_metadata(mid) or {}
            tags = meta.get("tags", []) or []
            sd = meta.get("session_data", {}) or {}
            if "type:session" in tags and sd.get("status") == "active":
                return mid
        session_id = str(uuid.uuid4())
        emb = [0.0] * 384
        now = time.time()
        now_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(now))
        meta = {
            "text": f"# Session — {now_str}",
            "source": "system",
            "importance_score": 0.9,
            "access_count": 0,
            "timestamp": now,
            "tags": ["type:session", "protected"],
            "protected": True,
            "related_memories": [],
            "related_types": {},
            "session_data": {
                "entries": [],
                "started_at": now,
                "updated_at": now,
                "status": "active",
                "summary": "",
            },
        }
        self.store.store(session_id, emb, meta)
        return session_id

    def _rebuild_session_text(self, sd: dict) -> str:
        started = sd.get("started_at", time.time())
        header = f"# Session — {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(started))}"
        lines = [header, ""]
        for entry in sd.get("entries", []):
            ts = entry.get("ts", started)
            ts_str = time.strftime("%H:%M:%S", time.gmtime(ts))
            etype = entry.get("type", "message")
            content = entry.get("content", "")
            lines.append(f"## {ts_str} | {etype}")
            lines.append(content)
            lines.append("")
        return "\n".join(lines).strip()

    def append_to_session(self, entry_type: str, content: str, tags: Optional[List[str]] = None, importance: float = 0.5) -> dict:
        sid = self._ensure_active_session()
        meta = self.store.get_metadata(sid) or {}
        sd = dict(meta.get("session_data", {}) or {})
        entries = list(sd.get("entries", []))
        now = time.time()
        entries.append({
            "ts": now,
            "type": entry_type,
            "content": content,
            "importance": importance,
            "tags": tags or [],
        })
        sd["entries"] = entries
        sd["updated_at"] = now
        meta["session_data"] = sd
        meta["text"] = self._rebuild_session_text(sd)
        meta["importance_score"] = max(meta.get("importance_score", 0.5), importance)
        if sid in self.store._metadata:
            self.store._metadata[sid] = meta
        self.store.update_metadata_value(sid, "session_data", sd)
        self.store.update_metadata_value(sid, "text", meta["text"])
        self.store.update_metadata_value(sid, "importance_score", meta["importance_score"])
        if self.embedder:
            new_emb = self.embedder.embed(meta["text"])
            idx = self.store.get_index_of(sid)
            if idx is not None:
                self.store.set_embedding(idx, np.array(new_emb, dtype=np.float32))
        self.save_store()
        return {"status": "logged", "session_id": sid, "entry_count": len(entries)}

    def get_active_session(self) -> Optional[dict]:
        sid = self._ensure_active_session()
        mem = self.get_memory_by_id(sid)
        if not mem:
            return None
        meta = mem.get("metadata", {})
        sd = meta.get("session_data", {}) or {}
        return {
            "id": sid,
            "text": meta.get("text", ""),
            "session_data": sd,
            "entry_count": len(sd.get("entries", [])),
        }

    def close_active_session(self, summary: str = ""):
        sid = None
        for mid in self.store.list_all():
            meta = self.store.get_metadata(mid) or {}
            tags = meta.get("tags", []) or []
            sd = meta.get("session_data", {}) or {}
            if "type:session" in tags and sd.get("status") == "active":
                sid = mid
                break
        if not sid:
            return {"status": "no_active_session"}
        meta = self.store.get_metadata(sid) or {}
        sd = dict(meta.get("session_data", {}) or {})
        sd["status"] = "closed"
        sd["ended_at"] = time.time()
        sd["summary"] = summary or sd.get("summary", "")
        meta["session_data"] = sd
        meta["protected"] = False
        tags = meta.get("tags", []) or []
        tags = [t for t in tags if t != "protected"]
        if "session-summary" not in tags:
            tags.append("session-summary")
        meta["tags"] = tags
        meta["text"] = self._rebuild_session_text(sd) + (f"\n\n**Summary:** {summary}" if summary else "")
        if sid in self.store._metadata:
            self.store._metadata[sid] = meta
        self.store.update_metadata_value(sid, "session_data", sd)
        self.store.update_metadata_value(sid, "protected", False)
        self.store.update_metadata_value(sid, "tags", tags)
        self.store.update_metadata_value(sid, "text", meta["text"])
        if self.embedder:
            new_emb = self.embedder.embed(meta["text"])
            idx = self.store.get_index_of(sid)
            if idx is not None:
                self.store.set_embedding(idx, np.array(new_emb, dtype=np.float32))
        self.save_store()
        return {"status": "closed", "session_id": sid, "entry_count": len(sd.get("entries", [])), "summary": summary}

    def list_sessions(self, limit: int = 10) -> List[dict]:
        sessions = []
        for mid in self.store.list_all():
            meta = self.store.get_metadata(mid) or {}
            tags = meta.get("tags", []) or []
            if "type:session" not in tags:
                continue
            sd = meta.get("session_data", {}) or {}
            sessions.append({
                "id": mid,
                "text": (meta.get("text", "") or "")[:120],
                "status": sd.get("status", "unknown"),
                "started_at": sd.get("started_at", 0),
                "ended_at": sd.get("ended_at"),
                "entry_count": len(sd.get("entries", [])),
                "summary": sd.get("summary", ""),
            })
        sessions.sort(key=lambda s: s.get("started_at", 0), reverse=True)
        return sessions[:limit]

    # ── Knowledge Graph Traversal ──

    def traverse_graph(self, start_id: str, depth: int = 3, types: Optional[List[str]] = None) -> dict:
        if self.store.get_index_of(start_id) is None:
            return {"error": "Start node not found", "levels": []}
        visited = set()
        levels = []
        current = [(start_id, "self", 0)]
        for d in range(depth):
            if not current:
                break
            level_nodes = []
            level_edges = []
            next_level = []
            for node_id, edge_type, node_depth in current:
                if node_id in visited:
                    continue
                visited.add(node_id)
                meta = self.store.get_metadata(node_id) or {}
                text = meta.get("text", "") or ""
                tags = meta.get("tags", []) or []
                importance = meta.get("importance_score", 0.5)
                level_nodes.append({
                    "id": node_id,
                    "text": text[:80],
                    "importance": importance,
                    "tags": tags,
                })
                related = meta.get("related_memories", []) or []
                related_types = meta.get("related_types", {}) or {}
                for rid in related:
                    if rid in visited:
                        continue
                    link_type = related_types.get(rid, "related")
                    if types and link_type not in types:
                        continue
                    level_edges.append({"source": node_id, "target": rid, "type": link_type})
                    next_level.append((rid, link_type, d + 1))
            levels.append({"depth": d, "nodes": level_nodes, "edges": level_edges})
            current = next_level
        return {"levels": levels, "total_nodes": len(visited), "start_id": start_id, "depth": depth}

    def get_backlinks(self, memory_id: str) -> List[dict]:
        """Return all memories that link to the given memory_id."""
        backlinks = []
        for mid in self.store.list_all():
            if mid == memory_id:
                continue
            meta = self.store.get_metadata(mid) or {}
            related = meta.get("related_memories", []) or []
            if memory_id in related:
                related_types = meta.get("related_types", {}) or {}
                backlinks.append({
                    "id": mid,
                    "text": (meta.get("text", "") or "")[:120],
                    "type": related_types.get(memory_id, "related"),
                })
        return backlinks

    # ── Preference Learning ──

    def _obs_path(self) -> Path:
        return OBSERVATIONS_PATH

    def _load_observations(self) -> list:
        p = self._obs_path()
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                return []
        return []

    def _save_observations(self, obs: list):
        p = self._obs_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        if len(obs) > 10000:
            obs = obs[-10000:]
        p.write_text(json.dumps(obs, indent=2))

    def observe_preference(self, pref_type: str, signal: str = "", explicit: bool = False, source: str = ""):
        obs = self._load_observations()
        obs.append({
            "id": str(uuid.uuid4()),
            "type": pref_type,
            "signal": signal,
            "explicit": explicit,
            "source": source,
            "timestamp": time.time(),
        })
        self._save_observations(obs)
        return {"observed": True, "type": pref_type, "total": len(obs)}

    def log_skill_usage(self, skill_path: str, skill_name: str, task_category: str, task_description: str = ""):
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

    def get_learned_preferences(self) -> dict:
        mid = self._ensure_profile_node()
        meta = self.store.get_metadata(mid) or {}
        pd = meta.get("profile_data", {}) or {}
        return pd.get("learned_preferences", {})

    def consolidate_preferences(self) -> dict:
        obs = self._load_observations()
        thresholds = OBSERVATION_THRESHOLDS
        counts = {}
        skill_counts = {}

        for o in obs:
            t = o.get("type", "")
            if t == "skill_usage":
                key = (o.get("skill_name", ""), o.get("task_category", ""))
                skill_counts[key] = skill_counts.get(key, 0) + 1
            elif t in ("always_keyword", "never_keyword"):
                sig = o.get("signal", "").lower()
                counts[t] = {"count": counts.get(t, {}).get("count", 0) + 1, "signal": sig}
            else:
                counts[t] = counts.get(t, 0) + 1

        learned = {}

        # code_style: repeated request for code examples
        if counts.get("code_style", 0) >= thresholds["code_style"]:
            learned["code_examples"] = {"value": "always", "count": counts["code_style"], "confidence": min(counts["code_style"] / 10, 1.0)}

        # answer_verbosity: repeated detailed answers
        if counts.get("answer_verbosity", 0) >= thresholds["answer_verbosity"]:
            learned["answer_verbosity"] = {"value": "detailed_with_code", "count": counts["answer_verbosity"], "confidence": min(counts["answer_verbosity"] / 10, 1.0)}

        # always/never keywords — single occurrence triggers
        if counts.get("always_keyword", {}).get("count", 0) >= thresholds.get("always_keyword", 1):
            sig = counts["always_keyword"].get("signal", "")
            if "code" in sig or "example" in sig:
                learned["code_examples"] = {"value": "always", "count": counts["always_keyword"]["count"], "confidence": 0.9, "source": "explicit"}
            if "detail" in sig or "thorough" in sig:
                learned["answer_verbosity"] = {"value": "detailed_with_code", "count": counts["always_keyword"]["count"], "confidence": 0.9, "source": "explicit"}
        if counts.get("never_keyword", {}).get("count", 0) >= thresholds.get("never_keyword", 1):
            sig = counts["never_keyword"].get("signal", "")
            if "code" in sig:
                learned["code_examples"] = {"value": "never", "count": counts["never_keyword"]["count"], "confidence": 0.9, "source": "explicit"}

        # skill_usage: same skill for same task category >= 3
        skills_for_tasks = {}
        for (skill_name, task_cat), cnt in skill_counts.items():
            if cnt >= thresholds["skill_usage"]:
                if task_cat not in skills_for_tasks:
                    skills_for_tasks[task_cat] = {"skills": [], "count": 0, "confidence": 0.0}
                skills_for_tasks[task_cat]["skills"].append(skill_name)
                skills_for_tasks[task_cat]["count"] = cnt
                skills_for_tasks[task_cat]["confidence"] = min(cnt / 6, 1.0)

        if skills_for_tasks:
            learned["skills_for_tasks"] = skills_for_tasks

        # Merge into profile
        mid = self._ensure_profile_node()
        meta = self.store.get_metadata(mid) or {}
        pd = dict(meta.get("profile_data", {}))
        existing_learned = pd.get("learned_preferences", {})

        locked = set(pd.get("locked_preferences", []))

        # Only set learned prefs that aren't locked by manual override
        for key, value in learned.items():
            if key not in locked:
                existing_learned[key] = value

        pd["learned_preferences"] = existing_learned

        # Merge learned into effective preferences (respect locks)
        effective = dict(pd.get("preferences", {}))
        skill_map = effective.get("skills_for_tasks", {})
        for cat, info in existing_learned.get("skills_for_tasks", {}).items():
            if cat not in skill_map:
                skill_map[cat] = info["skills"]
        if skill_map:
            effective["skills_for_tasks"] = skill_map

        for key, info in existing_learned.items():
            if key == "skills_for_tasks":
                continue
            if key not in locked and key in ("code_examples", "answer_verbosity"):
                effective[key] = info.get("value", effective.get(key, "when_relevant"))

        pd["preferences"] = effective
        pd["learned_preferences"] = existing_learned

        meta["profile_data"] = pd
        if mid in self.store._metadata:
            self.store._metadata[mid]["profile_data"] = pd
        self.save_store()

        self._generate_identity_doc()

        return {
            "learned": learned,
            "effective_preferences": effective,
            "observation_counts": {k: v for k, v in counts.items()},
        }

    def _generate_identity_doc(self):
        mid = self._ensure_profile_node()
        meta = self.store.get_metadata(mid) or {}
        pd = dict(meta.get("profile_data", {}))
        effective = pd.get("preferences", {})
        learned = pd.get("learned_preferences", {})

        name = pd.get("name", "User")
        role = pd.get("role", "")
        bio = pd.get("bio", "")
        goals = pd.get("learning_goals", [])

        lines = []
        lines.append(f"# Lino Identity — {name}")
        lines.append("")
        if role:
            lines.append(f"**Role:** {role}")
        if bio:
            lines.append(f"**Bio:** {bio}")
        lines.append("")

        if goals:
            lines.append("## Learning Goals")
            for g in goals:
                lines.append(f"- {g}")
            lines.append("")

        lines.append("## Active Preferences")
        ans = effective.get("answer_style", "concise")
        lines.append(f"- **Answer Style:** {ans}")
        code = effective.get("code_examples", "when_relevant")
        code_src = ""
        lc = learned.get("code_examples", {})
        if lc and "source" in lc:
            code_src = f" (learned, {lc.get('confidence', 0):.0%} confidence, from {lc.get('count', 0)} observations)"
        elif lc:
            code_src = f" (learned, {lc.get('confidence', 0):.0%} confidence)"
        lines.append(f"- **Code Examples:** {code}{code_src}")

        extra = effective.get("extra", {})
        if extra:
            lines.append("")
            lines.append("### Extra Preferences")
            for k, v in extra.items():
                lines.append(f"- **{k}:** {v}")
        lines.append("")

        # Skills
        skills = effective.get("skills_for_tasks", {})
        if skills:
            lines.append("## Skill Mappings")
            for cat, skill_list in skills.items():
                s_info = learned.get("skills_for_tasks", {}).get(cat, {})
                cnt_info = f" ({s_info.get('count', '?')} uses, auto-invoke)" if s_info else " (manual)"
                skill_str = ", ".join(skill_list) if isinstance(skill_list, list) else skill_list
                lines.append(f"- **{cat}** → {skill_str}{cnt_info}")
            lines.append("")

        # Active projects from graph
        projects = []
        for mid2 in self.store.list_all():
            mmeta = self.store.get_metadata(mid2) or {}
            tags = mmeta.get("tags", []) or []
            for t in tags:
                if t.startswith("project:"):
                    pname = t.split(":", 1)[1].strip()
                    if pname and pname not in projects:
                        projects.append(pname)
        if projects:
            lines.append("## Active Projects")
            for p in projects:
                lines.append(f"- {p}")
            lines.append("")

        # Key people
        people = []
        for mid2 in self.store.list_all():
            mmeta = self.store.get_metadata(mid2) or {}
            tags = mmeta.get("tags", []) or []
            for t in tags:
                if t.startswith("person:"):
                    pname = t.split(":", 1)[1].strip()
                    if pname and pname not in people:
                        people.append(pname)
        if people:
            lines.append("## Key People")
            for p in people:
                lines.append(f"- {p}")
            lines.append("")

        # Learned preferences history
        if learned:
            lines.append("## Learned Preferences History")
            for key, info in learned.items():
                if key == "skills_for_tasks":
                    for cat, sinfo in info.items():
                        lines.append(f"- **skills_for_tasks/{cat}** → {', '.join(sinfo['skills'])} ({sinfo['count']} observations, {sinfo['confidence']:.0%} confidence)")
                else:
                    val = info.get("value", "?")
                    cnt = info.get("count", "?")
                    conf = info.get("confidence", 0)
                    src = f", {info['source']}" if "source" in info else ""
                    lines.append(f"- **{key}** → {val} ({cnt} observations, {conf:.0%} confidence{src})")

        content = "\n".join(lines) + "\n"
        IDENTITY_PATH.parent.mkdir(parents=True, exist_ok=True)
        IDENTITY_PATH.write_text(content)
        return content

    def create_memory(self, text: str, metadata: dict) -> dict:
        memory_id = str(uuid.uuid4())
        metadata_clean = {
            "text": text,
            "source": metadata.get("source", "manual"),
            "importance_score": metadata.get("importance_score", 0.5),
            "access_count": 0,
            "timestamp": time.time(),
        }
        tags = metadata.get("tags", []) or []
        if "tags" in metadata:
            metadata_clean["tags"] = tags
        if self.embedder:
            emb = self.embedder.embed(text)
        else:
            emb = np.zeros(384).tolist()
        self.store.store(memory_id, emb, metadata_clean)
        related = []
        profile_mid = self._ensure_profile_node()
        if profile_mid:
            self._link_memories(profile_mid, memory_id)
            related.append(profile_mid)
        # Create/link entity nodes for project:* and person:* tags
        for tag in tags:
            if tag.startswith("project:"):
                pname = tag.split(":", 1)[1].strip()
                if pname:
                    eid = self._ensure_entity_node(pname, "project", importance=0.9)
                    self._link_memories(eid, memory_id)
                    if profile_mid:
                        self._link_memories(profile_mid, eid)
                    related.append(eid)
            if tag.startswith("person:"):
                pname = tag.split(":", 1)[1].strip()
                if pname:
                    eid = self._ensure_entity_node(pname, "person", importance=0.8)
                    self._link_memories(eid, memory_id)
                    if profile_mid:
                        self._link_memories(profile_mid, eid)
                    related.append(eid)
        # Typed wikilink resolution: "works at [[X]]" → link with specific type (run BEFORE auto-linking to claim types)
        typed_ids = set()
        for wlid, link_type in self._extract_typed_links(text):
            if wlid != memory_id:
                self._link_memories(memory_id, wlid, link_type)
                if wlid not in related:
                    related.append(wlid)
                typed_ids.add(wlid)
        # Bare wikilink resolution: [[memory text]] → link (type:related), skip if already typed
        for wlid in self._extract_wikilinks(text):
            if wlid in typed_ids:
                continue
            if wlid not in related and wlid != memory_id:
                self._link_memories(memory_id, wlid)
                related.append(wlid)
        # Semantic auto-linking: find top-5 similar memories and connect (won't overwrite typed link types)
        if self.retriever and len(text) > 10:
            try:
                similar = self.retriever.retrieve(text, k=6, min_score=0.25)
                for r in similar:
                    sid = r.get("id", "")
                    if sid and sid != memory_id and sid not in related:
                        self._link_memories(memory_id, sid)
                        related.append(sid)
            except Exception:
                pass

        if related:
            if memory_id not in self.store._metadata:
                self.store._metadata[memory_id] = {}
            self.store._metadata[memory_id]["related_memories"] = related
        self.save_store()
        metadata_clean["related_memories"] = related
        return {
            "id": memory_id,
            "text": text,
            "metadata": metadata_clean,
        }

    def batch_relink(self, skip_semantic: bool = False) -> dict:
        all_ids = self.store.list_all()
        total = len(all_ids)
        changed = 0
        for memory_id in all_ids:
            meta = self.store.get_metadata(memory_id) or {}
            text = (meta.get("text", "") or "")
            if not text:
                continue
            existing_related = set(meta.get("related_memories", []) or [])
            linked = set(existing_related)
            typed_ids = set()
            for wlid, link_type in self._extract_typed_links(text):
                if wlid != memory_id:
                    self._link_memories(memory_id, wlid, link_type)
                    linked.add(wlid)
                    typed_ids.add(wlid)
            for wlid in self._extract_wikilinks(text):
                if wlid in typed_ids or wlid == memory_id:
                    continue
                self._link_memories(memory_id, wlid)
                linked.add(wlid)
            if not skip_semantic and self.retriever and len(text) > 10:
                try:
                    similar = self.retriever.retrieve(text, k=6, min_score=0.25)
                    for r in similar:
                        sid = r.get("id", "")
                        if sid and sid != memory_id and sid not in linked:
                            self._link_memories(memory_id, sid)
                            linked.add(sid)
                except Exception:
                    pass
            if len(linked) > len(existing_related):
                changed += 1
        if changed:
            self.save_store()
        return {"total": total, "changed": changed}

    def update_memory(self, memory_id: str, updates: dict) -> Optional[dict]:
        idx = self.store.get_index_of(memory_id)
        if idx is None:
            return None
        meta = self.store.get_metadata(memory_id) or {}
        if "text" in updates:
            meta["text"] = updates["text"]
            if self.embedder:
                new_emb = self.embedder.embed(updates["text"])
                self.store.set_embedding(idx, np.array(new_emb, dtype=np.float32))
        writable_keys = ("source", "importance_score", "access_count", "tags", "vault_file", "related_memories", "profile_data", "preference_data", "project_goals", "session_data")
        for key in writable_keys:
            if key in updates:
                meta[key] = updates[key]
        self.store.update_metadata_value(memory_id, "text", meta.get("text", ""))
        for key in writable_keys:
            if key in updates:
                self.store.update_metadata_value(memory_id, key, updates[key])
        self.save_store()
        return self.get_memory_by_id(memory_id)

    def delete_memory(self, memory_id: str) -> bool:
        if memory_id not in self.store.list_all():
            return False
        self.store.delete(memory_id)
        self.save_store()
        return True

    def search(self, query: str, k: int = 10, threshold: float = 0.0) -> List[dict]:
        if not self.retriever:
            return []
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
        return enriched

    def synthesize(self, query: str, k: int = 10, model: Optional[str] = None) -> dict:
        if not self.retriever:
            return {"answer": "Retriever unavailable — cannot search memories.", "citations": [], "gaps": ["No retriever available"]}

        results = self.retriever.retrieve(query, k=k, min_score=0.0)
        if not results:
            return {"answer": "No relevant memories found.", "citations": [], "gaps": ["No matching memories for query"]}

        memory_context = []
        citations = []
        for i, r in enumerate(results):
            meta = dict(r.get("metadata", {}) or {})
            text = meta.pop("text", "") or ""
            memory_context.append(f"[{i+1}] {text}")
            citations.append({"id": r["id"], "text": text[:200], "score": r.get("score", 0)})

        context_str = "\n\n".join(memory_context)
        prompt = f"""You are Lino, a neural memory assistant. You have the following memories from your knowledge base. Use them to answer the user's question. Cite sources using [1], [2] etc. If the memories don't contain enough information to fully answer, acknowledge what you don't know. Be concise but thorough.

Memories:
{context_str}

Question: {query}

Answer (with citations):"""

        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            return {"answer": "GROQ_API_KEY not configured. Set it in .env or environment.", "citations": citations, "gaps": ["No API key"]}

        model_name = model or "llama-3.3-70b-versatile"
        import requests as req
        try:
            resp = req.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json={
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": "You are Lino, a neural memory assistant. Answer using the provided memories with citations. Be concise."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 2048,
                },
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            gaps = []
            gap_phrases = ["don't know", "not provided", "no information", "cannot answer", "not enough", "unable to", "not covered", "not mentioned", "no data"]
            for phrase in gap_phrases:
                if phrase in answer.lower():
                    gaps.append(f"Information gap: '{phrase}' mentioned in answer")
                    break
            if not gaps and len(answer) < 50:
                gaps.append("Answer was too brief — may lack sufficient context")

            return {"answer": answer, "citations": citations, "gaps": gaps, "model": model_name}
        except Exception as e:
            return {"answer": f"LLM call failed: {e}", "citations": citations, "gaps": [f"LLM error: {e}"]}

    def filter_by_metadata(self, filters: dict) -> List[dict]:
        if self.retriever:
            results = self.retriever.retrieve_by_metadata(filters)
        else:
            results = []
            for mid in self.store.list_all():
                meta = self.store.get_metadata(mid) or {}
                match = all(meta.get(k) == v for k, v in filters.items())
                if match:
                    results.append({"id": mid, "metadata": meta})
        enriched = []
        for r in results:
            meta = dict(r.get("metadata", {}))
            text = meta.pop("text", "")
            enriched.append({
                "id": r["id"],
                "text": text,
                "metadata": meta,
            })
        return enriched


memory_system = MemorySystem()


class ProfileResponse(BaseModel):
    id: str
    text: str
    profile_data: dict
    related_memories: List[str]


class ProfileUpdateRequest(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    bio: Optional[str] = None
    learning_goals: Optional[List[str]] = None
    preferences: Optional[Dict] = None


class ObserveRequest(BaseModel):
    type: str
    signal: str = ""
    explicit: bool = False
    source: str = ""

class LogSkillRequest(BaseModel):
    skill_path: str = ""
    skill_name: str = ""
    task_category: str = ""
    task_description: str = ""

class CreateMemoryRequest(BaseModel):
    text: str = Field(..., max_length=100000)
    source: str = "manual"
    importance: float = Field(0.5, ge=0.0, le=1.0)
    tags: Optional[List[str]] = None


class SynthesizeRequest(BaseModel):
    query: str = Field(..., max_length=100000)
    k: int = Field(10, ge=1, le=50)
    model: Optional[str] = None
    provider: Optional[str] = None


class PriorityUpdateRequest(BaseModel):
    priority: float = Field(..., ge=0.0, le=1.0)


class UpdateMemoryRequest(BaseModel):
    text: Optional[str] = Field(None, max_length=100000)
    source: Optional[str] = None
    importance: Optional[float] = Field(None, ge=0.0, le=1.0)
    tags: Optional[List[str]] = None
    vault_file: Optional[str] = None
    related_memories: Optional[List[str]] = None
    profile_data: Optional[Dict] = None
    preference_data: Optional[Dict] = None
    project_goals: Optional[List[str]] = None


class SearchRequest(BaseModel):
    query: str = Field(..., max_length=100000)
    k: int = 10
    threshold: float = 0.0


class FilterRequest(BaseModel):
    source: Optional[str] = None
    tags: Optional[List[str]] = None
    min_importance: Optional[float] = None


class PruneRequest(BaseModel):
    strategy: str = "hybrid"
    max_items: int = 10000


class SessionLogRequest(BaseModel):
    type: str = "message"
    content: str = Field(..., max_length=10000)
    tags: Optional[List[str]] = None
    importance: float = 0.5

class SessionCloseRequest(BaseModel):
    summary: str = ""

class SaveConfigRequest(BaseModel):
    embedder: Optional[Dict] = None
    consolidator: Optional[Dict] = None
    store: Optional[Dict] = None
    retriever: Optional[Dict] = None

class DreamRequest(BaseModel):
    max_memories: int = Field(5000, ge=100, le=50000)
    fail_fast: bool = False


# ── Pages ──

@app.get("/")
async def root_redirect():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/ui/")

@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(content={}, status_code=204)

@app.get("/ui/", response_class=HTMLResponse)
async def ui_page(request: Request, sort: str = ""):
    return templates.TemplateResponse(request, "index.html", {"request": request, "sort": sort})


# ── API Routes ──

@app.get("/api/memories")
@limiter.limit("200/minute")
async def list_memories(request: Request, limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0), reverse: bool = Query(False), sort: str = Query("newest", pattern="^(newest|importance)$")):
    memories, total = memory_system.list_memories(limit=limit, offset=offset, reverse=reverse, sort=sort)
    return {"memories": memories, "total": total, "limit": limit, "offset": offset, "sort": sort}


@app.get("/api/memories/{memory_id}")
@limiter.limit("200/minute")
async def get_memory(request: Request, memory_id: str):
    mem = memory_system.get_memory_by_id(memory_id)
    if mem is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return mem


@app.post("/api/memories")
@limiter.limit("30/minute")
async def create_memory(request: Request, req: CreateMemoryRequest):
    metadata = {
        "source": req.source,
        "importance_score": req.importance,
    }
    if req.tags:
        metadata["tags"] = req.tags
    try:
        result = memory_system.create_memory(req.text, metadata)
        return result
    except Exception as e:
        logger.error("Failed to create memory: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to create memory: {e}")


@app.put("/api/memories/{memory_id}")
@limiter.limit("30/minute")
async def update_memory(request: Request, memory_id: str, req: UpdateMemoryRequest):
    updates = {}
    if req.text is not None:
        updates["text"] = req.text
    if req.source is not None:
        updates["source"] = req.source
    if req.importance is not None:
        updates["importance_score"] = req.importance
    if req.tags is not None:
        updates["tags"] = req.tags
    if req.vault_file is not None:
        updates["vault_file"] = req.vault_file
    if req.related_memories is not None:
        updates["related_memories"] = req.related_memories
    result = memory_system.update_memory(memory_id, updates)
    if result is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return result


@app.patch("/api/memories/{memory_id}/priority")
@limiter.limit("30/minute")
async def update_priority(request: Request, memory_id: str, req: PriorityUpdateRequest):
    if memory_system.store.get_index_of(memory_id) is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    memory_system.store.update_metadata_value(memory_id, "importance_score", req.priority)
    memory_system.save_store()
    return {"status": "updated", "priority": req.priority}


@app.delete("/api/memories/{memory_id}")
@limiter.limit("30/minute")
async def delete_memory(request: Request, memory_id: str):
    ok = memory_system.delete_memory(memory_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"status": "deleted"}


@app.post("/api/relink")
@limiter.limit("5/minute")
async def batch_relink(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    skip_semantic = body.get("skip_semantic", False) if isinstance(body, dict) else False
    result = memory_system.batch_relink(skip_semantic=skip_semantic)
    return result


@app.post("/api/search")
@limiter.limit("60/minute")
async def search(request: Request, req: SearchRequest):
    results = memory_system.search(req.query, k=req.k, threshold=req.threshold)
    return {"results": results, "query": req.query, "k": req.k}


@app.post("/api/synthesize")
@limiter.limit("30/minute")
async def synthesize(request: Request, req: SynthesizeRequest):
    result = memory_system.synthesize(req.query, k=req.k, model=req.model)
    return result


@app.post("/api/filter")
@limiter.limit("60/minute")
async def filter_memories(request: Request, req: FilterRequest):
    filters = {}
    if req.source is not None:
        filters["source"] = req.source
    if req.tags is not None:
        filters["tags"] = req.tags
    if req.min_importance is not None:
        filters["importance_score"] = req.min_importance
    results = memory_system.filter_by_metadata(filters)
    return {"results": results}


@app.get("/api/graph")
@limiter.limit("60/minute")
async def get_graph(request: Request):
    store = memory_system.store
    ids = store.list_all()
    nodes = []
    edges = []
    edge_set = set()
    for mid in ids:
        meta = store.get_metadata(mid) or {}
        text = meta.pop("text", "")
        tags = meta.get("tags", []) or []
        importance = meta.get("importance_score", 0.5)
        related = meta.get("related_memories", []) or []
        related_types = meta.get("related_types", {}) or {}
        nodes.append({
            "id": mid,
            "text": text[:80] + ("..." if len(text) > 80 else ""),
            "importance": importance,
            "tags": tags,
            "source": meta.get("source", ""),
        })
        for rm in related:
            rid = rm.strip("[]") if rm.startswith("[") else rm
            ekey = tuple(sorted([mid, rid]))
            link_type = related_types.get(rid, "related")
            if rid in ids and ekey not in edge_set:
                edge_set.add(ekey)
                edges.append({"source": mid, "target": rid, "type": link_type})
    return {"nodes": nodes, "edges": edges, "total": len(nodes)}


@app.get("/api/stats")
@limiter.limit("60/minute")
async def stats(request: Request):
    store = memory_system.store
    total = len(store)
    dim = store.get_dimension()
    config = memory_system.config
    capacity = config.get("consolidator", {}).get("max_memories", 10000) if config else 10000
    embedder_ok = memory_system.embedder is not None
    sp = memory_system.store_path
    store_path = str(sp.parent.name + "/" + sp.name) if sp else ""
    return {
        "total": total,
        "dimension": dim,
        "capacity": capacity,
        "store_path": store_path,
        "embedder_online": embedder_ok,
        "status": "online" if embedder_ok else "degraded",
    }


@app.get("/api/profile")
@limiter.limit("30/minute")
async def get_profile(request: Request):
    mid = memory_system._ensure_profile_node()
    prefs_mid = memory_system._ensure_preferences_node()
    mem = memory_system.get_memory_by_id(mid)
    if not mem:
        raise HTTPException(status_code=500, detail="Profile not found")
    meta = mem["metadata"] or {}
    related = meta.get("related_memories", []) or []
    return {
        "id": mid,
        "text": meta.get("text", ""),
        "profile_data": meta.get("profile_data", {}),
        "preferences_id": prefs_mid,
        "related_memories": related,
    }


@app.put("/api/profile")
@limiter.limit("10/minute")
async def update_profile(request: Request, req: ProfileUpdateRequest):
    mid = memory_system._ensure_profile_node()
    meta = memory_system.store.get_metadata(mid) or {}
    pd = dict(meta.get("profile_data", {}))
    if req.name is not None:
        pd["name"] = req.name
    if req.role is not None:
        pd["role"] = req.role
    if req.bio is not None:
        pd["bio"] = req.bio
    if req.learning_goals is not None:
        pd["learning_goals"] = req.learning_goals
    if req.preferences is not None:
        existing_prefs = pd.get("preferences", {})
        existing_prefs.update(req.preferences)
        pd["preferences"] = existing_prefs
        prefs_mid = memory_system._ensure_preferences_node()
        prefs_meta = memory_system.store.get_metadata(prefs_mid) or {}
        prefs_meta["preference_data"] = req.preferences
        memory_system.store.update_metadata_value(prefs_mid, "text", f"Preferences: {req.preferences.get('answer_style', 'concise')} style")
        memory_system.store.update_metadata_value(prefs_mid, "preference_data", req.preferences)
    meta["profile_data"] = pd
    text_parts = []
    if pd.get("name"): text_parts.append(pd["name"])
    if pd.get("role"): text_parts.append(f"({pd['role']})")
    meta["text"] = " ".join(text_parts) if text_parts else "Your Identity"
    memory_system.store.update_metadata_value(mid, "profile_data", pd)
    memory_system.store.update_metadata_value(mid, "text", meta["text"])
    memory_system.save_store()
    return {"status": "saved", "profile_data": pd}


@app.post("/api/preferences/observe")
@limiter.limit("60/minute")
async def observe_preference(request: Request, req: ObserveRequest):
    result = memory_system.observe_preference(
        pref_type=req.type,
        signal=req.signal,
        explicit=req.explicit,
        source=req.source,
    )
    return result


@app.post("/api/preferences/log-skill")
@limiter.limit("60/minute")
async def log_skill(request: Request, req: LogSkillRequest):
    result = memory_system.log_skill_usage(
        skill_path=req.skill_path,
        skill_name=req.skill_name,
        task_category=req.task_category,
        task_description=req.task_description,
    )
    return result


@app.post("/api/preferences/consolidate")
@limiter.limit("10/minute")
async def consolidate_preferences(request: Request):
    result = memory_system.consolidate_preferences()
    return result


@app.get("/api/preferences/observations")
@limiter.limit("30/minute")
async def get_observations(request: Request):
    obs = memory_system._load_observations()
    return {"observations": obs, "count": len(obs)}


@app.get("/api/preferences/learned")
@limiter.limit("30/minute")
async def get_learned(request: Request):
    learned = memory_system.get_learned_preferences()
    return {"learned_preferences": learned}


@app.post("/api/prune")
@limiter.limit("10/minute")
async def prune(request: Request, req: PruneRequest):
    try:
        removed = memory_system.consolidator.prune(
            memory_system.store,
            max_size=req.max_items,
            strategy=req.strategy,
        )
        memory_system.save_store()
        return {"removed": removed, "remaining": len(memory_system.store)}
    except Exception as e:
        logger.error("Pruning failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Pruning failed: {e}")


# ── Session Endpoints ──

@app.post("/api/session/log")
@limiter.limit("60/minute")
async def session_log(request: Request, req: SessionLogRequest):
    return memory_system.append_to_session(
        entry_type=req.type,
        content=req.content,
        tags=req.tags,
        importance=req.importance,
    )

@app.get("/api/session")
@limiter.limit("30/minute")
async def session_get(request: Request):
    session = memory_system.get_active_session()
    if not session:
        return {"status": "no_active_session"}
    return session

@app.post("/api/session/close")
@limiter.limit("10/minute")
async def session_close(request: Request, req: SessionCloseRequest):
    return memory_system.close_active_session(summary=req.summary)

@app.get("/api/session/history")
@limiter.limit("30/minute")
async def session_history(request: Request, limit: int = Query(10, ge=1, le=100)):
    return {"sessions": memory_system.list_sessions(limit=limit)}

# ── Graph Traversal ──

@app.get("/api/graph/traverse")
@limiter.limit("30/minute")
async def graph_traverse(
    request: Request,
    start_id: str = Query(...),
    depth: int = Query(3, ge=1, le=10),
    types: Optional[str] = Query(None),
):
    type_list = types.split(",") if types else None
    result = memory_system.traverse_graph(start_id, depth=depth, types=type_list)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

@app.get("/api/memories/{memory_id}/backlinks")
@limiter.limit("60/minute")
async def get_backlinks(request: Request, memory_id: str):
    if memory_system.store.get_index_of(memory_id) is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    backlinks = memory_system.get_backlinks(memory_id)
    return {"backlinks": backlinks, "count": len(backlinks)}

@app.get("/api/config")
@limiter.limit("60/minute")
async def get_config(request: Request):
    return memory_system.config


@app.put("/api/config")
@limiter.limit("10/minute")
async def save_config(request: Request, req: SaveConfigRequest):
    cfg = memory_system.config
    if req.embedder is not None:
        cfg["embedder"] = req.embedder
    if req.consolidator is not None:
        cfg["consolidator"] = req.consolidator
    if req.store is not None:
        cfg["store"] = req.store
    if req.retriever is not None:
        cfg["retriever"] = req.retriever
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(cfg, f)
    memory_system.config = cfg
    return {"status": "saved", "config": cfg}


@app.post("/api/backup")
@limiter.limit("10/minute")
async def backup(request: Request):
    ids = memory_system.store.list_all()
    memories = []
    for mid in ids:
        idx = memory_system.store.get_index_of(mid)
        meta = memory_system.store.get_metadata(mid) or {}
        emb = memory_system.store.get_embedding(idx) if idx is not None else None
        text = meta.pop("text", "")
        memories.append({
            "id": mid,
            "text": text,
            "embedding": emb.tolist() if emb is not None else None,
            "metadata": meta,
        })
    return {"memories": memories, "exported_at": time.time()}


@app.post("/api/restore")
@limiter.limit("5/minute")
async def restore(request: Request, file: UploadFile):
    try:
        content = await file.read()
        data = json.loads(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid JSON in upload file")
    memories = data.get("memories", [])
    imported = 0
    for mem in memories:
        mid = mem.get("id", str(uuid.uuid4()))
        text = mem.get("text", "")
        embedding = mem.get("embedding")
        metadata = dict(mem.get("metadata", {}))
        if text:
            metadata["text"] = text
        if embedding is not None:
            if not isinstance(embedding, list):
                raise HTTPException(status_code=400, detail="Embedding must be a list of floats")
            if len(embedding) == 0:
                raise HTTPException(status_code=400, detail="Embedding must not be empty")
            memory_system.store.store(mid, embedding, metadata)
        elif memory_system.embedder:
            emb = memory_system.embedder.embed(text)
            memory_system.store.store(mid, emb, metadata)
        imported += 1
    memory_system.save_store()
    return {"imported": imported}


# ── Dream Cycle ──

from scripts.dream_cycle import run_dream_cycle

@app.post("/api/dream", response_class=JSONResponse)
@limiter.limit("5/minute")
async def dream_cycle_endpoint(request: Request, req: DreamRequest):
    logger.info("Dream Cycle triggered via API")
    try:
        result = run_dream_cycle({"max_memories": req.max_memories, "fail_fast": req.fail_fast})
        return {"status": "ok", "result": result}
    except Exception as e:
        logger.exception("Dream Cycle failed")
        raise HTTPException(status_code=500, detail=f"Dream Cycle failed: {e}")


# ── Job Queue ──

from scripts.job_queue import job_queue

class EnqueueJobRequest(BaseModel):
    job_type: str = Field(..., max_length=100)
    params: Optional[Dict] = None
    priority: int = Field(0, ge=-10, le=10)

class JobQueueStatusRequest(BaseModel):
    limit: int = Field(50, ge=1, le=500)

@app.post("/api/jobs/enqueue")
@limiter.limit("30/minute")
async def enqueue_job(request: Request, req: EnqueueJobRequest):
    job = job_queue.enqueue(job_type=req.job_type, params=req.params or {}, priority=req.priority)
    return {"status": "enqueued", "job": job.to_dict()}

@app.get("/api/jobs/next")
@limiter.limit("30/minute")
async def next_job(request: Request):
    job = job_queue.next()
    if job is None:
        return {"status": "no_job"}
    return {"status": "ok", "job": job.to_dict()}

@app.post("/api/jobs/{job_id}/complete")
@limiter.limit("30/minute")
async def complete_job(request: Request, job_id: str, result: str = Form("ok"), error: str = Form(None)):
    job_queue.complete(job_id, result=result, error=error)
    return {"status": "completed"}

@app.post("/api/jobs/{job_id}/fail")
@limiter.limit("30/minute")
async def fail_job(request: Request, job_id: str, error: str = Form(...)):
    job_queue.fail(job_id, error=error)
    return {"status": "failed"}

@app.get("/api/jobs/status", response_class=JSONResponse)
@limiter.limit("30/minute")
async def job_queue_status(request: Request, limit: int = Query(50, ge=1, le=500)):
    return job_queue.status(limit=limit)

@app.get("/api/jobs/stats")
@limiter.limit("30/minute")
async def job_queue_stats(request: Request):
    return job_queue.stats()
