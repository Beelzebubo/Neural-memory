import json
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src import TextEmbedder, VectorMemoryStore, MemoryRetriever, MemoryConsolidator

CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"
DEFAULT_STORE_PATH = Path.home() / ".neural_memory" / "store.pkl"

app = FastAPI(title="Neural Memory UI")


# ── Security Headers Middleware ──

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' https://unpkg.com; "
        "style-src 'self' https://fonts.googleapis.com https://unpkg.com; "
        "font-src 'self' https://fonts.gstatic.com https://unpkg.com; "
        "img-src 'self' data:; "
        "connect-src 'self'"
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

    def list_memories(self, limit: int = 50, offset: int = 0) -> tuple:
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
                "embedding": emb.tolist() if emb is not None else None,
                "metadata": meta,
            })
        return results, total

    def create_memory(self, text: str, metadata: dict) -> dict:
        memory_id = str(uuid.uuid4())
        metadata_clean = {
            "text": text,
            "source": metadata.get("source", "manual"),
            "importance_score": metadata.get("importance_score", 0.5),
            "access_count": 0,
            "timestamp": time.time(),
        }
        if "tags" in metadata:
            metadata_clean["tags"] = metadata["tags"]
        if self.embedder:
            emb = self.embedder.embed(text)
        else:
            dim = 384
            import numpy as np
            emb = np.zeros(dim).tolist()
        self.store.store(memory_id, emb, metadata_clean)
        self.save_store()
        return {
            "id": memory_id,
            "text": text,
            "metadata": metadata_clean,
        }

    def update_memory(self, memory_id: str, updates: dict) -> Optional[dict]:
        ids = self.store.list_all()
        if memory_id not in ids:
            return None
        meta = dict(self.store._metadata.get(memory_id, {}))
        if "text" in updates:
            meta["text"] = updates["text"]
            if self.embedder:
                new_emb = self.embedder.embed(updates["text"])
                idx = ids.index(memory_id)
                self.store._embeddings[idx] = __import__("numpy").array(new_emb, dtype=__import__("numpy").float32)
        for key in ("source", "importance_score", "access_count", "tags"):
            if key in updates:
                meta[key] = updates[key]
        self.store._metadata[memory_id] = meta
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

    def filter_by_metadata(self, filters: dict) -> List[dict]:
        if self.retriever:
            results = self.retriever.retrieve_by_metadata(filters)
        else:
            results = []
            for mid in self.store.list_all():
                meta = dict(self.store._metadata.get(mid, {}))
                match = True
                for k, v in filters.items():
                    if k not in meta or meta[k] != v:
                        match = False
                        break
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


class CreateMemoryRequest(BaseModel):
    text: str = Field(..., max_length=100000)
    source: str = "manual"
    importance: float = Field(0.5, ge=0.0, le=1.0)
    tags: Optional[List[str]] = None


class UpdateMemoryRequest(BaseModel):
    text: Optional[str] = Field(None, max_length=100000)
    source: Optional[str] = None
    importance: Optional[float] = Field(None, ge=0.0, le=1.0)
    tags: Optional[List[str]] = None


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


class SaveConfigRequest(BaseModel):
    embedder: Optional[Dict] = None
    consolidator: Optional[Dict] = None
    store: Optional[Dict] = None
    retriever: Optional[Dict] = None


# ── Pages ──

@app.get("/ui/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"request": request})


# ── API Routes ──

@app.get("/api/memories")
async def list_memories(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    memories, total = memory_system.list_memories(limit=limit, offset=offset)
    return {"memories": memories, "total": total, "limit": limit, "offset": offset}


@app.get("/api/memories/{memory_id}")
async def get_memory(memory_id: str):
    mem = memory_system.get_memory_by_id(memory_id)
    if mem is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return mem


@app.post("/api/memories")
async def create_memory(req: CreateMemoryRequest):
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
        raise HTTPException(status_code=500, detail="Failed to create memory")


@app.put("/api/memories/{memory_id}")
async def update_memory(memory_id: str, req: UpdateMemoryRequest):
    updates = {}
    if req.text is not None:
        updates["text"] = req.text
    if req.source is not None:
        updates["source"] = req.source
    if req.importance is not None:
        updates["importance_score"] = req.importance
    if req.tags is not None:
        updates["tags"] = req.tags
    result = memory_system.update_memory(memory_id, updates)
    if result is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return result


@app.delete("/api/memories/{memory_id}")
async def delete_memory(memory_id: str):
    ok = memory_system.delete_memory(memory_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"status": "deleted"}


@app.post("/api/search")
async def search(req: SearchRequest):
    results = memory_system.search(req.query, k=req.k, threshold=req.threshold)
    return {"results": results, "query": req.query, "k": req.k}


@app.post("/api/filter")
async def filter_memories(req: FilterRequest):
    filters = {}
    if req.source is not None:
        filters["source"] = req.source
    if req.tags is not None:
        filters["tags"] = req.tags
    if req.min_importance is not None:
        filters["importance_score"] = req.min_importance
    results = memory_system.filter_by_metadata(filters)
    return {"results": results}


@app.get("/api/stats")
async def stats():
    store = memory_system.store
    total = len(store)
    dim = store._dim if hasattr(store, "_dim") else 0
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


@app.post("/api/prune")
async def prune(req: PruneRequest):
    try:
        removed = memory_system.consolidator.prune(
            memory_system.store,
            max_size=req.max_items,
            strategy=req.strategy,
        )
        memory_system.save_store()
        return {"removed": removed, "remaining": len(memory_system.store)}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Pruning failed")


@app.get("/api/config")
async def get_config():
    return memory_system.config


@app.put("/api/config")
async def save_config(req: SaveConfigRequest):
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
async def backup():
    ids = memory_system.store.list_all()
    memories = []
    for mid in ids:
        idx = ids.index(mid)
        meta = dict(memory_system.store._metadata.get(mid, {}))
        emb = memory_system.store._embeddings[idx] if idx < len(memory_system.store._embeddings) else None
        memories.append({
            "id": mid,
            "text": meta.pop("text", ""),
            "embedding": emb.tolist() if emb is not None else None,
            "metadata": meta,
        })
    return {"memories": memories, "exported_at": time.time()}


@app.post("/api/restore")
async def restore(file: UploadFile):
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
