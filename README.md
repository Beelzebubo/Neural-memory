# Lino — Neural Memory Server

A memory system that runs on your laptop. Feed it notes, conversations, or code, and it stores them as searchable vectors. Then you can ask questions, browse a knowledge graph, or run brainstorming sessions. All through a web UI, API, or CLI.

## Quick Start

```bash
pip install -r requirements.txt
python -m uvicorn ui.app:app --host 127.0.0.1 --port 8210 --reload
```

Open [http://127.0.0.1:8210/ui/](http://127.0.0.1:8210/ui/).

## What it does

### Store and search

Give it text. It turns that into a 384-dimensional vector and stores it. Search by meaning, not keywords.

```python
from src import TextEmbedder, VectorMemoryStore, MemoryRetriever
embedder = TextEmbedder()
store = VectorMemoryStore()
retriever = MemoryRetriever(embedder, store)

emb = embedder.embed("The user prefers dark mode.")
store.store("mem_001", emb, {"source": "conversation", "importance": 0.8})
results = retriever.retrieve("What theme do they like?", k=5)
```

### Auto-linking

Every new memory links to:
- Your profile node (identity)
- Entity nodes for `project:*` and `person:*` tags
- Wikilink targets — `[[memory text]]` or typed `works at [[X]]`
- Top-5 similar memories (cosine ≥ 0.25)

Typed wikilinks: `works at`, `employed by`, `part of`, `member of`, `founded`, `created`, `related to`, `connected to`, `mentions`. Re-links automatically on server startup.

### Knowledge graph

All memories and connections form a graph. View it in the UI (force-directed D3.js layout) or query it:

```
GET /api/graph
GET /api/graph/traverse?start_id=...&depth=3&types=works_at,founded
GET /api/memories/{id}/backlinks
```

### Synthesis (RAG)

Ask questions, get answers from your own data with numbered citations:

```bash
curl -X POST http://localhost:8210/api/synthesize \
  -H "Content-Type: application/json" \
  -d '{"query": "What do I know about the user?", "k": 10}'
```

Uses Groq `llama-3.3-70b-versatile` (needs `GROQ_API_KEY` in `.env`).

### Brainstorming

Three-provider fallback (Gemini → Groq → Bluesminds). Two phases:

1. **Rough draft** — 13 curated skills produce initial ideas
2. **Expert plans** — Matches against 20 agency-agents divisions for detailed insights

Cross-pollination finds meta-insights. Similar ideas merge automatically. You can also merge multiple sessions into one.

```bash
curl -X POST http://localhost:8210/api/brainstorm \
  -H "Content-Type: application/json" \
  -d '{"topic": "How to get into NYU Abu Dhabi", "n_ideas": 5}'
```

The UI shows a force-directed graph. Click a node to see the thinking process. Bottom panel has a session summary with "Show More" for full detail.

### Session management

Track conversations as session memories. Log decisions, facts, changes in real time. Active sessions never get pruned.

```bash
curl -X POST http://localhost:8210/api/session/log \
  -H "Content-Type: application/json" \
  -d '{"type": "decision", "content": "Started Knowledge Graph feature", "tags": ["project:knowledge-graph"], "importance": 0.8}"
```

### Preference learning

The system watches what you do and learns over time.

| When you... | It learns... | After |
|---|---|---|
| Ask for code | Code style | 5 times |
| Ask for more/less detail | Answer length | 5 times |
| Use the same skill repeatedly | Skill mapping | 3 times |
| Say "always" or "never" | Direct preference | 1 time |

Preferences consolidate into `~/.neural_memory/lino-identity.md`.

### Profile

A special node (tagged `type:profile`) stores your name, role, bio, and locked preferences. Entity nodes are auto-created for `project:*` and `person:*` tags.

### Web UI

Dark SPA with:
- **Browse** — sort by newest or importance, filter by tags/source
- **Detail panel** — full text, metadata, priority slider, related memories, backlinks
- **Knowledge graph** — D3.js force-directed graph with edge labels and legend
- **Brainstorm view** — topics sidebar, graph, thinking process panel, summary
- **Create / edit / delete** memories
- **Semantic search**

## REST API

| Method | Path | What it does |
|--------|------|-------------|
| `GET` | `/ui/` | Web UI |
| `GET/POST/PUT/DELETE` | `/api/memories` | CRUD memories |
| `POST` | `/api/search` | Semantic search |
| `POST` | `/api/synthesize` | RAG Q&A with citations |
| `GET` | `/api/graph` | Full knowledge graph |
| `GET` | `/api/graph/traverse` | BFS traversal |
| `GET` | `/api/memories/{id}/backlinks` | Incoming links |
| `GET` | `/api/stats` | System stats |
| `POST` | `/api/relink` | Re-link all memories |
| `POST` | `/api/backup` / `restore` | Export / import JSON |
| `POST` | `/api/prune` | Prune old memories |
| `GET/PUT` | `/api/profile` | Identity profile |
| `POST` | `/api/preferences/*` | Preference learning |
| `POST` | `/api/session/*` | Session management |
| `POST` | `/api/brainstorm` | Run brainstorm |
| `GET` | `/api/brainstorm/sessions` | List sessions |
| `GET/DELETE` | `/api/brainstorm/session/{id}` | Get / delete session |
| `POST` | `/api/brainstorm/merge` | Merge sessions |
| `POST` | `/api/brainstorm/consolidate/{id}` | Consolidate ideas |

## Integration

**Hermes MCP:** 23 tools over stdio JSON-RPC:

```bash
hermes mcp add lino --command "python3 /path/to/integration/mcp_server.py"
```

**CLI:**

```bash
python integration/cli.py store "text" --source x --importance 0.8
python integration/cli.py search "query" -k 5
python integration/cli.py stats
```

## Deployment

```bash
# Systemd (auto-start)
cp systemd/lino.service ~/.config/systemd/user/
systemctl --user enable --now lino.service

# Docker
docker compose up --build -d
```

Auto-start script at `bin/lino-server.sh` — called by opencode and Hermes.

## Configuration

| Var | Default | What it does |
|-----|---------|-------------|
| `GROQ_API_KEY` | — | For synthesis & compression |
| `LINO_API_KEY` | — | API key for writes |
| `LINO_HOST` | `127.0.0.1` | Bind address |
| `LINO_PORT` | `8210` | Port |

Config file at `config/config.yaml` — controls embedder model, retriever settings, pruning strategy, etc.

## Where data lives

| Path | Contents |
|------|----------|
| `~/.neural_memory/store.pkl` | All memories + embeddings |
| `~/.neural_memory/brainstorm_sessions.json` | Brainstorm sessions |
| `~/.neural_memory/lino-identity.md` | Generated identity document |
| `~/.neural_memory/preference_observations.json` | Learned preferences |
| `~/.neural_memory/job_queue.json` | Job queue |
| `~/.neural_memory/consolidated_exports/` | Dream cycle exports |

After each dream cycle, the cleaned store exports as markdown files. With Obsidian: set `OBSIDIAN_VAULT_PATH` in `.env`. Without: exports go to `consolidated_exports/`. Each folder has an `INDEX.md`.

## Scripts

| Script | What it does |
|--------|-------------|
| `scripts/start.sh` / `stop.sh` | Server start/stop |
| `scripts/backup.sh` / `restore.sh` | Backup/restore |
| `scripts/seed.sh` | Import Obsidian vault |
| `scripts/dream_cycle.py` | 24-phase overnight maintenance |
| `scripts/brainstorm.py` | Brainstorming engine |
| `scripts/job_queue.py` | Job queue with worker |
| `scripts/watchdog_sync.py` | Vault file watcher |
| `scripts/rip_and_compress.py` | LLM compression |
| `scripts/evaluate.py` | Retrieval accuracy |

## Tests

```bash
pytest tests/ -v
```

73 tests, 1 skipped. Uses mock embedders — no real Sentence Transformers needed.

## Roadmap

| Feature | Status |
|---------|--------|
| Auto-Linking | ✅ |
| Synthesis (RAG) | ✅ |
| Knowledge Graph | ✅ |
| Session Logging | ✅ |
| Preference Learning | ✅ |
| Profile / Identity | ✅ |
| Dream Cycle | ✅ |
| Job Queue | ✅ |
| Brainstorming | ✅ |
| Schema Packs | ⬜ |
| Code Intelligence (Tree-sitter) | ⬜ |
| Ingestion / Import | ⬜ |
| Calibration | ⬜ |

## License

MIT
