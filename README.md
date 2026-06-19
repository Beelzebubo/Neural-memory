# Lino — Neural Memory Server

A self-hosted semantic memory backend. Turns notes, conversations, and knowledge into persistent, searchable vector embeddings — accessible through a web UI, REST API, MCP tools, and CLI.

```
                    ┌──────────────────────┐
                    │   Web UI (Dark SPA)   │
                    │   D3 Knowledge Graph  │
                    └──────────┬───────────┘
                               │ HTTP
                    ┌──────────▼───────────┐
                    │   FastAPI Server      │  ~30 REST endpoints
                    │   Rate Limited        │  api key auth, CORS
                    └──────────┬───────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
   Hermes MCP Tools     Direct Python Import     CLI
   (17 tools)           (src/)                   (10 commands)
                               │
                    ┌──────────▼───────────┐
                    │   MemorySystem        │
                    │   auto-link + wikilink│
                    │   synthesis (RAG Q&A) │
                    │   knowledge graph BFS │
                    │   session management  │
                    │   preference learning │
                    └──────────┬───────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
  TextEmbedder         VectorMemoryStore      MemoryRetriever
  (384-d, norm'd)      FAISS / numpy          semantic + hybrid
         │                pickle persist      metadata filter
         └────────────────────┬─────────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │ MemoryConsolidator  │
                    │ cluster / merge     │
                    │ prune (4 strategies)│
                    │ link cleanup        │
                    └────────────────────┘
```

## Quick Start

```bash
pip install -r requirements.txt

# Dev mode (hot reload)
cd scripts && ./start.sh dev

# Or manually:
python -m uvicorn ui.app:app --host 127.0.0.1 --port 8210 --reload
```

Open [http://127.0.0.1:8210/ui/](http://127.0.0.1:8210/ui/) in a browser.

## Features

### Semantic Memory

Store text → it's embedded as a 384-dimensional vector using Sentence Transformers. Search by meaning, not keywords.

```python
from src import TextEmbedder, VectorMemoryStore, MemoryRetriever

embedder = TextEmbedder()
store = VectorMemoryStore()
retriever = MemoryRetriever(embedder, store)

emb = embedder.embed("The user prefers dark mode.")
store.store("mem_001", emb, {"source": "conversation", "importance": 0.8})

results = retriever.retrieve("What theme do they like?", k=5)
```

### Auto-Linking on Write

Every new memory is automatically linked to:
- The **profile node** (identity)
- **Entity nodes** for `project:*` and `person:*` tags
- **Wikilink targets** — bare `[[memory text]]` or typed `works at [[X]]`
- **Semantic neighbors** — top-5 similar memories (cosine ≥ 0.25)

Typed wikilinks recognized: `works at`, `employed by`, `part of`, `member of`, `founded`, `created`, `related to`, `connected to`, `mentions`.

### Batch Relink (`POST /api/relink`)

Re-scans all existing memories to re-establish wikilinks, typed links, and semantic links. Runs automatically on server startup.

### Knowledge Graph

Full bidirectional graph of all memories and their relationships.

- `GET /api/graph` — all nodes and edges with typed relationship colors
- `GET /api/graph/traverse?start_id=...&depth=3&types=works_at,founded` — BFS traversal
- `GET /api/memories/{id}/backlinks` — incoming links to a memory

The web UI renders an interactive D3.js force-directed graph with edge labels for typed links.

### Synthesis (RAG Q&A)

Ask questions and get answers sourced from your memories:

```bash
curl -X POST http://localhost:8210/api/synthesize \
  -H "Content-Type: application/json" \
  -d '{"query": "What do I know about the user?", "k": 10}'
```

Returns answer text, numbered citations `[1]`, `[2]`, and knowledge gaps. Uses Groq `llama-3.3-70b-versatile` (requires `GROQ_API_KEY` in `.env`).

### Session Management

Active conversations are tracked as session memories. Log decisions, facts, changes in real-time:

```bash
# Log an entry
curl -X POST http://localhost:8210/api/session/log \
  -H "Content-Type: application/json" \
  -d '{"type": "decision", "content": "Started Knowledge Graph feature", "tags": ["project:knowledge-graph"], "importance": 0.8}'

# Close session with summary
curl -X POST http://localhost:8210/api/session/close \
  -H "Content-Type: application/json" \
  -d '{"summary": "Implemented typed edges and BFS traversal"}'
```

Active sessions (`protected: True`) are never pruned. Closed sessions are eligible for nightly cleanup.

### Brainstorming

Two-phase brainstorming engine with 3-provider LLM fallback (Gemini → Groq → Bluesminds):

- **Phase 1** — Rough draft using 13 curated `.hermes` skills
- **Phase 2** — Expert plans matched against 20 agency-agents divisions
- **Cross-pollination** — Meta-insights combining multiple Phase 2 ideas
- **Idea consolidation** — Automatically merges semantically similar nodes (word-overlap > 35%)
- **Session merge** — Combine multiple brainstorm sessions into one with cross-session dedup
- **Web UI** — Force-directed graph with typed E-R edges, right-side thinking process panel, session summary with "Show More" full-page view
- **Topics sidebar** — Per-topic delete, active indicator (✓), click-to-toggle selection

```bash
curl -X POST http://localhost:8210/api/brainstorm \
  -H "Content-Type: application/json" \
  -d '{"topic": "How to get into NYU Abu Dhabi", "n_ideas": 5}'
```

Stored separately from the knowledge graph at `~/.neural_memory/brainstorm_sessions.json`.

### Preference Learning

The system learns user preferences over time from observations:

| Observation Type | Trigger | Threshold |
|---|---|---|
| `code_style` | User asks for code examples | 5 |
| `answer_verbosity` | User asks for more/fewer details | 5 |
| `skill_usage` | Same skill used repeatedly | 3 |
| `explicit_preference` | Direct statement | 1 |

Preferences consolidate into a profile identity document at `~/.neural_memory/lino-identity.md`.

### Profile / Identity

A special memory node (tagged `type:profile`) stores user identity: name, role, bio, learning goals, and locked preferences. Entity nodes are auto-created for `project:*` and `person:*` tags.

### Web UI

Dark-themed SPA with:
- **Browse** — sort by newest or importance, filter by tags/source
- **Detail panel** — full memory text, metadata, priority slider, related memories with relationship types, backlinks
- **Knowledge graph** — interactive D3.js force-directed graph with edge labels and legend
- **Create / edit / delete** memories
- **Search** — semantic search with results grouped by score

### Priority Slider

Each memory has a drag-adjustable priority (0.0–1.0). Auto-saves after 500ms debounce with visual save indicator — no re-embedding needed.

## REST API

All ~30 endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/ui/` | Web UI SPA |
| `GET` | `/api/memories` | List (limit, offset, sort=newest|importance) |
| `GET` | `/api/memories/{id}` | Get by ID |
| `POST` | `/api/memories` | Create (text, source, importance, tags) |
| `PUT` | `/api/memories/{id}` | Update |
| `PATCH` | `/api/memories/{id}/priority` | Update priority (fast, no re-embed) |
| `DELETE` | `/api/memories/{id}` | Delete |
| `POST` | `/api/relink` | Batch relink all memories |
| `POST` | `/api/search` | Semantic search (query, k, threshold) |
| `POST` | `/api/synthesize` | RAG Q&A with citations |
| `POST` | `/api/filter` | Filter by source, tags, importance |
| `GET` | `/api/graph` | Full knowledge graph |
| `GET` | `/api/graph/traverse` | BFS traversal from node |
| `GET` | `/api/memories/{id}/backlinks` | Incoming links |
| `GET` | `/api/stats` | System stats |
| `GET` | `/api/config` | Get config |
| `PUT` | `/api/config` | Update config |
| `POST` | `/api/backup` | Export all as JSON |
| `POST` | `/api/restore` | Import from JSON |
| `POST` | `/api/prune` | Prune memories |
| `GET` | `/api/profile` | Get identity profile |
| `PUT` | `/api/profile` | Update profile |
| `POST` | `/api/preferences/observe` | Log preference observation |
| `POST` | `/api/preferences/log-skill` | Log skill usage |
| `POST` | `/api/preferences/consolidate` | Consolidate learned prefs |
| `GET` | `/api/preferences/observations` | Raw observations |
| `GET` | `/api/preferences/learned` | Learned preferences |
| `POST` | `/api/session/log` | Log to active session |
| `GET` | `/api/session` | Get active session |
| `POST` | `/api/session/close` | Close session with summary |
| `GET` | `/api/session/history` | Past sessions |
| `POST` | `/api/brainstorm` | Run brainstorm (topic, n_ideas) |
| `GET` | `/api/brainstorm/sessions` | List all brainstorm sessions |
| `GET` | `/api/brainstorm/session/{id}` | Get session with nodes + edges |
| `DELETE` | `/api/brainstorm/session/{id}` | Delete session |
| `POST` | `/api/brainstorm/merge` | Merge multiple sessions into one |
| `POST` | `/api/brainstorm/consolidate/{id}` | Consolidate similar ideas in session |

## Integration

### Hermes Agent (MCP)

The MCP server exposes **23 tools** over stdio JSON-RPC (19 Hermes + 3 brainstorm + 1 consolidate):

```bash
hermes mcp add lino --command "python3 /path/to/integration/mcp_server.py"
```

Available tools: `store`, `search`, `recall`, `list`, `stats`, `delete`, `prune`, `update_priority`, `run_sync`, `run_compress`, `watchdog`, `session_done`, `link`, `get_profile`, `observe_preference`, `log_skill`, `consolidate_preferences`, `log_session`, `restore`, `backup`.

### Direct CLI

```bash
python integration/cli.py store "text" --source x --importance 0.8
python integration/cli.py search "query" -k 5 --threshold 0.5
python integration/cli.py get <id>
python integration/cli.py stats
python integration/cli.py link <id> [--all]
python integration/cli.py compress [--dry-run] [--provider groq]
python integration/cli.py session-done "summary summary"
```

### Any HTTP Agent (opencode, Copilot, etc.)

```bash
curl -X POST http://localhost:8210/api/memories \
  -H "Content-Type: application/json" \
  -d '{"text": "User prefers dark mode", "importance": 0.8}'

curl "http://localhost:8210/api/search?query=dark+mode&k=5"
```

## Deployment

### Systemd (auto-start at login)

```bash
cp systemd/lino.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now lino.service
```

Health-check auto-start script at `bin/lino-server.sh` — called by opencode session start and Hermes plugin init.

### Docker

```bash
docker compose up --build -d
```

### Nginx (reverse proxy)

Config at `nginx/lino.conf` — proxies `lino.local` to port 8210, caches static assets for 7 days.

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `GROQ_API_KEY` | — | For synthesis & compression |
| `LINO_API_KEY` | — | API key for write endpoints |
| `LINO_HOST` | `127.0.0.1` | Bind address |
| `LINO_PORT` | `8210` | Port |
| `LINO_TIMEOUT` | `30` | Server start timeout (s) |

Config file at `config/config.yaml`:

| Key | Default | Description |
|-----|---------|-------------|
| `embedder.model_name` | `all-MiniLM-L6-v2` | Sentence Transformer model |
| `retriever.default_k` | `10` | Default top-K |
| `retriever.hybrid_alpha` | `0.5` | Semantic vs keyword weight |
| `consolidator.max_memories` | `5000` | Hard cap |
| `consolidator.similarity_threshold` | `0.85` | Cosine threshold for dedup |
| `consolidator.prune_strategy` | `by_importance` | `by_age`, `by_importance`, `by_access_frequency`, `hybrid` |
| `vault.sync_interval_hours` | `6` | Vault sync frequency |

## Persistence

| Path | Contents |
|------|----------|
| `~/.neural_memory/store.pkl` | All memories + embeddings + metadata |
| `~/.neural_memory/preference_observations.json` | Observed preferences (max 10k) |
| `~/.neural_memory/lino-identity.md` | Generated identity document |
| `~/.neural_memory/watchdog.pid` | Watchdog daemon PID |
| `~/.neural_memory/consolidated_exports/` | Dream cycle markdown exports (fallback — no Obsidian) |
| `~/.neural_memory/job_queue.json` | Persistent job queue (crash-safe) |
| `~/.neural_memory/brainstorm_sessions.json` | Brainstorm sessions (separate from store.pkl) |
| `~/.neural_memory/dream_report.md` | Dream cycle phase report |

### Consolidated Export

After each dream cycle, the cleaned memory store is exported as markdown files:

- **With Obsidian vault:** Set `OBSIDIAN_VAULT_PATH=~/Documents/YourVault` in `.env`.  
  Exports go to `YourVault/_consolidated/YYYY-MM-DD/`. Each memory = one `.md` file.
- **Without Obsidian:** No config needed. Exports go to  
  `~/.neural_memory/_consolidated/YYYY-MM-DD/`. Same format, readable in any markdown viewer.

Each export folder includes an `INDEX.md` listing all exported memories with ID, title, importance, and tags.

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/start.sh` | Start server (dev / prod) |
| `scripts/stop.sh` | Graceful stop |
| `scripts/seed.sh` | Import Obsidian vault `.md` files |
| `scripts/backup.sh` | Timestamped backup (keeps last 10) |
| `scripts/restore.sh` | Restore from backup |
| `scripts/maintenance.sh` | Nightly compress + prune |
| `scripts/healthcheck.sh` | Server health check |
| `scripts/watchdog_sync.py` | Real-time vault file watcher daemon |
| `scripts/rip_and_compress.py` | LLM compression pipeline |
| `scripts/session_memory.py` | Session summarization |
| `scripts/evaluate.py` | Retrieval accuracy (Recall@K, MAP, MRR) |
| `scripts/train.py` | Batch embedding training |
| `scripts/dream_cycle.py` | 22-phase overnight maintenance pipeline (includes brainstorm phase) |
| `scripts/brainstorm.py` | Two-phase brainstorming engine with consolidation + merge |
| `scripts/job_queue.py` | Persistent job queue with worker daemon |

## Tests

```bash
pytest tests/ -v
```

73 tests, 1 skipped — covers embedder, store, retriever, and consolidator. Uses mock embedders; no real Sentence Transformers needed.

## Architecture

- **TextEmbedder** — Sentence Transformers (`all-MiniLM-L6-v2`, 384-d, L2-normalized)
- **VectorMemoryStore** — FAISS `IndexFlatIP` (falls back to numpy dot-product), pickle persistence
- **MemoryRetriever** — Semantic search + hybrid (semantic × keyword overlap), metadata filters
- **MemoryConsolidator** — Cosine-similarity clustering, multi-strategy pruning (transient first, then by importance with connection boosts), link cleanup

## Roadmap

| Feature | Status |
|---------|--------|
| Auto-Linking (semantic + wikilinks) | ✅ Done |
| Synthesis (RAG Q&A with citations) | ✅ Done |
| Knowledge Graph (typed edges, BFS, backlinks) | ✅ Done |
| Real-time Session Logging | ✅ Done |
| Preference Learning & Consolidation | ✅ Done |
| Identity / Profile System | ✅ Done |
| Batch Relink (`POST /api/relink`) | ✅ Done |
| Auto-start on opencode / Hermes init | ✅ Done |
| Dream Cycle (22-phase overnight maintenance) | ✅ Done |
| Job Queue (crash-safe sub-agents + worker) | ✅ Done |
| Schema Packs (canonical page types) | ⬜ |
| Code Intelligence (Tree-sitter) | ⬜ |
| Ingestion / Import (file watcher, webhooks) | ⬜ |
| Evaluation Framework | ⬜ |
| Calibration (Brier scores, bias tags) | ⬜ |
| Brainstorming (LLM judge with consolidation + merge) | ✅ Done |
| OAuth 2.1 (PKCE + scopes) | ⬜ |
| Skills System (43 curated + optimizer) | ⬜ |
| Multi-Source Federation | ⬜ |
| 40+ MCP Tools | ⬜ |

## License

MIT
