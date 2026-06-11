# Lino — Neural Memory Server

A self-hosted semantic memory server. Turns notes, conversations, and knowledge into persistent, searchable vector embeddings — accessible through a web UI, REST API, MCP tools, and CLI.

## What it does

Four layers that stack together:

- **Embedder** — Runs text through Sentence Transformers (384-d, normalized) and gives you vectors back.
- **Store** — Keeps those vectors in a FAISS index (or plain numpy if you don't have FAISS), saves to pickle.
- **Retriever** — Finds what's relevant: straight semantic search or a hybrid that mixes semantic with keyword overlap.
- **Consolidator** — Finds similar entries by cosine similarity, merges them, prunes stale ones by age, importance, or access count.

## Quick Start

```bash
pip install -r requirements.txt
cd scripts

# Dev mode (hot reload)
./start.sh dev
# or: python -m uvicorn ui.app:app --host 127.0.0.1 --port 8210 --reload

# Seed from Obsidian vault
./seed.sh

# Backup
./backup.sh
```

## Web UI

FastAPI server at `ui/app.py` with a dark-themed SPA. Browse, search, filter, create, delete, back up, prune memories, and **adjust priority sliders** with auto-save debounce.

```bash
python -m uvicorn ui.app:app --host 127.0.0.1 --port 8210
```

## New Features

### Priority Slider (Web UI)
Each memory in the web UI has a priority slider (0.0–1.0). Drag to adjust importance — changes auto-save after 500ms debounce with a save indicator. No re-embedding needed; only metadata updates.

### Real-Time Vault Watchdog
Monitors your Obsidian vault for Markdown changes and syncs them to the vector store automatically:

```bash
python scripts/watchdog_sync.py        # Start in foreground
python integration/cli.py watchdog start  # Start as daemon
python integration/cli.py watchdog status
python integration/cli.py watchdog stop
```

### Rip & Compress Pipeline
Background cron job that compresses transient/low-priority memories via LLM:

```bash
python scripts/rip_and_compress.py --dry-run          # Preview only
python integration/cli.py compress --dry-run
python integration/cli.py compress --min-age 48 --provider groq
```

### Vault Sync with Wikilinks
One-way vault→neural-memory sync that generates Obsidian wikilinks (`[[memory_id]]`) in YAML frontmatter:

```bash
python integration/cli.py sync
python integration/cli.py sync --no-link              # Skip wikilink gen
```

## Using it from code

```python
from src import TextEmbedder, VectorMemoryStore, MemoryRetriever, MemoryConsolidator

embedder = TextEmbedder()
store = VectorMemoryStore()

# Store a memory
emb = embedder.embed("The user prefers dark mode.")
store.store("mem_001", emb, {"source": "conversation", "importance_score": 0.8})

# Retrieve relevant memories
retriever = MemoryRetriever(embedder, store)
results = retriever.retrieve("What theme do they like?", k=5)

# Clean up
consolidator = MemoryConsolidator()
consolidator.prune(store, max_size=10000, strategy="hybrid")

# Save and reload
store.save("data/memory_store.pkl")
store.load("data/memory_store.pkl")
```

## Agent Integration

This project works with any coding agent via MCP, REST API, direct Python, or CLI.

### Option 1: MCP — works with most agents

The MCP server (`integration/mcp_server.py`) exposes **11 tools**: `neural_memory_store`, `neural_memory_search`, `neural_memory_list`, `neural_memory_get`, `neural_memory_delete`, `neural_memory_stats`, `neural_memory_prune`, `neural_memory_update_priority`, `neural_memory_run_sync`, `neural_memory_run_compress`, `neural_memory_watchdog` — over stdio JSON-RPC.

**Hermes Agent**
```bash
hermes mcp add lino --command "python3 /path/to/lino/integration/mcp_server.py"
```

**Claude Code** — add to `~/.claude/.mcp.json`:
```json
{
  "mcpServers": {
    "lino": {
      "command": "python3",
      "args": ["/path/to/lino/integration/mcp_server.py"]
    }
  }
}
```

**Cline / Roo Code / Continue** — add to the agent's MCP config:
```json
{
  "mcpServers": {
    "lino": {
      "command": "python3",
      "args": ["/path/to/lino/integration/mcp_server.py"]
    }
  }
}
```

**Aider** — add to `.aider.mcp.json`:
```json
{
  "mcpServers": {
    "lino": {
      "command": "python3",
      "args": ["/path/to/lino/integration/mcp_server.py"]
    }
  }
}
```

### Option 2: REST API — works with any agent that has HTTP tools

The web UI doubles as a REST API. Start the server:

```bash
python -m uvicorn ui.app:app --host 127.0.0.1 --port 8210
```

Then agents can call it directly:

```bash
# Store a memory
curl -X POST http://localhost:8210/api/store \
  -H "Content-Type: application/json" \
  -d '{"text": "User prefers dark mode", "source": "conversation", "importance": 0.8}'

# Update priority
curl -X PATCH http://localhost:8210/api/memories/memory_01/priority \
  -H "Content-Type: application/json" \
  -d '{"priority": 0.95}'

# Search
curl "http://localhost:8210/api/search?q=dark+mode&k=5"

# List all
curl "http://localhost:8210/api/list?limit=50"

# Stats
curl "http://localhost:8210/api/stats"
```

Works with **Codex CLI**, **GitHub Copilot**, **OpenAI functions**, or any agent that can make HTTP requests.

### Option 3: Direct Python import — works everywhere

The core library is pure Python with no agent dependencies:

```python
from src import TextEmbedder, VectorMemoryStore, MemoryRetriever

embedder = TextEmbedder()
store = VectorMemoryStore()
store.load("data/memory_store.pkl")

# Search across sessions
query_emb = embedder.embed("What does the user like?")
results = store.search(query_emb, k=5)
for r in results:
    print(f"{r['id']} (score: {r['score']:.3f}): {r['metadata'].get('text', '')}")
```

### Option 4: CLI tool — works in any shell

Standalone CLI at `integration/cli.py`:

```bash
python integration/cli.py store "User prefers dark mode" --source conversation
python integration/cli.py search "dark mode" -k 5
python integration/cli.py list --source conversation
python integration/cli.py stats
python integration/cli.py priority-update memory_01 --priority 0.95
python integration/cli.py sync [--no-link] [--max-related 5]
python integration/cli.py compress [--dry-run] [--min-age 24]
python integration/cli.py watchdog start|stop|status
```

## Project Structure

```
lino/
├── src/                     # Core library
│   ├── embedder.py          # TextEmbedder
│   ├── memory_store.py      # VectorMemoryStore
│   ├── retriever.py         # MemoryRetriever
│   └── consolidator.py      # MemoryConsolidator
├── ui/                      # FastAPI web interface
│   ├── app.py               # PATCH /api/memories/{id}/priority endpoint
│   ├── templates/index.html # Priority slider component
│   └── static/css/ + js/    # Debounce auto-save slider
├── integration/             # Hermes / MCP integration
│   ├── hermes_plugin.py     # 11 tool schemas + 4 new cmd methods
│   ├── mcp_server.py        # 11 MCP tools
│   └── cli.py               # Full CLI with 8 subcommands
├── scripts/
│   ├── rip_and_compress.py  # LLM compression pipeline
│   ├── watchdog_sync.py     # Real-time vault file watcher
│   ├── start.sh / stop.sh   # Dev/prod lifecycle
│   ├── seed.sh              # Obsidian vault import
│   ├── backup.sh / restore.sh
│   └── healthcheck.sh       # Web UI + API health
├── config/config.yaml
├── tests/                   # 74 tests
├── Makefile                 # Convenience commands
├── Dockerfile / docker-compose.yml
├── nginx/                   # Reverse proxy config
├── systemd/                 # Auto-start on boot
├── data/                    # Persisted stores (gitignored)
└── requirements.txt
```

## Configuration

Everything lives in `config/config.yaml`:

| Key | Default | Description |
|-----|---------|-------------|
| `embedder.model_name` | `all-MiniLM-L6-v2` | Sentence Transformer model |
| `store.backend` | `faiss` | `faiss` or numpy (auto-detected) |
| `store.persistence.save_path` | `data/memory_store.pkl` | Where the pickle goes |
| `retriever.default_k` | `10` | Default top-K |
| `retriever.hybrid_alpha` | `0.5` | Semantic vs keyword weight |
| `consolidator.similarity_threshold` | `0.85` | Cosine threshold for dedup |
| `consolidator.prune_strategy` | `hybrid` | `by_age`, `by_importance`, `by_access_frequency`, or `hybrid` |
| `consolidator.max_memories` | `10000` | Hard cap |
| `logging.level` | `INFO` | Verbosity |

## Tests

```bash
pytest tests/ -v     # 74 tests
```

## License

MIT
