# Neural Memory

A semantic memory system for AI agents. Takes text in, turns it into vectors, stores and retrieves them, cleans up duplicates, and has a web UI on top so you can actually see what's in there.

## What it does

Four layers that stack together:

- **Embedder** — Runs text through Sentence Transformers (384-d, normalized) and gives you vectors back.
- **Store** — Keeps those vectors in a FAISS index (or plain numpy if you don't have FAISS), saves to pickle.
- **Retriever** — Finds what's relevant: straight semantic search or a hybrid that mixes semantic with keyword overlap.
- **Consolidator** — Finds similar entries by cosine similarity, merges them, prunes stale ones by age, importance, or access count.

## Quick Start

```bash
pip install -r requirements.txt
python scripts/train.py --data data/texts.jsonl --output data/memory_store.pkl
python scripts/evaluate.py --test-data data/test_queries.json --config config/config.yaml
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

## Web UI

There's a FastAPI server at `ui/app.py` with a dark-themed SPA. Lets you browse, search, filter, create, delete, back up, and prune memories through a browser.

```bash
python -m uvicorn ui.app:app --host 127.0.0.1 --port 8210
```

## Agent Integration

The project works with any coding agent. You have three options depending on what your agent supports.

### Option 1: MCP (Model Context Protocol) — works with most agents

The MCP server (`integration/mcp_server.py`) exposes 7 tools — `neural_memory_store`, `neural_memory_search`, `neural_memory_list`, `neural_memory_get`, `neural_memory_delete`, `neural_memory_stats`, `neural_memory_prune` — over stdio JSON-RPC. Any agent that supports MCP can use them.

**Hermes Agent**
```bash
hermes mcp add neural-memory --command "python3 /path/to/neural-memory/integration/mcp_server.py"
```

**Claude Code** — add to `~/.claude/.mcp.json`:
```json
{
  "mcpServers": {
    "neural-memory": {
      "command": "python3",
      "args": ["/path/to/neural-memory/integration/mcp_server.py"]
    }
  }
}
```

**Cline / Roo Code / Continue** — add to the agent's MCP config:
```json
{
  "mcpServers": {
    "neural-memory": {
      "command": "python3",
      "args": ["/path/to/neural-memory/integration/mcp_server.py"]
    }
  }
}
```

**Aider** — add to `.aider.mcp.json`:
```json
{
  "mcpServers": {
    "neural-memory": {
      "command": "python3",
      "args": ["/path/to/neural-memory/integration/mcp_server.py"]
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

# Search
curl "http://localhost:8210/api/search?q=dark+mode&k=5"

# List all
curl "http://localhost:8210/api/list?limit=50"

# Stats
curl "http://localhost:8210/api/stats"
```

This works with **Codex CLI**, **GitHub Copilot**, **OpenAI functions**, or any agent that can make HTTP requests.

### Option 3: Direct Python import — works everywhere

The core library is pure Python with no agent dependencies. Any agent running Python code can use it:

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

There's a standalone CLI at `integration/cli.py`:

```bash
python integration/cli.py store "User prefers dark mode" --source conversation
python integration/cli.py search "dark mode" -k 5
python integration/cli.py list --source conversation
python integration/cli.py stats
```

## Project Structure

```
neural-memory/
├── src/                     # Core library
│   ├── embedder.py          # TextEmbedder
│   ├── memory_store.py      # VectorMemoryStore
│   ├── retriever.py         # MemoryRetriever
│   └── consolidator.py      # MemoryConsolidator
├── ui/                      # FastAPI web interface
│   ├── app.py
│   ├── templates/index.html
│   └── static/css/ + js/
├── integration/             # Hermes / MCP integration
│   ├── hermes_plugin.py
│   ├── mcp_server.py
│   └── cli.py
├── config/config.yaml
├── scripts/train.py         # Bulk import from JSONL
├── scripts/evaluate.py      # Recall@K, MAP, MRR
├── tests/                   # 74 tests
├── data/                    # persisted stores (gitignored)
├── .env.example
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
