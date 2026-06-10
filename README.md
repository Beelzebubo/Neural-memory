# Neural Memory

Persistent long-term memory system using vector embeddings (numpy/FAISS).

## Architecture

```
Input → Embedder → Store → Retriever → Consolidator → Output
```

- **Embedder** — Converts text into dense vector embeddings using Sentence Transformers (384-d, normalized).
- **Store** — Persists vectors and metadata to pickle, indexed via FAISS (with numpy fallback).
- **Retriever** — Performs semantic and hybrid (semantic + keyword) retrieval.
- **Consolidator** — Deduplicates similar memories, merges clusters, and prunes by age/importance/frequency.

## Quick Start

```bash
pip install -r requirements.txt
python scripts/train.py --data data/texts.jsonl --output data/memory_store.pkl
python scripts/evaluate.py --test-data data/test_queries.json --config config/config.yaml
```

## Usage

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

# Consolidate (dedup + prune)
consolidator = MemoryConsolidator()
consolidator.prune(store, max_size=10000, strategy="hybrid")

# Persist
store.save("data/memory_store.pkl")

# Reload
store.load("data/memory_store.pkl")
```

## Project Structure

```
neural-memory/
├── src/
│   ├── __init__.py        # public API exports
│   ├── embedder.py        # TextEmbedder (Sentence Transformers)
│   ├── memory_store.py    # VectorMemoryStore (FAISS/numpy + pickle)
│   ├── retriever.py       # MemoryRetriever (semantic + hybrid)
│   └── consolidator.py    # MemoryConsolidator (dedup, merge, prune)
├── config/
│   └── config.yaml        # main configuration
├── scripts/
│   ├── train.py           # CLI: embed + store text data
│   └── evaluate.py        # CLI: evaluate retrieval accuracy (Recall@K, MAP, MRR)
├── tests/
│   ├── test_embedder.py
│   ├── test_memory_store.py
│   ├── test_retriever.py
│   └── test_consolidator.py
├── data/                  # persisted memory files (gitignored)
├── logs/                  # application logs (gitignored)
├── .env.example
├── requirements.txt
└── README.md
```

## Configuration

Settings are loaded from `config/config.yaml`:

| Key | Default | Description |
|-----|---------|-------------|
| `embedder.model_name` | `all-MiniLM-L6-v2` | Sentence Transformer model |
| `store.backend` | `faiss` | Backend: `faiss` or numpy (auto-detected) |
| `store.persistence.save_path` | `data/memory_store.pkl` | Pickle file path |
| `retriever.default_k` | `10` | Default top-K results |
| `retriever.hybrid_alpha` | `0.5` | Semantic vs keyword weight |
| `consolidator.similarity_threshold` | `0.85` | Cosine threshold for dedup |
| `consolidator.prune_strategy` | `hybrid` | `by_age`, `by_importance`, `by_access_frequency`, or `hybrid` |
| `consolidator.max_memories` | `10000` | Hard cap on memories |
| `logging.level` | `INFO` | Logging verbosity |

## Tests

```bash
pytest tests/ -v     # 74 tests covering all 4 classes
```

## License

MIT
