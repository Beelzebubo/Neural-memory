#!/usr/bin/env python3
"""End-to-end neural memory demo using real sentence-transformers embeddings."""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

import numpy as np
from src import TextEmbedder, VectorMemoryStore, MemoryRetriever, MemoryConsolidator

print("=" * 72)
print("NEURAL MEMORY — REAL EMBEDDER DEMO (all-MiniLM-L6-v2, 384-d)")
print("=" * 72)

# ── Phase 1: Load real embedder ──
print("\n[1/6] Loading sentence-transformers model (all-MiniLM-L6-v2)...")
t0 = time.time()
embedder = TextEmbedder(model_name="all-MiniLM-L6-v2")
t1 = time.time()
print(f"     Model loaded in {t1-t0:.1f}s — dim={len(embedder)}")

# ── Phase 2: Store diverse memories ──
print("\n[2/6] Embedding and storing 6 memories...")
store = VectorMemoryStore()

memories = [
    ("mem_001", "The user prefers dark mode in their code editor and IDE", "conversation", 0.90),
    ("mem_002", "Project uses Python 3.11 with FastAPI backend and PostgreSQL", "project_scan", 0.70),
    ("mem_003", "User's favorite color is blue, specifically navy and midnight blue", "preferences", 0.80),
    ("mem_004", "API endpoints follow REST conventions with OpenAPI documentation", "codebase", 0.60),
    ("mem_005", "Dark mode reduces eye strain for developers working at night", "knowledge", 0.85),
    ("mem_006", "User deployed the application on a Linux server running Fedora", "deployment", 0.75),
]

for eid, text, source, importance in memories:
    t0 = time.time()
    emb = embedder.embed(text)
    t_emb = (time.time() - t0) * 1000
    store.store(eid, emb, {
        "source": source,
        "importance_score": importance,
        "access_count": 0,
    })
    print(f"     {eid}: \"{text[:50]}...\"  [{t_emb:.0f}ms]  imp={importance}")

print(f"\n     Store: {len(store)} entries, dim={store._dim}")

# ── Phase 3: Semantic search ──
print("\n[3/6] Semantic search...")
queries = [
    "What theme do they like for coding?",
    "Tell me about the API structure",
    "Where is this deployed?",
]

for query in queries:
    emb = embedder.embed(query)
    results = store.search(emb, k=3)
    print(f"\n     Query: \"{query}\"")
    for i, r in enumerate(results):
        m = r["metadata"]
        print(f"       [{i+1}] {r['id']}  score={r['score']:.4f}  "
              f"source={m.get('source')}  imp={m.get('importance_score')}")
    # Show access_count bumped on each search
    for r in results:
        m = store._metadata[r["id"]]
        print(f"             → access_count now = {m['access_count']}")

# ── Phase 4: Metadata + hybrid retrieval ──
print("\n[4/6] Metadata filter + hybrid retrieval...")
retriever = MemoryRetriever(embedder, store)

# Metadata filter
meta_results = retriever.retrieve_by_metadata({"source": "conversation"})
print(f"     Filter by source='conversation': {len(meta_results)} result(s)")
for r in meta_results:
    m = r["metadata"]
    print(f"       {r['id']} — source={m.get('source')}, imp={m.get('importance_score')}")

# Hybrid
hybrid_results = retriever.hybrid_retrieve("code dark theme", k=3, alpha=0.7)
print(f"\n     Hybrid search 'code dark theme' (α=0.7):")
for r in hybrid_results:
    print(f"       {r['id']}  score={r['score']:.4f}")

# Pure keyword (alpha=0)
kw_results = retriever.hybrid_retrieve("night Fedora", k=3, alpha=0.0)
print(f"\n     Pure keyword 'night Fedora' (α=0.0):")
for r in kw_results:
    print(f"       {r['id']}  score={r['score']:.4f}")

# ── Phase 5: Consolidation (prune) ──
print("\n[5/6] Consolidation — prune by importance (keep top 3)...")
consolidator = MemoryConsolidator()

# Snapshot before
before_ids = store.list_all()
print(f"     Before: {len(before_ids)} memories: {before_ids}")

pruned = consolidator.prune(store, max_size=3, strategy="by_importance")
after_ids = store.list_all()
print(f"     Pruned {pruned} — kept {len(after_ids)}: {after_ids}")

# ── Phase 6: Persist and reload ──
print("\n[6/6] Persistence round-trip...")
pkl_path = Path("/tmp/neural_memory_demo.pkl")
store.save(str(pkl_path))
print(f"     Saved {pkl_path} ({pkl_path.stat().st_size:,} bytes)")

store2 = VectorMemoryStore()
store2.load(str(pkl_path))
print(f"     Reloaded: {len(store2)} entries, dim={store2._dim}")

# Verify
assert list(store2.list_all()) == list(store.list_all()), "ID mismatch after reload"
for eid in store2.list_all():
    orig_m = store._metadata[eid]
    load_m = store2._metadata[eid]
    assert orig_m["importance_score"] == load_m["importance_score"]
    assert orig_m["source"] == load_m["source"]

# Cleanup
pkl_path.unlink()
print("     Round-trip verified ✓")

print("\n" + "=" * 72)
print("FULL DEMO PASSED — real sentence-transformers, real FAISS/numpy store")
print("=" * 72)
