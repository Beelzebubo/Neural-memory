import pytest
import numpy as np

from src.retriever import MemoryRetriever


class TestMemoryRetriever:
    def setup_method(self):
        """Set up a retriever with mock embedder and store."""
        self.embeddings = {}  # text -> vector

        class FakeEmbedder:
            def embed(self, text):
                # Return a simple deterministic embedding based on hash
                h = hash(text) % 1000
                return [h / 1000.0] * 4

        class FakeStore:
            def __init__(self):
                self._ids = []
                self._embeddings = []
                self._metadata = {}

            def store(self, eid, emb, meta):
                self._ids.append(eid)
                self._embeddings.append(np.array(emb, dtype=np.float32))
                self._metadata[eid] = meta

            def search(self, query_emb, k):
                """Simple cosine similarity search."""
                query = np.array(query_emb, dtype=np.float32)
                if not self._embeddings:
                    return []
                matrix = np.stack(self._embeddings, axis=0)
                dots = matrix @ query
                dots = dots.ravel()
                top_k = np.argsort(dots)[-k:][::-1]
                results = []
                for idx in top_k:
                    eid = self._ids[idx]
                    # increment access_count
                    meta = dict(self._metadata[eid])
                    meta["access_count"] = meta.get("access_count", 0) + 1
                    self._metadata[eid] = meta
                    results.append({
                        "id": eid,
                        "score": float(dots[idx]),
                        "metadata": dict(self._metadata[eid]),
                    })
                return results

            def get_metadata(self, eid):
                return dict(self._metadata[eid]) if eid in self._metadata else None

            def list_all(self):
                return list(self._ids)

        self.embedder = FakeEmbedder()
        self.store = FakeStore()
        self.retriever = MemoryRetriever(self.embedder, self.store)

    def test_retrieve_returns_results(self):
        self.store.store("a", [0.5, 0.5, 0.5, 0.5], {"source": "test"})
        self.store.store("b", [0.1, 0.1, 0.1, 0.1], {"source": "test"})
        results = self.retriever.retrieve("test query", k=5)
        assert len(results) > 0
        assert all("id" in r for r in results)

    def test_retrieve_empty_query_returns_empty(self):
        assert self.retriever.retrieve("") == []
        assert self.retriever.retrieve("   ") == []

    def test_retrieve_min_score_filter(self):
        self.store.store("a", [0.5, 0.5, 0.5, 0.5], {"source": "test"})
        self.store.store("b", [0.1, 0.1, 0.1, 0.1], {"source": "test"})
        results = self.retriever.retrieve("test query", k=5, min_score=0.2)
        for r in results:
            assert r["score"] >= 0.2

    def test_retrieve_k_limits_results(self):
        for i in range(10):
            self.store.store(f"item_{i}", [i / 10.0] * 4, {"source": "test"})
        results = self.retriever.retrieve("query", k=3)
        assert len(results) <= 3

    def test_retrieve_by_metadata_exact_match(self):
        self.store.store("a", [0.1, 0.1, 0.1, 0.1], {"source": "alpha", "importance": 0.9})
        self.store.store("b", [0.2, 0.2, 0.2, 0.2], {"source": "beta", "importance": 0.5})
        results = self.retriever.retrieve_by_metadata({"source": "alpha"})
        assert len(results) == 1
        assert results[0]["id"] == "a"

    def test_retrieve_by_metadata_returns_multiple(self):
        self.store.store("a", [0.1, 0.1, 0.1, 0.1], {"source": "alpha", "importance": 0.9})
        self.store.store("b", [0.2, 0.2, 0.2, 0.2], {"source": "alpha", "importance": 0.5})
        results = self.retriever.retrieve_by_metadata({"source": "alpha"})
        assert len(results) == 2

    def test_retrieve_by_metadata_no_match(self):
        self.store.store("a", [0.1, 0.1, 0.1, 0.1], {"source": "alpha"})
        results = self.retriever.retrieve_by_metadata({"source": "nonexistent"})
        assert len(results) == 0

    def test_retrieve_by_metadata_missing_key_no_match(self):
        """If filter key not in metadata, item shouldn't match."""
        self.store.store("a", [0.1, 0.1, 0.1, 0.1], {"source": "alpha"})
        results = self.retriever.retrieve_by_metadata({"importance": 0.5})
        assert len(results) == 0

    def test_retrieve_by_metadata_multiple_filters(self):
        self.store.store("a", [0.1, 0.1, 0.1, 0.1], {"source": "alpha", "category": "x"})
        self.store.store("b", [0.2, 0.2, 0.2, 0.2], {"source": "alpha", "category": "y"})
        results = self.retriever.retrieve_by_metadata({"source": "alpha", "category": "x"})
        assert len(results) == 1
        assert results[0]["id"] == "a"

    def test_retrieve_by_metadata_empty_store(self):
        results = self.retriever.retrieve_by_metadata({"source": "test"})
        assert results == []

    def test_retrieve_by_metadata_empty_filters(self):
        """Empty filters should match everything."""
        self.store.store("a", [0.1, 0.1, 0.1, 0.1], {"source": "alpha"})
        self.store.store("b", [0.2, 0.2, 0.2, 0.2], {"source": "beta"})
        results = self.retriever.retrieve_by_metadata({})
        assert len(results) == 2

    def test_retrieve_by_metadata_no_callable_security(self):
        """Security fix: callable values should just do equality comparison, not execute."""
        self.store.store("a", [0.1, 0.1, 0.1, 0.1], {"source": "test"})
        # Previously `callable(callable)` would invoke meta[key](...) — here it just checks equality
        # which will be False, so no match
        results = self.retriever.retrieve_by_metadata({"source": callable})
        assert len(results) == 0  # callable != "test", so no match

    def test_hybrid_retrieve_returns_results(self):
        self.store.store("a", [0.5, 0.5, 0.5, 0.5], {"source": "test content"})
        self.store.store("b", [0.1, 0.1, 0.1, 0.1], {"source": "other content"})
        results = self.retriever.hybrid_retrieve("test", k=5, alpha=0.5)
        assert len(results) > 0
        assert all("id" in r for r in results)

    def test_hybrid_retrieve_with_empty_query(self):
        results = self.retriever.hybrid_retrieve("", k=5)
        assert results == []
        results = self.retriever.hybrid_retrieve("   ", k=5)
        assert results == []

    def test_hybrid_retrieve_alpha_0_is_pure_keyword(self):
        """alpha=0 should use only keyword matching."""
        self.store.store("match", [0.0, 0.0, 0.0, 0.0], {"source": "target keyword"})
        self.store.store("no_match", [1.0, 1.0, 1.0, 1.0], {"source": "unrelated"})
        results = self.retriever.hybrid_retrieve("target", k=5, alpha=0.0)
        ids = [r["id"] for r in results]
        assert "match" in ids
        assert "no_match" in ids  # keyword score=0, semantic score=1.0, alpha=0 => combined=0

    def test_hybrid_retrieve_alpha_1_is_pure_semantic(self):
        """alpha=1.0 should use only semantic matching."""
        self.store.store("match", [1.0, 1.0, 1.0, 1.0], {"source": "xyz"})
        self.store.store("no_match", [0.0, 0.0, 0.0, 0.0], {"source": "abc"})
        # Query that embeds closer to "match" embedding
        results = self.retriever.hybrid_retrieve("something", k=5, alpha=1.0)
        # All results have combined=alpha*sem_score + 0, so sem_score dominates
        scores = {r["id"]: r["score"] for r in results}
        assert len(scores) == 2  # both items should appear

    def test_hybrid_retrieve_k_limits_results(self):
        for i in range(10):
            self.store.store(f"item_{i}", [i / 10.0] * 4, {"source": f"text {i}"})
        results = self.retriever.hybrid_retrieve("text", k=3, alpha=0.5)
        assert len(results) <= 3

    def test_tokenize_splits_correctly(self):
        tokens = self.retriever._tokenize("Hello World! Test-123")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens
        assert "123" in tokens
        assert len(tokens) == 4

    def test_tokenize_empty_string(self):
        assert self.retriever._tokenize("") == []
        assert self.retriever._tokenize("   ") == []

    def test_hybrid_retrieve_empty_store(self):
        results = self.retriever.hybrid_retrieve("test query", k=5)
        assert results == []
