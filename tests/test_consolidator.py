import pytest
import numpy as np

from src.consolidator import MemoryConsolidator


class TestMemoryConsolidatorConsolidate:
    def setup_method(self):
        self.consolidator = MemoryConsolidator()

    def test_empty_list_returns_empty(self):
        assert self.consolidator.consolidate([]) == []

    def test_single_memory_passes_through(self):
        mem = {"text": "hello", "embedding": [0.1, 0.2, 0.3], "metadata": {}}
        result = self.consolidator.consolidate([mem])
        assert len(result) == 1
        assert result[0]["text"] == "hello"

    def test_items_without_embedding_are_preserved(self):
        """Bug fix: items without 'embedding' key were silently dropped."""
        memories = [
            {"text": "a", "embedding": [0.1, 0.2, 0.3], "metadata": {}},
            {"text": "b", "metadata": {}},  # no embedding
            {"text": "c", "embedding": [0.4, 0.5, 0.6], "metadata": {}},
        ]
        result = self.consolidator.consolidate(memories, threshold=0.99)
        texts = {m["text"] for m in result}
        assert "a" in texts
        assert "b" in texts
        assert "c" in texts
        assert len(result) == 3

    def test_similar_embeddings_get_merged(self):
        """Embeddings with high cosine similarity should be clustered."""
        # Use normalized unit-length vectors so raw dot product ≈ cosine similarity
        import math
        n1 = math.sqrt(0.1**2 + 0.2**2 + 0.3**2)
        n2 = math.sqrt(0.11**2 + 0.21**2 + 0.31**2)
        n3 = math.sqrt(0.9**2 + 0.8**2 + 0.7**2)
        memories = [
            {"text": "hello world", "embedding": [0.1/n1, 0.2/n1, 0.3/n1], "metadata": {"source": "test"}},
            {"text": "hello there", "embedding": [0.11/n2, 0.21/n2, 0.31/n2], "metadata": {"source": "test"}},
            {"text": "completely different topic", "embedding": [0.9/n3, 0.8/n3, 0.7/n3], "metadata": {"source": "other"}},
        ]
        result = self.consolidator.consolidate(memories, threshold=0.95)
        # First two (nearly identical direction) should be merged, third stays separate
        assert len(result) == 2

    def test_low_threshold_never_merges(self):
        memories = [
            {"text": "hello", "embedding": [0.1, 0.2, 0.3], "metadata": {}},
            {"text": "world", "embedding": [0.4, 0.5, 0.6], "metadata": {}},
        ]
        result = self.consolidator.consolidate(memories, threshold=1.0)
        assert len(result) == 2

    def test_all_items_without_embedding_returns_all(self):
        memories = [
            {"text": "a", "metadata": {}},
            {"text": "b", "metadata": {}},
        ]
        result = self.consolidator.consolidate(memories)
        assert len(result) == 2

    def test_merged_entry_has_metadata(self):
        import math
        n1 = math.sqrt(0.1**2 + 0.2**2 + 0.3**2)
        n2 = math.sqrt(0.11**2 + 0.21**2 + 0.31**2)
        memories = [
            {"text": "dup a", "embedding": [0.1/n1, 0.2/n1, 0.3/n1], "metadata": {"source": "s1", "importance_score": 0.8}},
            {"text": "dup b", "embedding": [0.11/n2, 0.21/n2, 0.31/n2], "metadata": {"source": "s1", "importance_score": 0.6}},
        ]
        result = self.consolidator.consolidate(memories, threshold=0.95)
        assert len(result) == 1
        assert "metadata" in result[0]
        assert result[0]["metadata"]["source"] == "s1"
        # Importance should be the max
        assert result[0]["metadata"]["importance_score"] == 0.8


class TestMemoryConsolidatorSummarize:
    def setup_method(self):
        self.consolidator = MemoryConsolidator()

    def test_empty_memories_returns_message(self):
        assert "No memories" in self.consolidator.summarize([])

    def test_single_memory_summary(self):
        mems = [{"metadata": {"source": "test", "importance_score": 0.7, "timestamp": 1000000}}]
        summary = self.consolidator.summarize(mems)
        assert "1 memories" in summary
        assert "test" in summary
        assert "0.70" in summary

    def test_summary_without_metadata(self):
        mems = [{"text": "hello"}]
        summary = self.consolidator.summarize(mems)
        assert mems[0]["text"] not in summary  # text not in summary
        assert "1 memories" in summary

    def test_date_range_in_summary(self):
        mems = [
            {"metadata": {"source": "a", "timestamp": 1000000}},
            {"metadata": {"source": "b", "timestamp": 2000000}},
        ]
        summary = self.consolidator.summarize(mems)
        assert "Date range" in summary
        assert "1970" in summary  # both timestamps are in 1970


class TestMemoryConsolidatorPrune:
    def setup_method(self):
        self.consolidator = MemoryConsolidator()

    def make_store(self, items):
        """Helper to build a VectorMemoryStore with test data."""
        from src.memory_store import VectorMemoryStore
        store = VectorMemoryStore()
        for i, (eid, ts, imp, acc) in enumerate(items):
            emb = [0.1 * i, 0.2 * i, 0.3 * i]
            store.store(eid, emb, {
                "timestamp": ts,
                "importance_score": imp,
                "access_count": acc,
            })
        return store

    def test_no_prune_when_under_max(self):
        store = self.make_store([("a", 100, 0.5, 0), ("b", 200, 0.5, 0)])
        assert self.consolidator.prune(store, 10) == 0
        assert len(store) == 2

    def test_prune_exact_count(self):
        store = self.make_store([("a", 100, 0.5, 0), ("b", 200, 0.5, 0), ("c", 300, 0.5, 0)])
        assert self.consolidator.prune(store, 2) == 1
        assert len(store) == 2

    def test_prune_by_age_removes_oldest(self):
        store = self.make_store([
            ("old", 100, 0.5, 0),
            ("mid", 200, 0.5, 0),
            ("new", 300, 0.5, 0),
        ])
        self.consolidator.prune(store, 2, strategy="by_age")
        remaining = store.list_all()
        assert "old" not in remaining

    def test_prune_by_importance_removes_least_important(self):
        store = self.make_store([
            ("low", 100, 0.2, 0),
            ("mid", 200, 0.5, 0),
            ("high", 300, 0.9, 0),
        ])
        self.consolidator.prune(store, 2, strategy="by_importance")
        remaining = store.list_all()
        assert "low" not in remaining

    def test_prune_by_access_frequency_removes_least_accessed(self):
        store = self.make_store([
            ("rare", 100, 0.5, 1),
            ("often", 200, 0.5, 10),
            ("medium", 300, 0.5, 5),
        ])
        self.consolidator.prune(store, 2, strategy="by_access_frequency")
        remaining = store.list_all()
        assert "rare" not in remaining

    def test_hybrid_prune_removes_oldest_when_tie(self):
        """Hybrid strategy: sort by importance, access_count, then timestamp (ascending = oldest first)."""
        store = self.make_store([
            ("old", 100, 0.5, 0),
            ("new", 300, 0.5, 0),
        ])
        self.consolidator.prune(store, 1, strategy="hybrid")
        remaining = store.list_all()
        assert "old" not in remaining  # oldest removed first when importance/access tied

    def test_prune_on_empty_store(self):
        from src.memory_store import VectorMemoryStore
        store = VectorMemoryStore()
        assert self.consolidator.prune(store, 10) == 0


class TestMemoryConsolidatorMergeCluster:
    def setup_method(self):
        self.consolidator = MemoryConsolidator()

    def test_single_item_returns_unchanged(self):
        item = {"text": "hello", "embedding": [0.1, 0.2, 0.3], "metadata": {"source": "test"}}
        result = self.consolidator._merge_cluster([item])
        assert result["text"] == "hello"
        assert result["embedding"] == [0.1, 0.2, 0.3]

    def test_merge_deduplicates_text(self):
        items = [
            {"text": "hello", "embedding": [0.1, 0.2, 0.3], "metadata": {"source": "s1"}},
            {"text": "hello", "embedding": [0.11, 0.21, 0.31], "metadata": {"source": "s1"}},
        ]
        result = self.consolidator._merge_cluster(items)
        # same text should be deduplicated
        assert result["text"] == "hello"

    def test_merge_uses_first_source(self):
        items = [
            {"text": "a", "embedding": [0.1, 0.2, 0.3], "metadata": {"source": "first"}},
            {"text": "b", "embedding": [0.11, 0.21, 0.31], "metadata": {"source": "second"}},
        ]
        result = self.consolidator._merge_cluster(items)
        assert result["metadata"]["source"] == "first"

    def test_merge_takes_max_importance_and_sum_access(self):
        items = [
            {"text": "a", "embedding": [0.1, 0.2, 0.3], "metadata": {"importance_score": 0.5, "access_count": 3}},
            {"text": "b", "embedding": [0.11, 0.21, 0.31], "metadata": {"importance_score": 0.9, "access_count": 7}},
        ]
        result = self.consolidator._merge_cluster(items)
        assert result["metadata"]["importance_score"] == 0.9  # max
        assert result["metadata"]["access_count"] == 10  # sum

    def test_merge_tracks_merged_count(self):
        items = [
            {"text": "a", "embedding": [0.1, 0.2, 0.3], "metadata": {}},
            {"text": "b", "embedding": [0.11, 0.21, 0.31], "metadata": {}},
            {"text": "c", "embedding": [0.12, 0.22, 0.32], "metadata": {}},
        ]
        result = self.consolidator._merge_cluster(items)
        assert result["metadata"]["merged_from"] == 3

    def test_merge_uses_first_embedding(self):
        items = [
            {"text": "a", "embedding": [0.1, 0.2, 0.3], "metadata": {}},
            {"text": "b", "embedding": [0.11, 0.21, 0.31], "metadata": {}},
        ]
        result = self.consolidator._merge_cluster(items)
        assert result["embedding"] == [0.1, 0.2, 0.3]

    def test_merge_without_embedding_key(self):
        items = [
            {"text": "a", "metadata": {}},
            {"text": "b", "metadata": {}},
        ]
        result = self.consolidator._merge_cluster(items)
        assert "embedding" not in result
