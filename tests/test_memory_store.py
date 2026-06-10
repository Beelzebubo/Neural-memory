import pickle
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from src.memory_store import VectorMemoryStore


@pytest.fixture
def store():
    return VectorMemoryStore()


def make_embedding(dim: int = 4, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    vec = rng.random(dim).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec.tolist()


class TestVectorMemoryStore:
    def test_store_and_len(self, store):
        assert len(store) == 0
        store.store("id1", make_embedding(seed=1), {"source": "test"})
        assert len(store) == 1
        store.store("id2", make_embedding(seed=2), {"source": "test"})
        assert len(store) == 2

    def test_search_returns_correct_number_of_results(self, store):
        dim = 4
        for i in range(10):
            store.store(f"id{i}", make_embedding(dim, seed=i), {"source": "test"})

        query = make_embedding(dim, seed=99)
        results = store.search(query, k=3)
        assert len(results) == 3

    def test_search_results_are_sorted_by_score(self, store):
        dim = 4
        for i in range(5):
            store.store(f"id{i}", make_embedding(dim, seed=i), {"source": "test"})

        query = make_embedding(dim, seed=99)
        results = store.search(query, k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_k_greater_than_count(self, store):
        for i in range(3):
            store.store(f"id{i}", make_embedding(seed=i), {"source": "test"})

        query = make_embedding(seed=99)
        results = store.search(query, k=10)
        assert len(results) == 3

    def test_search_with_empty_store_returns_empty_list(self, store):
        query = make_embedding()
        results = store.search(query, k=5)
        assert results == []

    def test_delete_removes_item(self, store):
        store.store("id1", make_embedding(seed=1), {"source": "test"})
        store.store("id2", make_embedding(seed=2), {"source": "test"})
        assert len(store) == 2

        store.delete("id1")
        assert len(store) == 1
        assert store.list_all() == ["id2"]

    def test_delete_nonexistent_id_does_nothing(self, store):
        store.store("id1", make_embedding(seed=1), {"source": "test"})
        store.delete("nonexistent")
        assert len(store) == 1

    def test_list_all_returns_all_ids(self, store):
        ids = [f"id{i}" for i in range(5)]
        for i, id_ in enumerate(ids):
            store.store(id_, make_embedding(seed=i), {"source": "test"})

        assert sorted(store.list_all()) == sorted(ids)

    def test_list_all_empty_store(self, store):
        assert store.list_all() == []

    def test_save_and_load_round_trip(self, store):
        dim = 4
        for i in range(3):
            store.store(f"id{i}", make_embedding(dim, seed=i), {"source": "test", "val": i})

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
            store.save(path)

        loaded = VectorMemoryStore()
        loaded.load(path)

        assert len(loaded) == 3
        assert sorted(loaded.list_all()) == ["id0", "id1", "id2"]

        for i in range(3):
            results = loaded.search(make_embedding(dim, seed=i), k=1)
            assert results[0]["id"] == f"id{i}"

        Path(path).unlink()

    def test_save_and_load_empty_store(self, store):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
            store.save(path)

        loaded = VectorMemoryStore()
        loaded.load(path)
        assert len(loaded) == 0

        Path(path).unlink()

    def test_metadata_tracks_timestamp(self, store):
        before = time.time()
        store.store("id1", make_embedding(seed=1), {"source": "doc"})
        after = time.time()

        results = store.search(make_embedding(seed=1), k=1)
        ts = results[0]["metadata"]["timestamp"]
        assert before <= ts <= after

    def test_metadata_tracks_source(self, store):
        store.store("id1", make_embedding(seed=1), {"source": "user_query"})
        results = store.search(make_embedding(seed=1), k=1)
        assert results[0]["metadata"]["source"] == "user_query"

    def test_metadata_default_source(self, store):
        store.store("id1", make_embedding(seed=1), {})
        results = store.search(make_embedding(seed=1), k=1)
        assert results[0]["metadata"]["source"] == "unknown"

    def test_metadata_tracks_importance_score(self, store):
        store.store("id1", make_embedding(seed=1), {"importance_score": 0.9})
        results = store.search(make_embedding(seed=1), k=1)
        assert results[0]["metadata"]["importance_score"] == 0.9

    def test_metadata_default_importance_score(self, store):
        store.store("id1", make_embedding(seed=1), {})
        results = store.search(make_embedding(seed=1), k=1)
        assert results[0]["metadata"]["importance_score"] == 0.5

    def test_metadata_tracks_access_count(self, store):
        store.store("id1", make_embedding(seed=1), {"access_count": 3})
        results = store.search(make_embedding(seed=1), k=1)
        assert results[0]["metadata"]["access_count"] == 4

    def test_access_count_increments_on_search(self, store):
        store.store("id1", make_embedding(seed=1), {})
        store.search(make_embedding(seed=1), k=1)
        results = store.search(make_embedding(seed=1), k=1)
        assert results[0]["metadata"]["access_count"] == 2

    def test_store_with_different_dimensions(self, store):
        store.store("id1", make_embedding(dim=8, seed=1), {"source": "test"})
        assert len(store) == 1

    def test_save_pickle_format(self, store):
        store.store("id1", make_embedding(seed=1), {"source": "test"})

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
            store.save(path)

        with open(path, "rb") as f:
            data = pickle.load(f)

        assert "ids" in data
        assert "embeddings" in data
        assert "metadata" in data
        assert "dim" in data
        assert data["ids"] == ["id1"]

        Path(path).unlink()
