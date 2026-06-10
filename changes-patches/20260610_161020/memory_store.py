import pickle
import time
from typing import Dict, List, Optional, Tuple

import numpy as np


class VectorMemoryStore:
    def __init__(self):
        self._ids: List[str] = []
        self._embeddings: List[np.ndarray] = []
        self._metadata: Dict[str, dict] = {}
        self._index = None
        self._use_faiss = False
        self._dim = 0

    def _try_init_faiss(self, dim: int):
        try:
            import faiss
            self._index = faiss.IndexFlatIP(dim)
            self._use_faiss = True
        except ImportError:
            self._use_faiss = False

    def store(self, id: str, embedding: List[float], metadata: dict):
        vec = np.array(embedding, dtype=np.float32)
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
        dim = vec.shape[1]

        if not self._ids:
            self._dim = dim
            self._try_init_faiss(dim)

        if self._use_faiss:
            self._index.add(vec)
        self._embeddings.append(vec.ravel().copy())
        self._ids.append(id)

        default_meta = {
            "timestamp": time.time(),
            "source": metadata.get("source", "unknown"),
            "importance_score": metadata.get("importance_score", 0.5),
            "access_count": metadata.get("access_count", 0),
        }
        default_meta.update(metadata)
        self._metadata[id] = default_meta

    def search(self, query_embedding: List[float], k: int) -> List[dict]:
        if not self._ids:
            return []
        k = min(k, len(self._ids))
        query = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

        if self._use_faiss:
            scores, indices = self._index.search(query, k)
        else:
            matrix = np.stack(self._embeddings, axis=0)
            dots = matrix @ query.T
            dots = dots.ravel()
            top_k = np.argsort(dots)[-k:][::-1]
            indices = top_k.reshape(1, -1)
            scores = dots[indices[0]].reshape(1, -1)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            entry_id = self._ids[idx]
            self._metadata[entry_id]["access_count"] = (
                self._metadata[entry_id].get("access_count", 0) + 1
            )
            results.append({
                "id": entry_id,
                "score": float(score),
                "metadata": dict(self._metadata[entry_id]),
            })
        return results

    def delete(self, id: str):
        if id not in self._metadata:
            return
        try:
            idx = self._ids.index(id)
        except ValueError:
            del self._metadata[id]
            return

        self._ids.pop(idx)
        self._embeddings.pop(idx)
        del self._metadata[id]

        if self._use_faiss:
            import faiss
            new_index = faiss.IndexFlatIP(self._dim)
            if self._embeddings:
                new_index.add(np.stack(self._embeddings, axis=0))
            self._index = new_index

    def list_all(self) -> List[str]:
        return list(self._ids)

    def save(self, path: str):
        data = {
            "ids": self._ids,
            "embeddings": np.stack(self._embeddings, axis=0) if self._embeddings else np.array([]),
            "metadata": self._metadata,
            "dim": self._dim,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._ids = data["ids"]
        self._embeddings = [data["embeddings"][i] for i in range(len(self._ids))]
        self._metadata = data["metadata"]
        self._dim = data["dim"]

        self._try_init_faiss(self._dim)
        if self._use_faiss and self._embeddings:
            self._index.add(np.stack(self._embeddings, axis=0))

    def __len__(self) -> int:
        return len(self._ids)
