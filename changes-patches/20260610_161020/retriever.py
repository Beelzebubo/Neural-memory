import re
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

from .embedder import TextEmbedder
from .memory_store import VectorMemoryStore


class MemoryRetriever:
    def __init__(self, embedder: TextEmbedder, store: VectorMemoryStore):
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, k: int = 10, min_score: float = 0.0) -> List[dict]:
        if not query or not query.strip():
            return []
        query_emb = self.embedder.embed(query)
        results = self.store.search(query_emb, k)
        return [r for r in results if r["score"] >= min_score]

    def retrieve_by_metadata(self, filters: dict) -> List[dict]:
        results = []
        for entry_id in self.store.list_all():
            meta = vars(self.store).get("_metadata", {}).get(entry_id, {})
            match = True
            for key, value in filters.items():
                if key not in meta:
                    match = False
                    break
                if callable(value):
                    if not value(meta[key]):
                        match = False
                        break
                elif meta[key] != value:
                    match = False
                    break
            if match:
                results.append({
                    "id": entry_id,
                    "metadata": dict(meta),
                })
        return results

    def hybrid_retrieve(self, query: str, k: int = 10, alpha: float = 0.5) -> List[dict]:
        if not query or not query.strip():
            return []

        query_emb = self.embedder.embed(query)
        semantic_results = self.store.search(query_emb, k * 2)
        semantic_map = {r["id"]: r["score"] for r in semantic_results}

        query_tokens = set(self._tokenize(query))
        keyword_scores: Dict[str, float] = {}
        for entry_id in self.store.list_all():
            meta = vars(self.store).get("_metadata", {}).get(entry_id, {})
            source = meta.get("source", "")
            source_tokens = self._tokenize(source)
            overlap = len(query_tokens & set(source_tokens)) if source_tokens else 0
            keyword_scores[entry_id] = overlap / max(len(query_tokens), 1)

        combined = []
        all_ids = set(semantic_map.keys()) | set(keyword_scores.keys())
        for entry_id in all_ids:
            sem_score = semantic_map.get(entry_id, 0.0)
            kw_score = keyword_scores.get(entry_id, 0.0)
            combined_score = alpha * sem_score + (1 - alpha) * kw_score

            meta = vars(self.store).get("_metadata", {}).get(entry_id, {})
            combined.append({
                "id": entry_id,
                "score": combined_score,
                "metadata": dict(meta),
            })

        combined.sort(key=lambda x: x["score"], reverse=True)
        return combined[:k]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())
