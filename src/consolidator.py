import time
from typing import Dict, List

import numpy as np

from .memory_store import VectorMemoryStore


class MemoryConsolidator:
    def consolidate(self, memories: List[dict], threshold: float = 0.85) -> List[dict]:
        if not memories:
            return []

        embeddings = []
        valid = []
        remainder = []
        for m in memories:
            if "embedding" in m:
                emb = np.array(m["embedding"], dtype=np.float32)
                embeddings.append(emb)
                valid.append(m)
            else:
                remainder.append(m)

        if not embeddings:
            return memories

        embeddings = np.stack(embeddings, axis=0)
        sim_matrix = embeddings @ embeddings.T
        assigned = [False] * len(valid)
        clusters = []

        for i in range(len(valid)):
            if assigned[i]:
                continue
            cluster = [valid[i]]
            assigned[i] = True
            for j in range(i + 1, len(valid)):
                if assigned[j]:
                    continue
                if sim_matrix[i, j] >= threshold:
                    cluster.append(valid[j])
                    assigned[j] = True
            clusters.append(cluster)

        merged = list(remainder)
        for cluster in clusters:
            merged_entry = self._merge_cluster(cluster)
            merged.append(merged_entry)

        return merged

    def summarize(self, memories: List[dict]) -> str:
        if not memories:
            return "No memories to summarize."

        sources = []
        importance_scores = []
        timestamps = []

        for m in memories:
            meta = m.get("metadata", {})
            if meta.get("source"):
                sources.append(str(meta["source"]))
            if meta.get("importance_score") is not None:
                importance_scores.append(float(meta["importance_score"]))
            if meta.get("timestamp"):
                timestamps.append(float(meta["timestamp"]))

        summary_parts = [f"Cluster of {len(memories)} memories."]

        if sources:
            unique_sources = list(set(sources))
            if len(unique_sources) <= 3:
                summary_parts.append(f"Sources: {', '.join(unique_sources)}.")
            else:
                summary_parts.append(f"Sources: {', '.join(unique_sources[:3])} and {len(unique_sources) - 3} more.")

        if importance_scores:
            avg_imp = sum(importance_scores) / len(importance_scores)
            summary_parts.append(f"Average importance: {avg_imp:.2f}.")

        if timestamps:
            earliest = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(min(timestamps)))
            latest = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(max(timestamps)))
            summary_parts.append(f"Date range: {earliest} to {latest}.")

        return " ".join(summary_parts)

    def prune(
        self,
        store: VectorMemoryStore,
        max_size: int,
        strategy: str = "hybrid",
    ) -> int:
        if len(store) <= max_size:
            return 0

        ids = store.list_all()
        remove_count = len(ids) - max_size

        if strategy == "by_age":
            ids_sorted = sorted(
                ids,
                key=lambda i: store._metadata.get(i, {}).get("timestamp", 0),
            )
        elif strategy == "by_importance":
            ids_sorted = sorted(
                ids,
                key=lambda i: store._metadata.get(i, {}).get("importance_score", 0.5),
            )
        elif strategy == "by_access_frequency":
            ids_sorted = sorted(
                ids,
                key=lambda i: store._metadata.get(i, {}).get("access_count", 0),
            )
        else:
            ids_sorted = sorted(
                ids,
                key=lambda i: (
                    store._metadata.get(i, {}).get("importance_score", 0.5),
                    store._metadata.get(i, {}).get("access_count", 0),
                    store._metadata.get(i, {}).get("timestamp", 0),
                ),
            )

        removed = 0
        for entry_id in ids_sorted[:remove_count]:
            store.delete(entry_id)
            removed += 1

        return removed

    def _merge_cluster(self, cluster: List[dict]) -> dict:
        if len(cluster) == 1:
            return cluster[0]

        texts = [m.get("text", "") for m in cluster if m.get("text")]
        source = cluster[0].get("metadata", {}).get("source", "merged")
        timestamps = [
            m.get("metadata", {}).get("timestamp", 0)
            for m in cluster
        ]
        importance = max(
            m.get("metadata", {}).get("importance_score", 0.5)
            for m in cluster
        )
        access = sum(
            m.get("metadata", {}).get("access_count", 0)
            for m in cluster
        )
        merged_meta = {
            "timestamp": max(timestamps) if timestamps else time.time(),
            "source": source,
            "importance_score": importance,
            "access_count": access,
            "merged_from": len(cluster),
        }

        merged_text = ""
        if texts:
            unique_texts = list(dict.fromkeys(texts))
            merged_text = " ".join(unique_texts)

        merged = {
            "text": merged_text,
            "metadata": merged_meta,
        }

        if "embedding" in cluster[0]:
            emb = cluster[0]["embedding"]
            merged["embedding"] = emb

        return merged
