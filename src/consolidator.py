import logging
import time
from typing import Dict, List

import numpy as np

from .memory_store import VectorMemoryStore

logger = logging.getLogger(__name__)


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

        def _meta_val(i: str, key: str, default):
            meta = store.get_metadata(i)
            if meta is not None:
                return meta.get(key, default)
            return default

        # Phase 1: Remove transient memories first (tier=transient)
        removed = 0
        transient_ids = [
            i for i in ids
            if _meta_val(i, "tier", "active") == "transient"
        ]
        for entry_id in transient_ids:
            self._cleanup_links(store, entry_id)
            store.delete(entry_id)
            removed += 1
            if removed >= remove_count:
                logger.info("Prune removed %d memories (all transient)", removed)
                return removed

        ids = store.list_all()
        remaining_remove = remove_count - removed

        if remaining_remove <= 0:
            return removed

        # Phase 2: Separate protected memories (never pruned)
        protectable = []
        protected_count = 0
        for i in ids:
            if _meta_val(i, "protected", False):
                protected_count += 1
            else:
                protectable.append(i)

        if not protectable:
            logger.info("All %d memories are protected — nothing to prune", protected_count)
            return 0

        # Phase 3: Compute effective score for each unprotected memory
        def _effective_score(i: str) -> float:
            meta = store.get_metadata(i)
            if meta is None:
                return 0.0
            importance = meta.get("importance_score", 0.5)
            tags = meta.get("tags", []) or []
            related = meta.get("related_memories", []) or []

            # Connection boost: +0.15 per related memory (capped at +0.3)
            connection_boost = min(len(related) * 0.15, 0.3)

            # Project/summary boost: +0.1 for tagged memories
            project_boost = 0.0
            if any(t.startswith("project:") for t in tags):
                project_boost += 0.1
            if "session-summary" in tags or "summary" in tags:
                project_boost += 0.1

            return importance + connection_boost + project_boost

        # Phase 4: Sort by effective score ascending (lowest first = delete first)
        protectable.sort(key=_effective_score)

        for entry_id in protectable[:remaining_remove]:
            self._cleanup_links(store, entry_id)
            store.delete(entry_id)
            removed += 1

        logger.info(
            "Prune removed %d memories (%d transient, %d protected kept, %d total after)",
            removed, len(transient_ids), protected_count, len(store)
        )
        return removed

    def _cleanup_links(self, store: VectorMemoryStore, removed_id: str):
        """Remove references to a deleted memory from other memories' related_memories lists."""
        for mid in store.list_all():
            meta = store.get_metadata(mid)
            if meta is None:
                continue
            related = meta.get("related_memories", []) or []
            if not related:
                continue
            cleaned = [r for r in related if r != removed_id and r != f"[[{removed_id}]]"]
            if len(cleaned) != len(related):
                store.update_metadata_value(mid, "related_memories", cleaned)

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
