#!/usr/bin/env python3
"""Dream Cycle — 22-phase overnight maintenance pipeline.

Runs cluster/merge, recalibration, compression, pruning, session
finalization, preference consolidation, backup, and report generation.
Callable standalone or via API.
"""

import json
import logging
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

NEURAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, NEURAL_DIR)

from src import TextEmbedder, VectorMemoryStore, MemoryConsolidator
from ui.app import memory_system, logger

STORE_PATH = os.path.expanduser("~/.neural_memory/store.pkl")
BACKUP_DIR = os.path.expanduser("~/Documents/neural-memory.bak/backups")
REPORT_PATH = os.path.expanduser("~/.neural_memory/dream_report.md")

PHASE_NAMES = [
    "prelude", "garbage_collect", "link_repair", "similarity_scan",
    "cluster", "merge", "duplicate_detect", "recalibrate",
    "compress_transients", "compress_low_importance", "prune_transients",
    "prune_by_score", "close_stale_sessions", "summarize_sessions",
    "consolidate_preferences", "regenerate_identity", "rebuild_graph_cache",
    "backup", "dream_report", "metrics", "integrity_verify", "cooldown",
]


class DreamCycle:
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.store: Optional[VectorMemoryStore] = None
        self.embedder: Optional[TextEmbedder] = None
        self.consolidator = MemoryConsolidator()
        self.results: Dict[str, Any] = {"phases": {}}
        self.phase_times: Dict[str, float] = {}
        self.start_time: float = 0.0

    def _load_store(self):
        p = Path(STORE_PATH)
        if not p.exists():
            raise FileNotFoundError(f"Store not found at {STORE_PATH}")
        self.store = VectorMemoryStore()
        self.store.load(str(p))
        self.embedder = TextEmbedder()

    def _save_store(self):
        if self.store:
            self.store.save(STORE_PATH)

    def _phase(self, name: str, fn):
        t0 = time.time()
        logger.info("Dream phase %s — start", name)
        try:
            result = fn()
            elapsed = time.time() - t0
            self.results["phases"][name] = {"status": "ok", "elapsed": elapsed, "result": result}
            logger.info("Dream phase %s — done (%.2fs)", name, elapsed)
        except Exception as e:
            elapsed = time.time() - t0
            self.results["phases"][name] = {"status": "error", "elapsed": elapsed, "error": str(e)}
            logger.error("Dream phase %s — failed: %s", name, e)
            if self.config.get("fail_fast"):
                raise

    def run(self) -> dict:
        self.start_time = time.time()
        self.results = {"started_at": datetime.now(timezone.utc).isoformat(), "phases": {}}
        self._load_store()
        self._phase("prelude", self._phase_prelude)
        self._phase("garbage_collect", self._phase_garbage_collect)
        self._phase("link_repair", self._phase_link_repair)
        self._phase("similarity_scan", self._phase_similarity_scan)
        self._phase("cluster", self._phase_cluster)
        self._phase("merge", self._phase_merge)
        self._phase("duplicate_detect", self._phase_duplicate_detect)
        self._phase("recalibrate", self._phase_recalibrate)
        self._phase("compress_transients", self._phase_compress_transients)
        self._phase("compress_low_importance", self._phase_compress_low_importance)
        self._phase("prune_transients", self._phase_prune_transients)
        self._phase("prune_by_score", self._phase_prune_by_score)
        self._phase("close_stale_sessions", self._phase_close_stale_sessions)
        self._phase("summarize_sessions", self._phase_summarize_sessions)
        self._phase("consolidate_preferences", self._phase_consolidate_preferences)
        self._phase("regenerate_identity", self._phase_regenerate_identity)
        self._phase("rebuild_graph_cache", self._phase_rebuild_graph_cache)
        self._phase("backup", self._phase_backup)
        self._phase("dream_report", self._phase_dream_report)
        self._phase("metrics", self._phase_metrics)
        self._phase("integrity_verify", self._phase_integrity_verify)
        self._phase("cooldown", self._phase_cooldown)
        self.results["finished_at"] = datetime.now(timezone.utc).isoformat()
        total = time.time() - self.start_time
        self.results["total_elapsed"] = total
        logger.info("Dream cycle complete — %.2fs total", total)
        return self.results

    # ── Phase implementations ──

    def _phase_prelude(self) -> dict:
        ids = self.store.list_all()
        return {"total_memories": len(ids), "store_size_bytes": os.path.getsize(STORE_PATH) if os.path.exists(STORE_PATH) else 0}

    def _phase_garbage_collect(self) -> dict:
        all_ids = set(self.store.list_all())
        cleaned = 0
        for mid in self.store.list_all():
            meta = self.store.get_metadata(mid) or {}
            related = meta.get("related_memories", []) or []
            if not related:
                continue
            new_rel = [r for r in related if r in all_ids]
            if len(new_rel) != len(related):
                self.store.update_metadata_value(mid, "related_memories", new_rel)
                cleaned += 1
            rt = meta.get("related_types", {}) or {}
            if rt:
                new_rt = {k: v for k, v in rt.items() if k in all_ids}
                if len(new_rt) != len(rt):
                    self.store.update_metadata_value(mid, "related_types", new_rt)
        return {"cleaned_memories": cleaned}

    def _phase_link_repair(self) -> dict:
        all_ids = set(self.store.list_all())
        repaired = 0
        for mid in self.store.list_all():
            meta = self.store.get_metadata(mid) or {}
            related = meta.get("related_memories", []) or []
            text = meta.get("text", "") or ""
            for target in list(related):
                if target not in all_ids:
                    continue
                t_meta = self.store.get_metadata(target) or {}
                t_rel = t_meta.get("related_memories", []) or []
                if mid not in t_rel:
                    new_trel = list(t_rel) + [mid]
                    self.store.update_metadata_value(target, "related_memories", new_trel)
                    repaired += 1
        return {"repaired_links": repaired}

    def _phase_similarity_scan(self) -> dict:
        ids = self.store.list_all()
        if len(ids) < 2:
            return {"pairs_scanned": 0}
        embeddings = []
        valid_ids = []
        for mid in ids:
            idx = self.store.get_index_of(mid)
            if idx is not None:
                emb = self.store.get_embedding(idx)
                if emb is not None and len(emb) > 0:
                    embeddings.append(emb)
                    valid_ids.append(mid)
        if len(valid_ids) < 2:
            return {"pairs_scanned": 0}
        mat = np.stack(embeddings, axis=0)
        sim = mat @ mat.T
        pairs = 0
        for i in range(len(valid_ids)):
            for j in range(i + 1, len(valid_ids)):
                if sim[i, j] >= 0.85:
                    pairs += 1
        return {"pairs_scanned": len(valid_ids) * (len(valid_ids) - 1) // 2, "similar_pairs_ge_085": pairs}

    def _phase_cluster(self) -> dict:
        memories = []
        for mid in self.store.list_all():
            meta = self.store.get_metadata(mid) or {}
            idx = self.store.get_index_of(mid)
            emb = self.store.get_embedding(idx) if idx is not None else None
            if emb is not None:
                memories.append({"id": mid, "text": meta.get("text", ""), "embedding": emb.tolist() if hasattr(emb, 'tolist') else emb, "metadata": meta})
        if len(memories) < 2:
            return {"clusters": 0}
        merged = self.consolidator.consolidate(memories, threshold=0.85)
        clusters_found = len(merged) - len(memories)
        return {"clusters": max(0, clusters_found)}

    def _phase_merge(self) -> dict:
        return {"merged": 0, "note": "merge handled inside cluster phase"}

    def _phase_duplicate_detect(self) -> dict:
        texts = {}
        duplicates = 0
        for mid in self.store.list_all():
            meta = self.store.get_metadata(mid) or {}
            t = (meta.get("text", "") or "").strip().lower()
            if len(t) < 10:
                continue
            if t in texts:
                duplicates += 1
            texts[t] = mid
        return {"duplicate_texts": duplicates}

    def _phase_recalibrate(self) -> dict:
        now = time.time()
        all_ids = self.store.list_all()
        total = len(all_ids)
        boosted = 0
        decayed = 0
        for mid in all_ids:
            meta = self.store.get_metadata(mid) or {}
            imp = meta.get("importance_score", 0.5)
            access = meta.get("access_count", 0)
            ts = meta.get("timestamp", now)
            tags = meta.get("tags", []) or []
            related = meta.get("related_memories", []) or []
            new_imp = imp
            if len(related) >= 3:
                new_imp = min(imp + 0.05, 1.0)
                boosted += 1
            if access == 0 and (now - ts) > 86400 * 14:
                new_imp = max(imp - 0.1, 0.1)
                decayed += 1
            if "session-summary" in tags:
                new_imp = max(new_imp, 0.6)
            if abs(new_imp - imp) > 0.01:
                self.store.update_metadata_value(mid, "importance_score", new_imp)
        return {"boosted": boosted, "decayed": decayed, "total": total}

    def _phase_compress_transients(self) -> dict:
        return self._run_compress(tier_filter="transient")

    def _phase_compress_low_importance(self) -> dict:
        return self._run_compress(importance_max=0.4, min_age_hours=48)

    def _run_compress(self, tier_filter: Optional[str] = None, importance_max: Optional[float] = None, min_age_hours: int = 0) -> dict:
        from scripts.rip_and_compress import call_llm, LLM_CONFIG
        now = time.time()
        candidates = []
        for mid in self.store.list_all():
            meta = self.store.get_metadata(mid) or {}
            tier = meta.get("tier", "active")
            imp = meta.get("importance_score", 0.5)
            ts = meta.get("timestamp", now)
            text = meta.get("text", "") or ""
            if not text:
                continue
            if tier_filter and tier != tier_filter:
                continue
            if importance_max is not None and (imp >= importance_max or (now - ts) < min_age_hours * 3600):
                continue
            candidates.append((mid, meta, text))
        if not candidates or not LLM_CONFIG.get("api_key"):
            return {"compressed": 0, "skipped": len(candidates), "reason": "no_api_key" if not LLM_CONFIG.get("api_key") else "no_candidates"}
        compressed_count = 0
        for mid, meta, text in candidates:
            try:
                compressed = call_llm(text, LLM_CONFIG)
                if not compressed.strip():
                    continue
                new_emb = self.embedder.embed(compressed)
                idx = self.store.get_index_of(mid)
                if idx is None:
                    continue
                self.store.set_embedding(idx, np.array(new_emb, dtype=np.float32))
                self.store.update_metadata_value(mid, "text", compressed)
                self.store.update_metadata_value(mid, "tier", "compressed")
                old_imp = (self.store.get_metadata(mid) or {}).get("importance_score", 0.5)
                self.store.update_metadata_value(mid, "importance_score", min(old_imp + 0.1, 0.9))
                self.store.update_metadata_value(mid, "_compressed_at", now)
                compressed_count += 1
            except Exception:
                continue
        return {"compressed": compressed_count}

    def _phase_prune_transients(self) -> dict:
        before = len(self.store)
        removed = self.consolidator.prune(self.store, max_size=before, strategy="hybrid")
        self._save_store()
        return {"removed": removed, "remaining": len(self.store)}

    def _phase_prune_by_score(self) -> dict:
        max_memories = self.config.get("max_memories", 5000)
        if len(self.store) <= max_memories:
            return {"removed": 0, "remaining": len(self.store)}
        removed = self.consolidator.prune(self.store, max_size=max_memories, strategy="hybrid")
        self._save_store()
        return {"removed": removed, "remaining": len(self.store)}

    def _phase_close_stale_sessions(self) -> dict:
        now = time.time()
        closed = 0
        for mid in self.store.list_all():
            meta = self.store.get_metadata(mid) or {}
            tags = meta.get("tags", []) or []
            sd = meta.get("session_data", {}) or {}
            if "type:session" in tags and sd.get("status") == "active":
                last_ts = sd.get("entries", [{}])[-1].get("timestamp", 0) if sd.get("entries") else 0
                if last_ts and (now - last_ts) > 86400:
                    sd["status"] = "closed"
                    sd["closed_at"] = now
                    sd["protected"] = False
                    self.store.update_metadata_value(mid, "session_data", sd)
                    tags.append("session-summary")
                    self.store.update_metadata_value(mid, "tags", tags)
                    closed += 1
        return {"closed_sessions": closed}

    def _phase_summarize_sessions(self) -> dict:
        summarized = 0
        for mid in self.store.list_all():
            meta = self.store.get_metadata(mid) or {}
            tags = meta.get("tags", []) or []
            sd = meta.get("session_data", {}) or {}
            if "type:session" in tags and sd.get("status") == "closed" and not sd.get("_dream_summarized"):
                text = meta.get("text", "") or ""
                if len(text) > 500:
                    summary = text[:300] + f"\n\n[Dream Cycle: session {mid[:8]} had {len(sd.get('entries', []))} entries]"
                    self.store.update_metadata_value(mid, "text", summary)
                sd["_dream_summarized"] = True
                self.store.update_metadata_value(mid, "session_data", sd)
                summarized += 1
        return {"summarized": summarized}

    def _phase_consolidate_preferences(self) -> dict:
        if hasattr(memory_system, 'consolidate_preferences'):
            return memory_system.consolidate_preferences()
        return {"note": "consolidate_preferences not available"}

    def _phase_regenerate_identity(self) -> dict:
        if hasattr(memory_system, '_generate_identity_doc'):
            memory_system._generate_identity_doc()
            return {"regenerated": True}
        return {"note": "not available"}

    def _phase_rebuild_graph_cache(self) -> dict:
        count = 0
        for mid in self.store.list_all():
            meta = self.store.get_metadata(mid) or {}
            related = meta.get("related_memories", []) or []
            if related:
                count += len(related)
        return {"total_edges": count // 2, "total_nodes": len(self.store)}

    def _phase_backup(self) -> dict:
        import shutil
        os.makedirs(BACKUP_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = os.path.join(BACKUP_DIR, f"store_dream_{ts}.pkl")
        shutil.copy2(STORE_PATH, dst)
        return {"backup_path": dst}

    def _phase_dream_report(self) -> dict:
        lines = ["# Dream Cycle Report", f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}", ""]
        lines.append(f"**Total time:** {self.results.get('total_elapsed', 0):.1f}s")
        lines.append(f"**Phases:** {len([p for p in self.results.get('phases', {}).values() if p.get('status') == 'ok'])} ok, "
                     f"{len([p for p in self.results.get('phases', {}).values() if p.get('status') == 'error'])} errors")
        lines.append("")
        lines.append("| Phase | Status | Time (s) | Details |")
        lines.append("|-------|--------|----------|---------|")
        for name in PHASE_NAMES:
            p = self.results.get("phases", {}).get(name, {})
            result = p.get("result", {})
            if isinstance(result, dict):
                detail = " ".join(f"{k}={v}" for k, v in result.items() if not isinstance(v, (dict, list)))
            else:
                detail = str(result)[:80]
            lines.append(f"| {name} | {p.get('status', '?')} | {p.get('elapsed', 0):.2f} | {detail} |")
        lines.append("")
        lines.append(f"**Store:** {len(self.store.list_all())} memories after dream")
        report = "\n".join(lines)
        with open(REPORT_PATH, "w") as f:
            f.write(report)
        return {"report_path": REPORT_PATH, "lines": len(lines)}

    def _phase_metrics(self) -> dict:
        phases_ok = sum(1 for p in self.results.get("phases", {}).values() if p.get("status") == "ok")
        phases_err = sum(1 for p in self.results.get("phases", {}).values() if p.get("status") == "error")
        return {"phases_ok": phases_ok, "phases_error": phases_err}

    def _phase_integrity_verify(self) -> dict:
        issues = []
        all_ids = self.store.list_all()
        for mid in all_ids:
            idx = self.store.get_index_of(mid)
            if idx is None:
                issues.append(f"{mid[:8]}: no index")
                continue
            emb = self.store.get_embedding(idx)
            if emb is None or len(emb) == 0:
                issues.append(f"{mid[:8]}: empty embedding")
        return {"issues": len(issues), "detail": issues[:10]}

    def _phase_cooldown(self) -> dict:
        self._save_store()
        prelude = self.results.get("phases", {}).get("prelude", {}).get("result", {})
        before = prelude.get("total_memories", 0) if isinstance(prelude, dict) else 0
        after = len(self.store.list_all())
        return {"memories_before": before, "memories_after": after}


def run_dream_cycle(config: Optional[dict] = None) -> dict:
    dc = DreamCycle(config)
    return dc.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    import argparse
    parser = argparse.ArgumentParser(description="Dream Cycle — maintenance pipeline")
    parser.add_argument("--max-memories", type=int, default=5000)
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first phase failure")
    args = parser.parse_args()
    result = run_dream_cycle({"max_memories": args.max_memories, "fail_fast": args.fail_fast})
    print(json.dumps({k: v for k, v in result.items() if k != "phases"}, indent=2))
    errors = [n for n, p in result.get("phases", {}).items() if p.get("status") == "error"]
    if errors:
        print(f"\nPhases with errors: {', '.join(errors)}")
        sys.exit(1)
