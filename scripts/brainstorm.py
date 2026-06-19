#!/usr/bin/env python3
"""Brainstorming engine — two-phase pipeline with 3-provider fallback.

Phase 1: .hermes curated skills → rough draft + knowledge gaps
Phase 2: agency-agents matched divisions → detailed expert insights
Fallback chain: Gemini (free) → Groq (free) → Bluesminds (pay)
Stored separately from knowledge graph (brainstorm_sessions.json).
"""

import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

API_FILE = os.path.expanduser("~/Downloads/api.txt")

# ── Curated .hermes skills (always loaded in Phase 1) ──

CURATED_HERMES_SKILLS = [
    "/home/Beelzebub/.hermes/skills/dev/knowledge-agent/SKILL.md",
    "/home/Beelzebub/.hermes/skills/dev/completion-workflow/SKILL.md",
    "/home/Beelzebub/.hermes/skills/dev/pathfinder/SKILL.md",
    "/home/Beelzebub/.hermes/skills/dev/mem-search/SKILL.md",
    "/home/Beelzebub/.hermes/skills/marketing/marketing-psychology/SKILL.md",
    "/home/Beelzebub/.hermes/skills/marketing/churn-prevention/SKILL.md",
    "/home/Beelzebub/.hermes/skills/Custom_vault_for_skills/marketing-ideas.md",
    "/home/Beelzebub/.hermes/skills/Custom_vault_for_skills/impeccable.md",
    "/home/Beelzebub/.hermes/skills/creative/humanizer/SKILL.md",
    "/home/Beelzebub/.hermes/skills/creative/architecture-diagram/SKILL.md",
    "/home/Beelzebub/.hermes/skills/software-development/simplify-code/SKILL.md",
    "/home/Beelzebub/Documents/agency-agents/strategy/nexus-strategy.md",
    "/home/Beelzebub/Documents/agency-agents/project-management/project-manager-senior.md",
]

AGENCY_AGENTS_DIR = os.path.expanduser("~/Documents/agency-agents")


def _load_providers_from_file() -> List[Dict]:
    """Parse ~/Downloads/api.txt and return ordered provider list."""
    path = Path(API_FILE)
    if not path.exists():
        logger.warning("API file not found at %s", API_FILE)
        return []

    text = path.read_text()
    providers = []

    gemini_m = re.search(r'GEMINI_API_KEY=(\S+)', text)
    if gemini_m:
        providers.append({
            "name": "gemini",
            "type": "gemini_api",
            "model": "gemini-2.5-flash",
            "api_key": gemini_m.group(1),
            "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
        })

    groq_m = re.search(r'GROQ_API_KEY=(\S+)', text)
    if groq_m:
        providers.append({
            "name": "groq",
            "type": "openai_compat",
            "model": "llama-3.1-8b-instant",
            "api_key": groq_m.group(1),
            "endpoint": "https://api.groq.com/openai/v1/chat/completions",
        })

    bluesminds_key = None
    bluesminds_url = "https://api.bluesminds.com"
    json_m = re.search(r'\{"_type":"newapi_channel_conn","key":"([^"]+)","url":"([^"]+)"\}', text)
    if json_m:
        bluesminds_key = json_m.group(1)
        bluesminds_url = json_m.group(2)
    else:
        key_m = re.search(r'API_BLUESMINDS\s*:\s*(\S+)', text)
        if key_m:
            bluesminds_key = key_m.group(1)

    if bluesminds_key:
        endpoint = bluesminds_url.rstrip("/") + "/v1/chat/completions"
        providers.append({
            "name": "bluesminds",
            "type": "openai_compat",
            "model": "qwen/qwen3.5-397b-a17b",
            "api_key": bluesminds_key,
            "endpoint": endpoint,
        })

    return providers


class BrainstormEngine:
    """Two-phase brainstorming with provider fallback and chain linking."""

    def __init__(self):
        self.providers = _load_providers_from_file()
        self.hermes_skills_cache: Dict[str, str] = {}
        self.agency_index: Dict[str, List[Tuple[str, str]]] = {}
        self._build_indexes()

    def _build_indexes(self):
        """Pre-load .hermes skills and build agency-agents division index."""
        for path in CURATED_HERMES_SKILLS:
            p = Path(path).expanduser()
            if p.exists():
                self.hermes_skills_cache[path] = p.read_text()

        agency_dir = Path(AGENCY_AGENTS_DIR)
        if agency_dir.exists():
            for div_dir in sorted(agency_dir.iterdir()):
                if not div_dir.is_dir():
                    continue
                division = div_dir.name
                files = []
                for f in sorted(div_dir.glob("*.md")):
                    content = f.read_text()
                    desc = ""
                    if content.startswith("---"):
                        end = content.find("---", 3)
                        if end != -1:
                            fm = content[:end]
                            dm = re.search(r'description:\s*"([^"]*)"', fm)
                            if dm:
                                desc = dm.group(1)
                            else:
                                dm = re.search(r"description:\s*'([^']*)'", fm)
                                if dm:
                                    desc = dm.group(1)
                    if not desc:
                        desc = f.stem.replace("-", " ")
                    files.append((str(f), desc))
                self.agency_index[division] = files

    # ── LLM callers with fallback chain ──

    def _call_gemini(self, text: str, config: dict, temperature: float = 0.8, max_tokens: int = 4096) -> str:
        url = config["endpoint"] + "?key=" + config["api_key"]
        payload = {
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("Gemini returned no candidates")
        parts = candidates[0].get("content", {}).get("parts", [])
        text_out = parts[0].get("text", "") if parts else ""
        if not text_out:
            finish = candidates[0].get("finishReason", "")
            raise RuntimeError(f"Gemini returned empty text (finishReason: {finish})")
        return text_out

    def _call_openai_compat(self, text: str, config: dict, temperature: float = 0.8, max_tokens: int = 4096) -> str:
        headers = {"Authorization": f"Bearer {config['api_key']}", "Content-Type": "application/json"}
        payload = {
            "model": config["model"],
            "messages": [{"role": "user", "content": text}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post(config["endpoint"], headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "")

    def _call_llm(self, prompt: str, temperature: float = 0.8, max_tokens: int = 4096) -> str:
        """Try Gemini → Groq → Bluesminds with fallback on retryable errors."""
        if not self.providers:
            raise RuntimeError("No LLM providers — check ~/Downloads/api.txt")
        last_error = None
        for cfg in self.providers:
            try:
                if cfg["type"] == "gemini_api":
                    return self._call_gemini(prompt, cfg, temperature, max_tokens)
                return self._call_openai_compat(prompt, cfg, temperature, max_tokens)
            except requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else 0
                logger.warning("%s HTTP %d: %s", cfg["name"], status, e)
                last_error = e
                if status not in (429, 413, 500, 502, 503):
                    raise
            except Exception as e:
                logger.warning("%s error: %s", cfg["name"], e)
                last_error = e
        raise RuntimeError(f"All providers exhausted: {last_error}")

    # ── Agency-agents matching ──

    def _match_agency_divisions(self, text: str) -> List[Tuple[str, float, List[Tuple[str, str]]]]:
        words = set(re.findall(r'\b[a-zA-Z]{3,}\b', text.lower()))
        scored = []
        for division, files in self.agency_index.items():
            div_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', division.lower()))
            for _fp, desc in files:
                div_words.update(re.findall(r'\b[a-zA-Z]{3,}\b', desc.lower()))
            overlap = len(words & div_words)
            score = overlap / max(len(div_words), 1)
            scored.append((division, score, files))
        scored.sort(key=lambda x: -x[1])
        return [s for s in scored if s[1] > 0.15]

    def _load_division_content(self, matched: List[Tuple]) -> str:
        parts = []
        for division, score, files in matched:
            parts.append(f"\n## {division} (relevance {score:.2f})")
            limit = 60 if len(matched) > 2 else 100
            for fpath, _desc in files:
                content = Path(fpath).read_text()
                lines = content.split("\n")[:limit]
                parts.append(f"\n### {Path(fpath).stem}\n" + "\n".join(lines))
        return "\n".join(parts)

    # ── Phase 1: Rough draft ──

    def _phase1_rough(self, cluster_text: str) -> str:
        # Truncate each skill to first 30 lines to keep total prompt manageable
        skills_texts = []
        for path, content in self.hermes_skills_cache.items():
            lines = content.split("\n")
            snippet = "\n".join(lines[:30])
            skills_texts.append(f"-- {Path(path).stem} --\n{snippet}")
        skills_text = "\n\n".join(skills_texts)
        prompt = (
            "You are a creative brainstorming assistant with expertise in various domains:\n\n"
            f"{skills_text}\n\n"
            "Given this memory cluster, generate initial rough ideas:\n\n"
            f"MEMORY CLUSTER:\n{cluster_text}\n\n"
            "Write your thoughts in plain, flowing English paragraphs. Imagine you're explaining "
            "your ideas to a smart colleague over coffee. Cover a few broad approaches you see here "
            "and note what expertise or knowledge would help develop them further. No bullet points, "
            "no headings, no markdown — just clear, natural prose."
        )
        return self._call_llm(prompt, temperature=0.8, max_tokens=4096)

    # ── Phase 2: Expert plan ──

    def _phase2_expert(self, cluster_text: str, phase1_output: str) -> Tuple[List[Dict], str]:
        matched = self._match_agency_divisions(phase1_output + "\n" + cluster_text)
        agency_context = self._load_division_content(matched)

        non_matched = []
        for division, files in self.agency_index.items():
            if not any(m[0] == division for m in matched):
                descs = [f"{Path(fp).stem}: {d}" for fp, d in files[:3]]
                non_matched.append(f"{division}: {', '.join(descs)}")

        prompt = (
            "You are a strategic brainstorming expert with deep domain knowledge from these divisions:\n\n"
            f"{agency_context}\n\n"
            "Other available frameworks (condensed):\n"
            f"{chr(10).join(non_matched)}\n\n"
            f"MEMORY CLUSTER:\n{cluster_text}\n\n"
            "PHASE 1 ROUGH DRAFT:\n"
            f"{phase1_output}\n\n"
            "Write 3-5 detailed, actionable insights as natural paragraphs. Separate each insight "
            "with a blank line. In the text, mention which division inspired each idea (e.g. "
            "'drawing from the marketing/churn-prevention division'). End each insight with a "
            "score line like this:\n\n"
            "[Novel: 7/10 · Feasible: 8/10 · Relevant: 9/10 · Coherent: 8/10]\n\n"
            "Just write clear, flowing prose. No JSON, no bullet points, no markdown headings."
        )

        result = self._call_llm(prompt, temperature=0.7, max_tokens=4096)
        insights = self._parse_insights(result)
        chain_reasoning = self._generate_chain_reasoning(phase1_output, insights)
        return insights, chain_reasoning

    def _parse_insights(self, text: str) -> List[Dict]:
        insights = []
        blocks = re.split(r'\n\n+', text.strip())
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            scores = {"novelty": 0.5, "feasibility": 0.5, "relevance": 0.5, "coherence": 0.5}
            skills_used = []
            sm = re.search(r'\[Novel:\s*(\d+)/10.*?Feasible:\s*(\d+)/10.*?Relevant:\s*(\d+)/10.*?Coherent:\s*(\d+)/10\]', block, re.DOTALL)
            if sm:
                scores = {
                    "novelty": int(sm.group(1)) / 10,
                    "feasibility": int(sm.group(2)) / 10,
                    "relevance": int(sm.group(3)) / 10,
                    "coherence": int(sm.group(4)) / 10,
                }
                block = re.sub(r'\[Novel:.*?/10\]', '', block).strip()
            if not block:
                continue
            for m in re.finditer(r'([a-z][a-z0-9_-]+/[a-z][a-z0-9_-]+)', block):
                skills_used.append(m.group(1))
            for m in re.finditer(r"(?:drawing from|from the|using the|inspired by) ['\"]?([a-zA-Z][a-zA-Z0-9_-]+(?:/[a-zA-Z][a-zA-Z0-9_-]+)?)", block):
                skill = m.group(1)
                if skill not in skills_used and '/' not in skill:
                    skills_used.append(skill)
            insights.append({
                "text": block,
                "skills_used": skills_used,
                "novelty": scores["novelty"],
                "feasibility": scores["feasibility"],
                "relevance": scores["relevance"],
                "coherence": scores["coherence"],
            })
        if not insights:
            insights = [{"text": text[:800], "skills_used": [], "novelty": 0.5, "feasibility": 0.5, "relevance": 0.5, "coherence": 0.5}]
        return insights

    def _generate_chain_reasoning(self, phase1_text: str, insights: List[Dict]) -> str:
        insights_text = "\n\n".join(f"Idea {i+1}: {ins.get('text', '')[:500]}" for i, ins in enumerate(insights))
        prompt = (
            "Given these Phase 1 rough ideas and Phase 2 detailed insights, explain the reasoning chain.\n\n"
            f"PHASE 1:\n{phase1_text[:1500]}\n\n"
            f"PHASE 2 INSIGHTS:\n{insights_text}\n\n"
            "For each Phase 2 insight, write one short paragraph explaining WHY it evolved from Phase 1: "
            "what gap or category in Phase 1 led to this insight, and what knowledge (from which skill) "
            "filled that gap. Start each paragraph with 'Idea 1:', 'Idea 2:', etc. No JSON, no bullet points."
        )
        return self._call_llm(prompt, temperature=0.4, max_tokens=2048)

    def _cross_pollinate(self, all_insights: List[Dict], chain_reasoning: str) -> List[Dict]:
        if len(all_insights) < 2:
            return []
        insights_text = "\n\n".join(f"Insight {i+1}: {ins.get('text', '')[:500]}" for i, ins in enumerate(all_insights))
        prompt = (
            "Given these insights from brainstorming:\n\n"
            f"{insights_text}\n\n"
            "Look across all of them and identify 1-3 meta-insights that emerge from COMBINING multiple insights. "
            "Look for patterns, contradictions, synergies, or unexpected combinations.\n\n"
            "Write each meta-insight as a natural paragraph. Inside parentheses at the end, note which "
            "insight numbers you combined (e.g. \"(combined insights 1 and 3)\"). No JSON, no bullets, no markdown."
        )
        result = self._call_llm(prompt, temperature=0.7, max_tokens=2048)
        metas = []
        for block in re.split(r'\n\n+', result.strip()):
            block = block.strip()
            if not block:
                continue
            src_indices = []
            sm = re.search(r'\(combined insights?\s*([\d, ]+)\)', block)
            if sm:
                try:
                    src_indices = [int(x.strip()) - 1 for x in sm.group(1).split(",") if x.strip().isdigit()]
                except ValueError:
                    pass
            metas.append({"text": block, "source_indices": src_indices, "reasoning": ""})
        return metas

    # ── Session enrichment (edges + importance) ──

    @staticmethod
    def _enrich_session(session: dict):
        nodes = session.get("nodes", [])
        node_map = {n["id"]: n for n in nodes}
        edges = []

        for node in nodes:
            scores = node.get("scores", {})
            if isinstance(scores, dict):
                vals = []
                for k in ("novelty", "feasibility", "relevance", "coherence"):
                    v = scores.get(k)
                    if v is not None and isinstance(v, (int, float)):
                        vals.append(v)
                node["importance"] = sum(vals) / len(vals) if vals else 0.5
            else:
                node["importance"] = 0.5

            pid = node.get("parent_id")
            if pid and pid in node_map:
                edges.append({
                    "source": pid, "target": node["id"],
                    "type": "generated",
                    "reasoning": (node.get("reasoning", "") or "")[:120],
                })

            syn = node.get("synthesis_sources", [])
            if syn:
                for sid in syn:
                    if sid in node_map:
                        edges.append({
                            "source": sid, "target": node["id"],
                            "type": "synthesis",
                            "reasoning": "Cross-pollination",
                        })

        for pid in set(n.get("parent_id") for n in nodes if n.get("parent_id") and n["parent_id"] in node_map):
            siblings = [n["id"] for n in nodes if n.get("parent_id") == pid]
            for i in range(len(siblings)):
                for j in range(i + 1, len(siblings)):
                    if i == 0:
                        edges.append({
                            "source": siblings[i], "target": siblings[j],
                            "type": "related",
                            "reasoning": "Same cluster",
                        })

        seen = set()
        deduped = []
        for e in edges:
            key = tuple(sorted((e["source"], e["target"]))) + (e["type"],)
            if key not in seen:
                seen.add(key)
                deduped.append(e)
        session["edges"] = deduped

    # ── Clustering ──

    def _simple_cluster(self, items: List[Dict], threshold: float = 0.3) -> List[List[Dict]]:
        clusters = []
        used = set()
        for i, item in enumerate(items):
            if i in used:
                continue
            cluster = [item]
            used.add(i)
            wi = set(re.findall(r'\b[a-zA-Z]{4,}\b', item["text"].lower()))
            for j in range(i + 1, len(items)):
                if j in used:
                    continue
                wj = set(re.findall(r'\b[a-zA-Z]{4,}\b', items[j]["text"].lower()))
                if wi and wj:
                    overlap = len(wi & wj)
                    if overlap / max(len(wi), len(wj)) > threshold:
                        cluster.append(items[j])
                        used.add(j)
            clusters.append(cluster)
        return sorted(clusters, key=len, reverse=True)

    # ── Idea consolidation (merge semantically similar nodes) ──

    def consolidate_ideas(self, nodes: List[Dict], threshold: float = 0.35) -> List[Dict]:
        """Merge semantically similar nodes into one. Uses word-overlap similarity."""
        if len(nodes) < 2:
            return nodes
        merged_ids = set()
        result = []
        for i, node in enumerate(nodes):
            if i in merged_ids:
                continue
            wi = set(re.findall(r'\b[a-zA-Z]{4,}\b', node.get("text", "").lower()))
            for j in range(i + 1, len(nodes)):
                if j in merged_ids:
                    continue
                wj = set(re.findall(r'\b[a-zA-Z]{4,}\b', nodes[j].get("text", "").lower()))
                if wi and wj:
                    overlap = len(wi & wj)
                    sim = overlap / max(len(wi), len(wj))
                    if sim > threshold:
                        merged_ids.add(j)
                        node["text"] = node.get("text", "") + " — " + nodes[j].get("text", "")
                        node["text"] = node["text"][:1000]
                        s1 = node.get("scores", {})
                        s2 = nodes[j].get("scores", {})
                        for k in ("novelty", "feasibility", "relevance", "coherence"):
                            v1 = s1.get(k, 0.5) if isinstance(s1, dict) else 0.5
                            v2 = s2.get(k, 0.5) if isinstance(s2, dict) else 0.5
                            s1[k] = (v1 + v2) / 2
                        node["scores"] = s1
                        sk1 = node.get("skills_used", [])
                        sk2 = nodes[j].get("skills_used", [])
                        combined_skills = list(dict.fromkeys(sk1 + sk2))
                        node["skills_used"] = combined_skills
                        sm1 = node.get("source_memories", [])
                        sm2 = nodes[j].get("source_memories", [])
                        combined_sm = list(dict.fromkeys(sm1 + sm2))
                        node["source_memories"] = combined_sm
            result.append(node)
        return result

    # ── Dream cycle entry ──

    def dream_cycle_run(self, store, config: Optional[Dict] = None) -> Dict:
        cfg = config or {}
        all_ids = store.list_all()
        texts = []
        for mid in all_ids:
            meta = store.get_metadata(mid) or {}
            t = (meta.get("text", "") or "").strip()
            if len(t) > 20:
                texts.append({"id": mid, "text": t, "imp": meta.get("importance_score", 0.5)})
        if len(texts) < 3:
            return {"session_id": None, "reason": "too_few_memories"}

        clusters = self._simple_cluster(texts, threshold=0.3)[:cfg.get("max_clusters", 10)]
        session_id = str(uuid.uuid4())
        ts = datetime.now(timezone.utc).isoformat()
        session = {"id": session_id, "topic": f"Dream cycle — {ts[:10]}", "trigger": "dream",
                    "created_at": ts, "deleted": False, "nodes": []}
        all_insights = []

        for i, cluster in enumerate(clusters):
            cluster_text = "\n\n".join(c["text"][:500] for c in cluster)
            phase1 = self._phase1_rough(cluster_text)
            root_id = str(uuid.uuid4())
            session["nodes"].append({
                "id": root_id, "session_id": session_id, "phase": 1,
                "text": phase1[:1000], "parent_id": None,
                "reasoning": f"Cluster {i+1}: {','.join(c['id'][:8] for c in cluster[:3])}",
                "level": 0,
                "scores": {"novelty": 0.5, "feasibility": 0.5, "relevance": 0.8, "coherence": 0.7},
                "skills_used": list(self.hermes_skills_cache.keys()),
                "source_memories": [c["id"] for c in cluster[:5]],
                "created_at": ts,
            })
            insights, reasoning_text = self._phase2_expert(cluster_text, phase1)
            reasoning_blocks = re.split(r'\n\n+', reasoning_text.strip()) if reasoning_text else []
            for j, ins in enumerate(insights):
                rid = str(uuid.uuid4())
                rtext = ""
                for rb in reasoning_blocks:
                    if rb.startswith(f"Idea {j+1}:") or rb.startswith(f"Idea {j+1}:"):
                        rtext = rb.split(":", 1)[1].strip() if ":" in rb else rb
                        break
                if not rtext and j < len(reasoning_blocks):
                    rtext = reasoning_blocks[j][:200]
                session["nodes"].append({
                    "id": rid, "session_id": session_id, "phase": 2,
                    "text": ins.get("text", "")[:1000], "parent_id": root_id,
                    "reasoning": rtext, "level": 1,
                    "scores": {
                        "novelty": ins.get("novelty", 0.5),
                        "feasibility": ins.get("feasibility", 0.5),
                        "relevance": ins.get("relevance", 0.5),
                        "coherence": ins.get("coherence", 0.5),
                    },
                    "skills_used": ins.get("skills_used", []),
                    "source_memories": [c["id"] for c in cluster[:3]],
                    "created_at": ts,
                })
                all_insights.append(ins)

        # Consolidate similar Phase 2 nodes within this cluster
        cluster_phase2 = [n for n in session["nodes"] if n["phase"] == 2]
        if cluster_phase2:
            consolidated = self.consolidate_ideas(cluster_phase2, threshold=0.35)
            session["nodes"] = [n for n in session["nodes"] if n["phase"] != 2] + consolidated
            all_insights = [
                {"text": n.get("text", ""), "skills_used": n.get("skills_used", []),
                 "novelty": n.get("scores", {}).get("novelty", 0.5),
                 "feasibility": n.get("scores", {}).get("feasibility", 0.5),
                 "relevance": n.get("scores", {}).get("relevance", 0.5),
                 "coherence": n.get("scores", {}).get("coherence", 0.5)}
                for n in consolidated
            ]

        meta = self._cross_pollinate(all_insights, "")
        phase2_nodes = [n for n in session["nodes"] if n["phase"] == 2]
        for m in meta:
            mid = str(uuid.uuid4())
            src = m.get("source_indices", [])
            synthesis_sources = []
            for idx in src:
                if idx < len(phase2_nodes):
                    synthesis_sources.append(phase2_nodes[idx]["id"])
            parent = synthesis_sources[0] if synthesis_sources else None
            session["nodes"].append({
                "id": mid, "session_id": session_id, "phase": 2,
                "text": m.get("text", "")[:1000], "parent_id": parent,
                "reasoning": m.get("reasoning", "Cross-pollination"),
                "level": 2 if parent else 1,
                "scores": {"novelty": 0.8, "feasibility": 0.6, "relevance": 0.9, "coherence": 0.75},
                "skills_used": [], "source_memories": [], "created_at": ts,
                "synthesis_sources": synthesis_sources,
                "synthesis": True,
            })

        self._enrich_session(session)
        return {"session_id": session_id, "clusters": len(clusters),
                "nodes": len(session["nodes"]), "session": session}

    # ── Active brainstorm (API/MCP) ──

    def active_brainstorm(self, topic: str, skills: Optional[List[str]] = None,
                          n_ideas: int = 5, recall_k: int = 20, memory_system=None) -> Dict:
        ts = datetime.now(timezone.utc).isoformat()
        session_id = str(uuid.uuid4())
        session = {"id": session_id, "topic": topic[:200], "trigger": "api",
                    "created_at": ts, "deleted": False, "nodes": []}

        cluster_text = f"User topic: {topic}"
        if memory_system and hasattr(memory_system, "retriever") and memory_system.retriever:
            try:
                results = memory_system.retriever.search(topic, k=recall_k)
                if results:
                    mems = "\n\n".join(r.get("text", "")[:300] for r in results if r.get("text"))
                    cluster_text += "\n\nRelated memories:\n" + mems
            except Exception as e:
                logger.warning("Memory retrieval failed: %s", e)

        phase1 = self._phase1_rough(cluster_text)
        root_id = str(uuid.uuid4())
        session["nodes"].append({
            "id": root_id, "session_id": session_id, "phase": 1,
            "text": phase1[:1000], "parent_id": None,
            "reasoning": f"Active brainstorm on: {topic[:100]}",
            "level": 0,
            "scores": {"novelty": 0.5, "feasibility": 0.5, "relevance": 0.8, "coherence": 0.7},
            "skills_used": list(self.hermes_skills_cache.keys()),
            "source_memories": [], "created_at": ts,
        })
        insights, reasoning_text = self._phase2_expert(cluster_text, phase1)
        reasoning_blocks = re.split(r'\n\n+', reasoning_text.strip()) if reasoning_text else []
        all_phase2_insights = []
        for j, ins in enumerate(insights[:n_ideas]):
            rid = str(uuid.uuid4())
            rtext = ""
            for rb in reasoning_blocks:
                if rb.startswith(f"Idea {j+1}:") or rb.startswith(f"Idea {j+1}:"):
                    rtext = rb.split(":", 1)[1].strip() if ":" in rb else rb
                    break
            if not rtext and j < len(reasoning_blocks):
                rtext = reasoning_blocks[j][:200]
            session["nodes"].append({
                "id": rid, "session_id": session_id, "phase": 2,
                "text": ins.get("text", "")[:1000], "parent_id": root_id,
                "reasoning": rtext, "level": 1,
                    "scores": {
                        "novelty": ins.get("novelty", 0.5),
                        "feasibility": ins.get("feasibility", 0.5),
                        "relevance": ins.get("relevance", 0.5),
                        "coherence": ins.get("coherence", 0.5),
                    },
                    "skills_used": ins.get("skills_used", []),
                    "source_memories": [], "created_at": ts,
            })
            all_phase2_insights.append(ins)

        # Consolidate similar Phase 2 nodes
        phase2_before = [n for n in session["nodes"] if n["phase"] == 2]
        if phase2_before:
            consolidated = self.consolidate_ideas(phase2_before, threshold=0.35)
            session["nodes"] = [n for n in session["nodes"] if n["phase"] != 2] + consolidated
            all_phase2_insights = [
                {"text": n.get("text", ""), "skills_used": n.get("skills_used", []),
                 "novelty": n.get("scores", {}).get("novelty", 0.5),
                 "feasibility": n.get("scores", {}).get("feasibility", 0.5),
                 "relevance": n.get("scores", {}).get("relevance", 0.5),
                 "coherence": n.get("scores", {}).get("coherence", 0.5)}
                for n in consolidated
            ]

        meta = self._cross_pollinate(all_phase2_insights, "")
        phase2_nodes = [n for n in session["nodes"] if n["phase"] == 2]
        for m in meta:
            mid = str(uuid.uuid4())
            src = m.get("source_indices", [])
            synthesis_sources = []
            for idx in src:
                if idx < len(phase2_nodes):
                    synthesis_sources.append(phase2_nodes[idx]["id"])
            parent = synthesis_sources[0] if synthesis_sources else None
            session["nodes"].append({
                "id": mid, "session_id": session_id, "phase": 2,
                "text": m.get("text", "")[:1000], "parent_id": parent,
                "reasoning": m.get("reasoning", "Cross-pollination"),
                "level": 2 if parent else 1,
                "scores": {"novelty": 0.8, "feasibility": 0.6, "relevance": 0.9, "coherence": 0.75},
                "skills_used": [], "source_memories": [], "created_at": ts,
                "synthesis_sources": synthesis_sources,
                "synthesis": True,
            })

        self._enrich_session(session)
        return {"session_id": session_id, "nodes": len(session["nodes"]), "session": session}
