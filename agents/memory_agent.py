from __future__ import annotations
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np

from agents.base_agent import BaseAgent
from core.event_bus import EventBus
from core.shared_context import SharedContext
from utils.metrics import ConfigDelta, compute_cosine_similarity, text_to_feature_vector

# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a cross-session experience manager agent for YOLO training optimization.

You have access to a memory store of past (weakness, fix, actual_improvement) triples.
Your role is to decide:
1. Which recalled memories are ACTUALLY relevant to the current situation (not just similar text)
2. Whether domain or dataset differences should discount recalled confidence
3. How to synthesize conflicting memories (e.g. same pattern fixed differently across iterations)
4. Whether a new experience is worth storing (is it genuinely informative?)

Key reasoning rules:
- High cosine similarity does NOT guarantee relevance — domain shift matters
- A fix that worked once with +0.02 improvement is less trustworthy than one that worked 3 times with +0.05 avg
- If a weakness is recurring across multiple iterations, weight fixes that address ROOT CAUSE over symptom fixes
- Memory with negative improvement (fix made things worse) is highly informative — flag it as a warning

Output a curated, relevance-assessed list with confidence adjustments."""

OUTPUT_SCHEMA = """{
  "curated_memories": [
    {
      "run_id": "original run id",
      "weakness_signature": "original signature",
      "config_delta_changes": {"param": value},
      "original_improvement": float,
      "adjusted_confidence": float (0.0-1.0, your assessment of how applicable this is NOW),
      "relevance_reasoning": "why this is or isn't relevant",
      "is_warning": false (true if this fix made things worse — avoid repeating it)
    }
  ],
  "synthesis": "one paragraph summarizing what the Planner should know from memory",
  "should_store_current": true | false,
  "store_reasoning": "why or why not store the current experience"
}"""
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MemoryEntry:
    weakness_signature: str
    weakness_embedding: np.ndarray
    config_delta: dict
    actual_map_improvement: float
    experiment_run_id: str
    timestamp: str
    iteration_count: int = 1  # how many times this pattern appeared

    def to_serializable(self) -> dict:
        return {
            "weakness_signature": self.weakness_signature,
            "weakness_embedding": self.weakness_embedding.tolist(),
            "config_delta": self.config_delta,
            "actual_map_improvement": self.actual_map_improvement,
            "experiment_run_id": self.experiment_run_id,
            "timestamp": self.timestamp,
            "iteration_count": self.iteration_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryEntry":
        return cls(
            weakness_signature=d["weakness_signature"],
            weakness_embedding=np.array(d["weakness_embedding"], dtype=np.float32),
            config_delta=d["config_delta"],
            actual_map_improvement=d["actual_map_improvement"],
            experiment_run_id=d["experiment_run_id"],
            timestamp=d["timestamp"],
            iteration_count=d.get("iteration_count", 1),
        )


class MemoryAgent(BaseAgent):
    """
    Cross-session retrieval-augmented memory.
    LLM evaluates relevance, adjusts confidence, and synthesizes recalled memories
    rather than returning raw cosine-similarity results.
    """

    AGENT_NAME = "MemoryAgent"
    SYSTEM_PROMPT = SYSTEM_PROMPT

    def _setup(self) -> None:
        cfg = self.config.get("memory", {})
        self.max_entries: int = cfg.get("max_entries", 500)
        self.similarity_threshold: float = cfg.get("similarity_threshold", 0.65)
        self.embedding_dim: int = cfg.get("embedding_dim", 128)
        self.top_k_raw: int = cfg.get("top_k_recall", 8)   # raw candidates for LLM
        self.top_k_final: int = cfg.get("top_k_final", 3)  # after LLM curation

        store_path = cfg.get("store_path", "memory_store.pkl")
        self.store_path: Path = self.experiment_dir / Path(store_path).name
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

        self._entries: list[MemoryEntry] = []
        self._load()

    def run(self, weaknesses: list[dict], **kwargs) -> dict:
        """
        Main entry: recall + LLM curation.
        Returns a dict with curated memories and synthesis for the Planner.
        """
        if not self._entries:
            return {"curated_memories": [], "synthesis": "No memory entries yet.", "raw_count": 0}

        # Step 1: raw cosine recall (deterministic)
        raw_recalls = self._raw_recall(weaknesses, top_k=self.top_k_raw)
        if not raw_recalls:
            return {"curated_memories": [], "synthesis": "No similar past experiences found.", "raw_count": 0}

        # Step 2: LLM curation
        perception = self._build_perception(weaknesses, raw_recalls)
        decision = self.reason(perception, OUTPUT_SCHEMA)

        if not decision:
            decision = self._fallback_curation(raw_recalls)

        decision["raw_count"] = len(raw_recalls)
        self.logger.info(
            f"Memory: {len(raw_recalls)} raw -> "
            f"{len(decision.get('curated_memories', []))} curated | "
            f"synthesis: {decision.get('synthesis', '')[:80]}..."
        )
        return decision

    def store(
        self,
        weaknesses: list[dict],
        config_delta: ConfigDelta,
        actual_map_improvement: float,
        run_id: str,
    ) -> None:
        from datetime import datetime
        signature = self._build_signature(weaknesses)
        embedding = text_to_feature_vector(signature, self.embedding_dim)

        # Check if same signature exists — if so, update iteration count
        existing = next(
            (e for e in self._entries if e.weakness_signature == signature), None
        )
        if existing:
            # Update with running average
            n = existing.iteration_count
            existing.actual_map_improvement = (existing.actual_map_improvement * n + actual_map_improvement) / (n + 1)
            existing.iteration_count = n + 1
            existing.timestamp = datetime.now().isoformat()
            self.logger.info(f"Updated memory entry (n={n+1}) for signature: {signature[:60]}")
        else:
            entry = MemoryEntry(
                weakness_signature=signature,
                weakness_embedding=embedding,
                config_delta=config_delta.to_dict(),
                actual_map_improvement=actual_map_improvement,
                experiment_run_id=run_id,
                timestamp=datetime.now().isoformat(),
            )
            self._entries.append(entry)

        if len(self._entries) > self.max_entries:
            # Evict lowest-value entries (negative improvement + low iteration count)
            self._entries.sort(
                key=lambda e: e.actual_map_improvement * (1 + 0.1 * e.iteration_count),
                reverse=True,
            )
            self._entries = self._entries[: self.max_entries]

        self._save()
        self.logger.info(
            f"Memory stored for run {run_id} | "
            f"Δ={actual_map_improvement:+.4f} | total={len(self._entries)}"
        )

    # ── Perception builder ────────────────────────────────────────────────────

    def _build_perception(self, weaknesses: list[dict], raw_recalls: list[dict]) -> str:
        ctx_block = self.shared_context.to_prompt_block() if self.shared_context else ""

        current_wstr = "\n".join(
            f"  - [{w.get('severity','?')}] {w.get('message', w.get('type', ''))}"
            for w in weaknesses[:5]
        )

        memories_str = ""
        for i, rec in enumerate(raw_recalls, 1):
            memories_str += (
                f"\n  [{i}] similarity={rec['similarity']:.3f} | "
                f"improvement={rec['actual_map_improvement']:+.4f} | "
                f"n_times={rec['iteration_count']}\n"
                f"      pattern: {rec['weakness_signature']}\n"
                f"      fix: {json.dumps(rec['config_delta'].get('changes', {}))}\n"
            )

        return "\n".join([
            ctx_block,
            "",
            "## Current Weaknesses",
            current_wstr,
            "",
            "## Raw Recalled Memories (cosine similarity >= threshold)",
            memories_str or "  (none)",
        ])

    def _fallback_curation(self, raw_recalls: list[dict]) -> dict:
        curated = []
        for rec in raw_recalls[: self.top_k_final]:
            curated.append({
                "run_id": rec["run_id"],
                "weakness_signature": rec["weakness_signature"],
                "config_delta_changes": rec["config_delta"].get("changes", {}),
                "original_improvement": rec["actual_map_improvement"],
                "adjusted_confidence": rec["similarity"] * 0.8,
                "relevance_reasoning": f"Similarity={rec['similarity']:.2f} (rule-based)",
                "is_warning": rec["actual_map_improvement"] < 0,
            })
        top_improvement = max((r["actual_map_improvement"] for r in raw_recalls), default=0)
        return {
            "curated_memories": curated,
            "synthesis": f"Found {len(raw_recalls)} past experiences. Best improvement: {top_improvement:+.4f}.",
            "should_store_current": True,
            "store_reasoning": "Default: store all experiences",
        }

    # ── Low-level vector store ────────────────────────────────────────────────

    def _raw_recall(self, weaknesses: list[dict], top_k: int) -> list[dict]:
        query_sig = self._build_signature(weaknesses)
        query_vec = text_to_feature_vector(query_sig, self.embedding_dim)

        scored = []
        for entry in self._entries:
            sim = compute_cosine_similarity(query_vec, entry.weakness_embedding)
            if sim >= self.similarity_threshold:
                scored.append((sim, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for sim, entry in scored[:top_k]:
            results.append({
                "similarity": float(sim),
                "weakness_signature": entry.weakness_signature,
                "config_delta": entry.config_delta,
                "actual_map_improvement": entry.actual_map_improvement,
                "run_id": entry.experiment_run_id,
                "timestamp": entry.timestamp,
                "iteration_count": entry.iteration_count,
            })
        return results

    @staticmethod
    def _build_signature(weaknesses: list[dict]) -> str:
        parts = sorted(
            f"{w.get('type', '')}:{w.get('affected', '')}:{w.get('severity', '')}"
            for w in weaknesses[:5]
        )
        return " | ".join(parts)

    def _load(self) -> None:
        if not self.store_path.exists():
            return
        try:
            with open(self.store_path, "rb") as f:
                data = pickle.load(f)
            self._entries = [MemoryEntry.from_dict(d) for d in data]
            self.logger.info(f"Loaded {len(self._entries)} memory entries")
        except Exception as exc:
            self.logger.warning(f"Could not load memory: {exc}")
            self._entries = []

    def _save(self) -> None:
        with open(self.store_path, "wb") as f:
            pickle.dump([e.to_serializable() for e in self._entries], f)

    def get_stats(self) -> dict:
        if not self._entries:
            return {"total": 0}
        improvements = [e.actual_map_improvement for e in self._entries]
        return {
            "total": len(self._entries),
            "avg_improvement": float(np.mean(improvements)),
            "max_improvement": float(np.max(improvements)),
            "positive_entries": sum(1 for v in improvements if v > 0),
            "warning_entries": sum(1 for v in improvements if v < 0),
        }
