from __future__ import annotations
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from agents.base_agent import BaseAgent
from core.event_bus import EventBus, Event, EventType
from core.experiment_registry import ExperimentRegistry
from core.shared_context import SharedContext
from utils.metrics import ConfigDelta

# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an experiment scheduling and resource management agent for YOLO training.

You receive a list of proposed experiment configs from the Planner and must decide:
1. Which proposals to ACCEPT (considering compute budget, risk, expected value)
2. What ORDER to run them (highest expected value first, unless resource constraints dictate otherwise)
3. Whether to MERGE compatible proposals (e.g. two proposals that each change 2 params could be merged into one efficient experiment)
4. Whether to FLAG proposals as risky (large parameter jumps, untested combinations)

Decision criteria:
- Prefer proposals with high confidence AND high expected improvement
- Avoid running a proposal that failed in a previous iteration (check history)
- If compute is limited (few iterations remaining), be conservative — pick the single safest high-confidence proposal
- A proposal changing > 3 parameters simultaneously is risky — flag it
- If previous iterations showed no improvement, prefer exploratory proposals over incremental ones

Output a final execution plan."""

OUTPUT_SCHEMA = """{
  "accepted": [
    {
      "proposal_index": integer (0-based index in input proposals list),
      "priority": integer,
      "risk_level": "low | medium | high",
      "risk_reason": "why risky or empty string",
      "scheduling_note": "any special instruction for this run",
      "estimated_vram_ok": true | false
    }
  ],
  "rejected": [
    {
      "proposal_index": integer,
      "reason": "why rejected"
    }
  ],
  "merge_suggestion": null | {
    "indices": [0, 1],
    "merged_changes": {"param": value},
    "rationale": "why merge makes sense"
  },
  "execution_reasoning": "overall explanation of scheduling decisions"
}"""
# ─────────────────────────────────────────────────────────────────────────────


class TrainerAgent(BaseAgent):
    """
    Experiment scheduling + execution agent.
    LLM evaluates proposals, decides accept/reject/merge/order.
    Deterministic tools: VRAM check, dedup registry, Ultralytics runner.
    """

    AGENT_NAME = "TrainerAgent"
    SYSTEM_PROMPT = SYSTEM_PROMPT

    def _setup(self) -> None:
        cfg = self.config.get("trainer", {})
        self.max_parallel: int = cfg.get("max_parallel_jobs", 1)
        self.vram_buffer_gb: float = cfg.get("vram_buffer_gb", 2.0)
        self.dedup_check: bool = cfg.get("dedup_check", True)
        self.gpu_memory_gb: float = self.config.get("system", {}).get("gpu_memory_gb", 16.0)

        self.registry = ExperimentRegistry(self.experiment_dir / "registry.db")

    def run(
        self,
        proposals: list[ConfigDelta],
        base_config: dict,
        monitor_agent=None,
    ) -> list[str]:
        if not proposals:
            self.logger.info("No proposals to evaluate")
            return []

        # Step 1: perceive — describe proposals + context for LLM
        perception = self._build_perception(proposals, base_config)

        # Step 2: reason — LLM decides accept/reject/order/merge
        decision = self.reason(perception, OUTPUT_SCHEMA)

        if not decision:
            decision = self._fallback_decision(proposals, base_config)

        self.logger.info(f"Trainer decision: {decision.get('execution_reasoning', '')}")

        if self.shared_context:
            self.shared_context.trainer_decision = decision

        # Step 3: act — execute accepted proposals
        final_proposals = self._apply_decision(decision, proposals, base_config)
        run_ids = []
        for proposal in final_proposals:
            run_id = self._execute(proposal, base_config, monitor_agent)
            if run_id:
                run_ids.append(run_id)

        return run_ids

    # ── Perception builder ────────────────────────────────────────────────────

    def _build_perception(self, proposals: list[ConfigDelta], base_config: dict) -> str:
        ctx_block = self.shared_context.to_prompt_block() if self.shared_context else ""
        available_vram = self.gpu_memory_gb - self.vram_buffer_gb
        iterations_left = (
            self.shared_context.max_iterations - self.shared_context.iteration
            if self.shared_context else "unknown"
        )

        props_str = ""
        for i, p in enumerate(proposals):
            vram_est = self._estimate_vram_gb(
                base_config.get("training", base_config).get("imgsz", 640),
                base_config.get("training", base_config).get("batch", 16),
            )
            props_str += (
                f"\n  [{i}] {p.rationale}\n"
                f"      changes: {json.dumps(p.changes)}\n"
                f"      confidence={p.confidence:.2f}, expected_improvement={p.expected_map_improvement:+.3f}\n"
                f"      estimated_vram={vram_est:.1f}GB\n"
            )

        history_str = ""
        if self.shared_context and self.shared_context.iteration_history:
            last3 = self.shared_context.iteration_history[-3:]
            history_str = "\n".join(
                f"  iter {s.iteration}: mAP50={s.map50:.4f}, changes={json.dumps(s.config_changes)}"
                for s in last3
            )

        return "\n".join([
            ctx_block,
            "",
            f"## Resources",
            f"- Available VRAM: {available_vram:.1f}GB",
            f"- Iterations remaining: {iterations_left}",
            f"- Max parallel jobs: {self.max_parallel}",
            "",
            "## Proposals to Evaluate",
            props_str,
            "",
            "## Recent Experiment History",
            history_str or "  (none yet)",
        ])

    # ── Decision application ──────────────────────────────────────────────────

    def _apply_decision(
        self,
        decision: dict,
        proposals: list[ConfigDelta],
        base_config: dict,
    ) -> list[ConfigDelta]:
        accepted_indices = sorted(
            decision.get("accepted", []),
            key=lambda a: a.get("priority", 99),
        )

        # Check merge suggestion first
        merge = decision.get("merge_suggestion")
        if merge and merge.get("indices") and merge.get("merged_changes"):
            merged = ConfigDelta(
                changes=merge["merged_changes"],
                rationale=f"[MERGED] {merge.get('rationale', '')}",
                confidence=min(
                    proposals[i].confidence
                    for i in merge["indices"]
                    if i < len(proposals)
                ),
                expected_map_improvement=sum(
                    proposals[i].expected_map_improvement
                    for i in merge["indices"]
                    if i < len(proposals)
                ) * 0.7,  # conservative: merging compounds uncertainty
                priority=1,
            )
            self.logger.info(f"Merging proposals {merge['indices']}: {merge.get('rationale')}")
            return [merged]

        result = []
        for acc in accepted_indices:
            idx = acc.get("proposal_index", -1)
            if 0 <= idx < len(proposals):
                p = proposals[idx]
                risk = acc.get("risk_level", "low")
                if risk == "high":
                    self.logger.warning(
                        f"High-risk proposal accepted: {acc.get('risk_reason', '')} — proceeding"
                    )
                result.append(p)

        if not result and proposals:
            self.logger.warning("LLM rejected all proposals — using top-confidence fallback")
            result = [max(proposals, key=lambda p: p.confidence * p.expected_map_improvement)]

        return result[: self.max_parallel]

    def _fallback_decision(self, proposals: list[ConfigDelta], base_config: dict) -> dict:
        """Rule-based fallback."""
        training_cfg = base_config.get("training", base_config)
        accepted = []
        rejected = []
        for i, p in enumerate(proposals):
            vram_ok = self._vram_check({
                **training_cfg,
                **p.changes,
            })
            if vram_ok and p.confidence >= 0.6:
                accepted.append({
                    "proposal_index": i,
                    "priority": i + 1,
                    "risk_level": "low" if len(p.changes) <= 2 else "medium",
                    "risk_reason": "",
                    "scheduling_note": "",
                    "estimated_vram_ok": True,
                })
            else:
                rejected.append({
                    "proposal_index": i,
                    "reason": "Low confidence or VRAM exceeded" if not vram_ok else "Low confidence",
                })
        return {
            "accepted": accepted[:self.max_parallel],
            "rejected": rejected,
            "merge_suggestion": None,
            "execution_reasoning": "Rule-based fallback scheduling",
        }

    # ── Execution (deterministic) ─────────────────────────────────────────────

    def _execute(
        self,
        proposal: ConfigDelta,
        base_config: dict,
        monitor_agent=None,
    ) -> Optional[str]:
        new_config = proposal.apply_to(base_config)
        training_cfg = new_config.get("training", new_config)

        if self.dedup_check:
            existing = self.registry.is_duplicate(training_cfg)
            if existing:
                self.logger.warning(
                    f"Duplicate config (hash={existing.config_hash[:8]}, "
                    f"run={existing.run_id}) — skipping"
                )
                self.event_bus.publish(Event(
                    type=EventType.EXPERIMENT_DUPLICATE,
                    source=self.AGENT_NAME,
                    data={"existing_run_id": existing.run_id},
                ))
                return None

        if not self._vram_check(training_cfg):
            self.logger.error(
                f"VRAM exceeded for imgsz={training_cfg.get('imgsz')}, "
                f"batch={training_cfg.get('batch')} — skipping"
            )
            return None

        run_id = self._generate_run_id(training_cfg)
        self.registry.register(run_id, new_config)
        self.registry.update_status(run_id, "running")

        self.logger.info(
            f"[green]Starting[/green] [bold]{run_id}[/bold]\n"
            f"  Changes: {proposal.changes}\n"
            f"  Rationale: {proposal.rationale}"
        )

        self.event_bus.publish(Event(
            type=EventType.TRAINING_START,
            source=self.AGENT_NAME,
            data={"run_id": run_id, "config": new_config, "rationale": proposal.rationale},
        ))

        try:
            metrics = self._train(run_id, new_config, monitor_agent)
            self.registry.update_status(
                run_id, "completed",
                final_map50=metrics.get("map50", 0.0),
                final_map50_95=metrics.get("map50_95", 0.0),
                best_epoch=metrics.get("best_epoch", 0),
            )
            self.event_bus.publish(Event(
                type=EventType.TRAINING_COMPLETE,
                source=self.AGENT_NAME,
                data={"run_id": run_id, "metrics": metrics},
            ))
            self.logger.info(
                f"[green]Complete[/green] {run_id} | mAP50={metrics.get('map50', 0):.4f}"
            )
        except Exception as exc:
            self.logger.error(f"Training failed {run_id}: {exc}")
            self.registry.update_status(run_id, "failed", notes=str(exc))
            self.event_bus.publish(Event(
                type=EventType.TRAINING_FAILED,
                source=self.AGENT_NAME,
                data={"run_id": run_id, "error": str(exc)},
            ))

        return run_id

    def _train(self, run_id: str, config: dict, monitor_agent=None) -> dict:
        try:
            from ultralytics import YOLO
        except ImportError:
            self.logger.warning("Ultralytics not installed — returning mock result")
            return self._mock_result()

        training_cfg = config.get("training", config)
        model = YOLO(training_cfg.get("model", "yolov8n.pt"))
        kwargs = {k: v for k, v in training_cfg.items() if k not in {"model", "data"}}
        kwargs.update({"project": str(self.experiment_dir / run_id), "name": "train", "exist_ok": True})

        if monitor_agent is not None:
            for event_name, fn in monitor_agent.get_callbacks().items():
                model.add_callback(event_name, fn)

        results = model.train(data=training_cfg.get("data", "data.yaml"), **kwargs)
        rd = getattr(results, "results_dict", {}) or {}
        return {
            "map50": float(rd.get("metrics/mAP50(B)", 0.0)),
            "map50_95": float(rd.get("metrics/mAP50-95(B)", 0.0)),
            "best_epoch": getattr(results, "epoch", 0),
        }

    def _mock_result(self) -> dict:
        import random
        return {"map50": 0.4 + random.uniform(0, 0.15), "map50_95": 0.25 + random.uniform(0, 0.1), "best_epoch": 50}

    def _vram_check(self, training_cfg: dict) -> bool:
        imgsz = training_cfg.get("imgsz", 640)
        batch = training_cfg.get("batch", 16)
        est = self._estimate_vram_gb(imgsz, batch)
        ok = est <= (self.gpu_memory_gb - self.vram_buffer_gb)
        if not ok:
            self.logger.warning(f"VRAM: ~{est:.1f}GB needed, {self.gpu_memory_gb - self.vram_buffer_gb:.1f}GB available")
        return ok

    @staticmethod
    def _estimate_vram_gb(imgsz: int, batch: int) -> float:
        return (imgsz * imgsz * 3 * 4 * batch * 8.0) / (1024 ** 3)

    @staticmethod
    def _generate_run_id(training_cfg: dict) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        h = hashlib.md5(json.dumps(training_cfg, sort_keys=True).encode()).hexdigest()[:6]
        return f"run_{ts}_{h}"

    def get_summary(self) -> dict:
        return self.registry.summary()
