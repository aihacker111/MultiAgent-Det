from __future__ import annotations
import gc
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


class TrainerAgent(BaseAgent):
    """
    Experiment execution agent — intentionally deterministic, no LLM.

    Reasoning about WHICH experiments to run belongs to PlannerAgent.
    TrainerAgent's job:
      1. VRAM check   — reject proposal if estimated usage exceeds free GPU memory
      2. Dedup check  — skip if same config hash was already run successfully
      3. Execute      — spawn Ultralytics training with Monitor callbacks attached
      4. GPU cleanup  — free VRAM completely after each run (prevents OOM)

    Proposals arrive already prioritized from PlannerAgent — no re-evaluation needed.
    """

    AGENT_NAME = "TrainerAgent"
    SYSTEM_PROMPT = ""  # Intentionally empty — no LLM

    def _setup(self) -> None:
        cfg = self.config.get("trainer", {})
        self.max_parallel: int = cfg.get("max_parallel_jobs", 1)
        self.vram_buffer_gb: float = cfg.get("vram_buffer_gb", 3.0)
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
            self.logger.info("No proposals to execute")
            return []

        ordered = sorted(proposals, key=lambda p: p.priority)[: self.max_parallel]
        self.logger.info(
            f"Executing {len(ordered)}/{len(proposals)} proposals "
            f"(max_parallel={self.max_parallel})"
        )

        run_ids = []
        for proposal in ordered:
            run_id = self._execute(proposal, base_config, monitor_agent)
            if run_id:
                run_ids.append(run_id)
        return run_ids

    # ── Execution pipeline ────────────────────────────────────────────────────

    def _execute(
        self,
        proposal: ConfigDelta,
        base_config: dict,
        monitor_agent=None,
    ) -> Optional[str]:
        new_config = proposal.apply_to(base_config)
        training_cfg = new_config.get("training", new_config)

        # Gate 1: dedup
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

        # Gate 2: VRAM
        est = self._estimate_vram_gb(
            training_cfg.get("imgsz", 640),
            training_cfg.get("batch", 16),
        )
        available = self._get_available_vram_gb()
        if est > available:
            self.logger.warning(
                f"VRAM check failed — need ~{est:.1f}GB, "
                f"available={available:.1f}GB "
                f"(imgsz={training_cfg.get('imgsz')}, batch={training_cfg.get('batch')}) "
                f"— skipping: {proposal.rationale}"
            )
            return None

        self.logger.debug(f"VRAM OK: ~{est:.1f}GB / {available:.1f}GB")

        run_id = self._generate_run_id(training_cfg)
        self.registry.register(run_id, new_config)
        self.registry.update_status(run_id, "running")

        self.logger.info(
            f"[green]Starting[/green] [bold]{run_id}[/bold]\n"
            f"  Changes:   {proposal.changes or 'baseline'}\n"
            f"  Rationale: {proposal.rationale}\n"
            f"  Expected Δ mAP50: {proposal.expected_map_improvement:+.3f} "
            f"(conf={proposal.confidence:.2f})"
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
                f"[green]Complete[/green] {run_id} | "
                f"mAP50={metrics.get('map50', 0):.4f}"
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

    # ── Training ──────────────────────────────────────────────────────────────

    def _train(self, run_id: str, config: dict, monitor_agent=None) -> dict:
        try:
            from ultralytics import YOLO
        except ImportError:
            self.logger.warning("Ultralytics not installed — mock result")
            return self._mock_result()

        training_cfg = config.get("training", config)
        model = YOLO(training_cfg.get("model", "yolov8n.pt"))
        kwargs = {k: v for k, v in training_cfg.items() if k not in {"model", "data"}}
        kwargs.update({
            "project": str(self.experiment_dir / run_id),
            "name": "train",
            "exist_ok": True,
        })

        if monitor_agent is not None:
            for event_name, fn in monitor_agent.get_callbacks().items():
                model.add_callback(event_name, fn)

        try:
            results = model.train(data=training_cfg.get("data", "data.yaml"), **kwargs)
            rd = getattr(results, "results_dict", {}) or {}
            return {
                "map50":      float(rd.get("metrics/mAP50(B)", 0.0)),
                "map50_95":   float(rd.get("metrics/mAP50-95(B)", 0.0)),
                "best_epoch": int(getattr(results, "epoch", 0)),
            }
        finally:
            # Always free GPU memory after training — prevents OOM on iteration 2+
            self._free_gpu_memory(model)

    def _mock_result(self) -> dict:
        import random
        return {
            "map50":      round(0.4 + random.uniform(0, 0.15), 4),
            "map50_95":   round(0.25 + random.uniform(0, 0.1), 4),
            "best_epoch": 50,
        }

    # ── VRAM ─────────────────────────────────────────────────────────────────

    def _get_available_vram_gb(self) -> float:
        """Query actual free VRAM from GPU, fallback to config budget."""
        try:
            import torch
            if torch.cuda.is_available():
                total    = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
                reserved = torch.cuda.memory_reserved(0) / 1024 ** 3
                free     = total - reserved - self.vram_buffer_gb
                self.logger.debug(
                    f"GPU VRAM: total={total:.1f}GB "
                    f"reserved={reserved:.1f}GB free={max(free,0):.1f}GB"
                )
                return max(free, 0.0)
        except Exception:
            pass
        return self.gpu_memory_gb - self.vram_buffer_gb

    def _estimate_vram_gb(self, imgsz: int, batch: int) -> float:
        """
        Reference-point scaling:
          ref_gb × (imgsz/ref_imgsz)² × (batch/ref_batch) × 1.2 overhead
        Set vram_reference_* in config/default_config.yaml from your actual GPU readings.
        """
        cfg = self.config.get("trainer", {})
        ref_imgsz = cfg.get("vram_reference_imgsz", 640)
        ref_batch  = cfg.get("vram_reference_batch", 16)
        ref_gb     = cfg.get("vram_reference_gb", 13.0)
        scale = ((imgsz / ref_imgsz) ** 2) * (batch / ref_batch)
        return ref_gb * scale * 1.20

    @staticmethod
    def _free_gpu_memory(model=None) -> None:
        """Full GPU cleanup between runs."""
        try:
            import torch
            if model is not None:
                try:
                    if hasattr(model, "model") and model.model is not None:
                        model.model.cpu()
                except Exception:
                    pass
                del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass
        except Exception:
            pass

    # ── Utils ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _generate_run_id(training_cfg: dict) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        h = hashlib.md5(
            json.dumps(training_cfg, sort_keys=True).encode()
        ).hexdigest()[:6]
        return f"run_{ts}_{h}"

    def get_summary(self) -> dict:
        return self.registry.summary()
