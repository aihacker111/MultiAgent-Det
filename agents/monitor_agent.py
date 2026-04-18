from __future__ import annotations
from collections import deque
from pathlib import Path
from typing import Any

from agents.base_agent import BaseAgent
from core.event_bus import EventBus, Event, EventType
from core.shared_context import SharedContext
from utils.metrics import EpochMetrics
from utils.visualization import plot_training_curves

# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a training dynamics expert agent monitoring a YOLO object detection training run in real time.

Your role is to observe epoch-level metrics and decide:
1. Whether training is healthy or has a problem worth alerting
2. What type of problem it is (overfitting, plateau, gradient_explosion, lr_collapse, none)
3. Whether to trigger early stopping immediately
4. What dynamic thresholds make sense given this specific dataset and history

CRITICAL RULES — read carefully before deciding:
- Early epochs (< 10) are always volatile. NEVER suggest early stopping before epoch 10.
- High gradient norm in epoch 1-3 is NORMAL — model is warming up. Only flag if it persists beyond epoch 5.
- If train_loss AND map50 are BOTH exactly 0.0, the metrics extraction likely FAILED — treat this as HEALTHY (data error, not training error).
- A single bad epoch does NOT warrant stopping. Look for SUSTAINED trends over at least 3-5 epochs.
- mAP50 starting at 0.1-0.3 in early epochs and val_loss being high is completely NORMAL for YOLO.
- Only flag lr_collapse if lr < 1e-7 AND this has persisted for multiple epochs.
- Only flag overfitting if val_loss has been CONSISTENTLY increasing for 5+ consecutive epochs while train_loss decreases.

Be conservative with early stopping — only stop if you are HIGHLY confident continuing wastes compute."""

OUTPUT_SCHEMA = """{
  "assessment": "healthy | overfitting | plateau | gradient_explosion | lr_collapse",
  "should_alert": true | false,
  "should_stop": true | false,
  "severity": "low | medium | high | critical",
  "reasoning": "brief explanation of your decision",
  "dynamic_patience": integer,
  "message": "human-readable alert message or empty string if healthy"
}"""
# ─────────────────────────────────────────────────────────────────────────────


class MonitorAgent(BaseAgent):
    """
    Real-time training monitor via Ultralytics callbacks.

    Callback execution order per epoch:
      on_train_epoch_end  → training done, grad still in memory, NO val metrics yet
      on_val_end          → validation done, val losses available
      on_fit_epoch_end    → everything done, ALL metrics available — only place to decide

    Key design:
    - on_train_epoch_end: ONLY collect grad_norm into window, never alert
    - on_fit_epoch_end:   full check with all metrics
    - Early stop: set trainer.stop_training = True (Ultralytics-native, safe)
    - Min epoch guard: no stop/critical alert before epoch min_epochs_before_check
    - Sanity check: skip if train_loss=0 AND map50=0 (bad extraction)
    """

    AGENT_NAME = "MonitorAgent"
    SYSTEM_PROMPT = SYSTEM_PROMPT

    def _setup(self) -> None:
        cfg = self.config.get("monitor", {})
        self._fallback_overfitting_patience: int = cfg.get("overfitting_patience", 5)
        self._fallback_plateau_patience: int = cfg.get("plateau_patience", 8)
        self._fallback_plateau_threshold: float = cfg.get("plateau_threshold", 0.001)
        self._fallback_grad_norm_max: float = cfg.get("grad_norm_max", 100.0)
        self._fallback_lr_collapse: float = cfg.get("lr_collapse_threshold", 1e-7)
        self.min_epochs_before_check: int = cfg.get("min_epochs_before_check", 10)
        self.llm_check_interval: int = cfg.get("llm_check_interval", 5)

        self.history: list[EpochMetrics] = []
        self._val_loss_window: deque = deque(maxlen=10)
        self._train_loss_window: deque = deque(maxlen=10)
        self._map_window: deque = deque(maxlen=10)
        self._grad_norm_window: deque = deque(maxlen=5)
        self._should_stop: bool = False
        self.alerts: list[dict] = []
        self._dynamic_patience: int = self._fallback_plateau_patience
        self._trainer_ref = None  # kept for stop signal

    # ── Ultralytics callbacks ─────────────────────────────────────────────────

    def get_callbacks(self) -> dict[str, Any]:
        return {
            "on_train_epoch_end": self._on_train_epoch_end,
            "on_val_end": self._on_val_end,
            "on_fit_epoch_end": self._on_fit_epoch_end,
        }

    def _on_train_epoch_end(self, trainer) -> None:
        """Only collect grad_norm. DO NOT alert here — val metrics not available."""
        try:
            self._trainer_ref = trainer
            loss_items = getattr(trainer, "loss_items", None)
            if loss_items is not None and len(loss_items) >= 3:
                self._train_loss_window.append(float(sum(loss_items[:3])))
            grad_norm = self._get_grad_norm(trainer)
            if grad_norm > 0:
                self._grad_norm_window.append(grad_norm)
        except Exception as exc:
            self.logger.debug(f"on_train_epoch_end: {exc}")

    def _on_val_end(self, validator) -> None:
        try:
            loss_items = getattr(validator, "loss_items", None)
            if loss_items is not None and len(loss_items) >= 3:
                self._val_loss_window.append(float(sum(loss_items[:3])))
        except Exception as exc:
            self.logger.debug(f"on_val_end: {exc}")

    def _on_fit_epoch_end(self, trainer) -> None:
        """All metrics available here. Decision point."""
        try:
            self._trainer_ref = trainer

            # Apply stop if previously decided — use Ultralytics-native flag
            if self._should_stop:
                trainer.stop_training = True
                self.event_bus.publish(Event(
                    type=EventType.MONITOR_EARLY_STOP,
                    source=self.AGENT_NAME,
                    data={"epoch": getattr(trainer, "epoch", 0), "alerts": self.alerts},
                ))
                return

            metrics = self._extract_metrics(trainer)

            # Skip silently if data extraction clearly failed
            if self._metrics_look_corrupted(metrics):
                self.logger.debug(
                    f"Epoch {metrics.epoch}: extraction looks corrupted "
                    f"(train_loss={metrics.train_loss_total:.4f}, map50={metrics.map50:.4f}) — skipped"
                )
                self.event_bus.publish(Event(type=EventType.EPOCH_END, source=self.AGENT_NAME, data=metrics))
                return

            self.history.append(metrics)
            self._map_window.append(metrics.map50)

            epoch = metrics.epoch
            total_epochs = getattr(trainer, "epochs", 100)
            is_last = epoch >= total_epochs - 1

            # LLM check after min_epochs, every llm_check_interval epochs
            should_llm_check = (
                epoch >= self.min_epochs_before_check
                and ((epoch + 1) % self.llm_check_interval == 0 or is_last)
            )

            if should_llm_check:
                self._check_and_alert(metrics, trainer)
            else:
                self._lightweight_check(metrics, trainer)

            self.event_bus.publish(Event(type=EventType.EPOCH_END, source=self.AGENT_NAME, data=metrics))

        except Exception as exc:
            self.logger.debug(f"on_fit_epoch_end: {exc}")

    # ── Decision logic ────────────────────────────────────────────────────────

    def _check_and_alert(self, metrics: EpochMetrics, trainer=None) -> None:
        """Full LLM check — only after min_epochs_before_check."""
        perception = self._build_perception(metrics)
        decision = self.reason(perception, OUTPUT_SCHEMA)
        if not decision:
            decision = self._fallback_decision(metrics)
        self._apply_decision(decision, metrics, trainer)

    def _lightweight_check(self, metrics: EpochMetrics, trainer=None) -> None:
        """
        Rule-based only for early epochs.
        Only catches truly catastrophic failures that persist 3+ epochs.
        """
        norms = list(self._grad_norm_window)
        if len(norms) >= 3 and all(n > self._fallback_grad_norm_max * 5 for n in norms[-3:]):
            alert = {
                "type": "gradient_explosion", "epoch": metrics.epoch, "severity": "high",
                "message": f"Sustained gradient explosion: {[f'{n:.0f}' for n in norms[-3:]]}",
                "reasoning": "Lightweight rule: 3 consecutive epochs of extreme grad norm",
                "source": "rule",
            }
            self.alerts.append(alert)
            self.logger.warning(f"[orange3]MONITOR [HIGH][/orange3] {alert['message']}")
            self.event_bus.publish(Event(type=EventType.MONITOR_ALERT, source=self.AGENT_NAME, data=alert))

        if 0 < metrics.lr < self._fallback_lr_collapse and metrics.epoch > 3:
            alert = {
                "type": "lr_collapse", "epoch": metrics.epoch, "severity": "critical",
                "message": f"LR collapsed to {metrics.lr:.2e}",
                "reasoning": "Lightweight rule: LR collapse", "source": "rule",
            }
            self.alerts.append(alert)
            self.logger.warning(f"[red]MONITOR [CRITICAL][/red] {alert['message']}")
            self.event_bus.publish(Event(type=EventType.MONITOR_ALERT, source=self.AGENT_NAME, data=alert))
            self._apply_stop(trainer, metrics.epoch, "LR collapse")

    def _apply_decision(self, decision: dict, metrics: EpochMetrics, trainer=None) -> None:
        assessment = decision.get("assessment", "healthy")
        should_alert = decision.get("should_alert", False)
        should_stop = decision.get("should_stop", False)
        severity = decision.get("severity", "low")
        message = decision.get("message", "")
        self._dynamic_patience = int(decision.get("dynamic_patience", self._fallback_plateau_patience))

        if should_alert and message:
            alert = {
                "type": assessment, "epoch": metrics.epoch, "severity": severity,
                "message": message, "reasoning": decision.get("reasoning", ""),
                "source": "llm" if self._llm else "rule",
            }
            self.alerts.append(alert)
            color = {"critical": "red", "high": "orange3", "medium": "yellow"}.get(severity, "white")
            self.logger.warning(f"[{color}]MONITOR [{severity.upper()}][/{color}] {message}")
            self.event_bus.publish(Event(type=EventType.MONITOR_ALERT, source=self.AGENT_NAME, data=alert))
            if self.shared_context:
                self.shared_context.current_alerts.append(alert)

        if should_stop:
            self._apply_stop(trainer, metrics.epoch, decision.get("reasoning", ""))

    def _apply_stop(self, trainer, epoch: int, reason: str) -> None:
        """Use Ultralytics-native stop flag — safe, no exception needed."""
        self.logger.warning(f"[red]EARLY STOP[/red] @ epoch {epoch}: {reason}")
        self._should_stop = True
        if trainer is not None:
            trainer.stop_training = True

    # ── Perception ────────────────────────────────────────────────────────────

    def _build_perception(self, metrics: EpochMetrics) -> str:
        hist_tail = self.history[-10:]
        history_str = "\n".join(
            f"  epoch={m.epoch}: train={m.train_loss_total:.4f}, "
            f"val={m.val_loss_total:.4f}, mAP50={m.map50:.4f}, lr={m.lr:.2e}"
            for m in hist_tail
        )
        grad_trend = [f"{g:.1f}" for g in self._grad_norm_window]
        ctx = self.shared_context.to_prompt_block() if self.shared_context else ""
        return "\n".join([
            ctx, "",
            "## Current Epoch (all values valid — post-validation)",
            f"- Epoch: {metrics.epoch}",
            f"- Train loss: {metrics.train_loss_total:.4f} "
            f"(box={metrics.train_box_loss:.4f}, cls={metrics.train_cls_loss:.4f}, dfl={metrics.train_dfl_loss:.4f})",
            f"- Val loss: {metrics.val_loss_total:.4f}",
            f"- mAP50: {metrics.map50:.4f} | mAP50-95: {metrics.map50_95:.4f}",
            f"- LR: {metrics.lr:.2e}",
            f"- Recent grad norms (last {len(grad_trend)} train epochs): {grad_trend}",
            "", f"## History (last {len(hist_tail)} epochs)",
            history_str or "  (none yet)",
        ])

    def _fallback_decision(self, metrics: EpochMetrics) -> dict:
        maps = list(self._map_window)
        val_losses = list(self._val_loss_window)
        train_losses = list(self._train_loss_window)
        norms = list(self._grad_norm_window)

        if len(norms) >= 3 and all(n > self._fallback_grad_norm_max for n in norms[-3:]):
            return {"assessment": "gradient_explosion", "should_alert": True, "should_stop": False,
                    "severity": "high", "dynamic_patience": 5,
                    "message": f"Sustained gradient explosion: {[f'{n:.0f}' for n in norms[-3:]]}",
                    "reasoning": "Rule: sustained grad norm"}

        if 0 < metrics.lr < self._fallback_lr_collapse:
            return {"assessment": "lr_collapse", "should_alert": True, "should_stop": True,
                    "severity": "critical", "dynamic_patience": 0,
                    "message": f"LR collapsed to {metrics.lr:.2e}", "reasoning": "Rule: LR collapse"}

        n_ov = self._fallback_overfitting_patience
        if len(val_losses) >= n_ov and len(train_losses) >= n_ov:
            if all(val_losses[i+1] > val_losses[i] + 0.005 for i in range(len(val_losses)-1)) \
                    and train_losses[-1] < train_losses[0]:
                return {"assessment": "overfitting", "should_alert": True, "should_stop": False,
                        "severity": "high", "dynamic_patience": 3,
                        "message": f"Val loss rising for {n_ov} epochs while train falls",
                        "reasoning": "Rule: overfitting"}

        n_pl = self._fallback_plateau_patience
        if len(maps) >= n_pl and (max(maps) - min(maps)) < self._fallback_plateau_threshold:
            return {"assessment": "plateau", "should_alert": True, "should_stop": True,
                    "severity": "medium", "dynamic_patience": 0,
                    "message": f"mAP50 plateaued at {metrics.map50:.4f} for {n_pl} epochs",
                    "reasoning": "Rule: plateau"}

        return {"assessment": "healthy", "should_alert": False, "should_stop": False,
                "severity": "low", "dynamic_patience": self._fallback_plateau_patience,
                "message": "", "reasoning": "Rule: healthy"}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _extract_metrics(self, trainer) -> EpochMetrics:
        epoch = getattr(trainer, "epoch", len(self.history))
        m = EpochMetrics(epoch=epoch)
        metrics_dict = getattr(trainer, "metrics", {}) or {}
        m.map50 = float(metrics_dict.get("metrics/mAP50(B)", 0.0))
        m.map50_95 = float(metrics_dict.get("metrics/mAP50-95(B)", 0.0))
        loss_items = getattr(trainer, "loss_items", None)
        if loss_items is not None and len(loss_items) >= 3:
            m.train_box_loss, m.train_cls_loss, m.train_dfl_loss = map(float, loss_items[:3])
        validator = getattr(trainer, "validator", None)
        if validator:
            vl = getattr(validator, "loss_items", None)
            if vl is not None and len(vl) >= 3:
                m.val_box_loss, m.val_cls_loss, m.val_dfl_loss = map(float, vl[:3])
        pg = getattr(getattr(trainer, "optimizer", None), "param_groups", None)
        if pg:
            m.lr = float(pg[0]["lr"])
        norms = list(self._grad_norm_window)
        m.grad_norm = norms[-1] if norms else 0.0
        return m

    @staticmethod
    def _metrics_look_corrupted(m: EpochMetrics) -> bool:
        if m.epoch == 0:
            return False
        return m.train_loss_total == 0.0 and m.map50 == 0.0

    @staticmethod
    def _get_grad_norm(trainer) -> float:
        try:
            import torch
            model = getattr(trainer, "model", None)
            if model is None:
                return 0.0
            return float(sum(
                p.grad.data.norm(2).item() ** 2
                for p in model.parameters() if p.grad is not None
            ) ** 0.5)
        except Exception:
            return 0.0

    def should_stop(self) -> bool:
        return self._should_stop

    def reset(self) -> None:
        self.history.clear()
        self._val_loss_window.clear()
        self._train_loss_window.clear()
        self._map_window.clear()
        self._grad_norm_window.clear()
        self._should_stop = False
        self.alerts.clear()
        self._trainer_ref = None

    def save_plots(self, run_id: str) -> None:
        if self.history:
            plot_training_curves(self.history, self.experiment_dir / run_id / "plots" / "training_curves.png")

    def run(self, *args, **kwargs):
        pass
