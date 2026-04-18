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

You have access to the full training history and dataset profile.
You reason carefully before acting — do NOT alert on noise or single-epoch fluctuations.
You adapt your sensitivity based on context: a small dataset tolerates more variance; early epochs have more volatility.

Be conservative with early stopping — only stop if you are highly confident continuing wastes compute."""

OUTPUT_SCHEMA = """{
  "assessment": "healthy | overfitting | plateau | gradient_explosion | lr_collapse",
  "should_alert": true | false,
  "should_stop": true | false,
  "severity": "low | medium | high | critical",
  "reasoning": "brief explanation of your decision",
  "dynamic_patience": integer (how many more epochs to tolerate before stopping),
  "message": "human-readable alert message"
}"""
# ─────────────────────────────────────────────────────────────────────────────


class MonitorAgent(BaseAgent):
    """
    Real-time training monitor. Hooks into Ultralytics callbacks.
    LLM decides assessment + early-stop based on full history + shared context.
    Falls back to rule-based detection when no API key.
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

        self.history: list[EpochMetrics] = []
        self._val_loss_window: deque = deque(maxlen=10)
        self._train_loss_window: deque = deque(maxlen=10)
        self._map_window: deque = deque(maxlen=10)
        self._should_stop: bool = False
        self.alerts: list[dict] = []
        self._dynamic_patience: int = self._fallback_plateau_patience

    # ── Ultralytics callback hooks ────────────────────────────────────────────

    def get_callbacks(self) -> dict[str, Any]:
        return {
            "on_train_epoch_end": self._on_train_epoch_end,
            "on_val_end": self._on_val_end,
            "on_fit_epoch_end": self._on_fit_epoch_end,
        }

    def _on_train_epoch_end(self, trainer) -> None:
        try:
            loss_items = getattr(trainer, "loss_items", None)
            if loss_items is not None and len(loss_items) >= 3:
                self._train_loss_window.append(float(sum(loss_items[:3])))
            grad_norm = self._get_grad_norm(trainer)
            if grad_norm > self._fallback_grad_norm_max:
                m = EpochMetrics(epoch=getattr(trainer, "epoch", 0), grad_norm=grad_norm)
                self._check_and_alert(m, force_check=True)
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
        try:
            metrics = self._extract_metrics(trainer)
            self.history.append(metrics)
            self._map_window.append(metrics.map50)

            # LLM check every 5 epochs (or last epoch) to save API calls
            epoch = metrics.epoch
            total_epochs = getattr(trainer, "epochs", 100)
            is_last = epoch >= total_epochs - 1
            if epoch % 5 == 0 or is_last:
                self._check_and_alert(metrics)

            self.event_bus.publish(Event(
                type=EventType.EPOCH_END,
                source=self.AGENT_NAME,
                data=metrics,
            ))

            if self._should_stop:
                self.event_bus.publish(Event(
                    type=EventType.MONITOR_EARLY_STOP,
                    source=self.AGENT_NAME,
                    data={"epoch": epoch, "alerts": self.alerts},
                ))
        except Exception as exc:
            self.logger.debug(f"on_fit_epoch_end: {exc}")

    # ── Core reasoning ────────────────────────────────────────────────────────

    def _check_and_alert(self, metrics: EpochMetrics, force_check: bool = False) -> None:
        perception = self._build_perception(metrics)
        decision = self.reason(perception, OUTPUT_SCHEMA)

        if not decision:
            decision = self._fallback_decision(metrics)

        assessment = decision.get("assessment", "healthy")
        should_alert = decision.get("should_alert", False)
        should_stop = decision.get("should_stop", False)
        severity = decision.get("severity", "low")
        message = decision.get("message", "")
        self._dynamic_patience = int(decision.get("dynamic_patience", self._fallback_plateau_patience))

        if should_alert and message:
            alert = {
                "type": assessment,
                "epoch": metrics.epoch,
                "severity": severity,
                "message": message,
                "reasoning": decision.get("reasoning", ""),
                "source": "llm" if self._llm else "rule",
            }
            self.alerts.append(alert)
            color = {"critical": "red", "high": "orange3", "medium": "yellow"}.get(severity, "white")
            self.logger.warning(
                f"[{color}]MONITOR [{severity.upper()}][/{color}] {message}"
            )
            self.event_bus.publish(Event(
                type=EventType.MONITOR_ALERT,
                source=self.AGENT_NAME,
                data=alert,
            ))
            if self.shared_context:
                self.shared_context.current_alerts.append(alert)

        if should_stop:
            self.logger.warning(
                f"[red]EARLY STOP[/red] decided at epoch {metrics.epoch}: "
                f"{decision.get('reasoning', '')}"
            )
            self._should_stop = True

    def _build_perception(self, metrics: EpochMetrics) -> str:
        hist_tail = self.history[-10:] if len(self.history) >= 10 else self.history
        history_str = "\n".join(
            f"  epoch={m.epoch}: train_loss={m.train_loss_total:.4f}, "
            f"val_loss={m.val_loss_total:.4f}, mAP50={m.map50:.4f}, lr={m.lr:.2e}, grad={m.grad_norm:.1f}"
            for m in hist_tail
        )
        ctx_block = self.shared_context.to_prompt_block() if self.shared_context else ""

        return "\n".join([
            ctx_block,
            "",
            "## Current Epoch Metrics",
            f"- Epoch: {metrics.epoch}",
            f"- Train loss: {metrics.train_loss_total:.4f} "
            f"(box={metrics.train_box_loss:.4f}, cls={metrics.train_cls_loss:.4f}, dfl={metrics.train_dfl_loss:.4f})",
            f"- Val loss: {metrics.val_loss_total:.4f}",
            f"- mAP50: {metrics.map50:.4f} | mAP50-95: {metrics.map50_95:.4f}",
            f"- Learning rate: {metrics.lr:.2e}",
            f"- Gradient norm: {metrics.grad_norm:.2f}",
            "",
            f"## Training History (last {len(hist_tail)} epochs)",
            history_str or "  (no history yet)",
        ])

    def _fallback_decision(self, metrics: EpochMetrics) -> dict:
        """Rule-based fallback when LLM unavailable."""
        maps = list(self._map_window)
        val_losses = list(self._val_loss_window)
        train_losses = list(self._train_loss_window)

        if metrics.grad_norm > self._fallback_grad_norm_max:
            return {"assessment": "gradient_explosion", "should_alert": True, "should_stop": False,
                    "severity": "critical", "dynamic_patience": 5,
                    "message": f"Gradient norm {metrics.grad_norm:.1f} exceeds threshold",
                    "reasoning": "Rule-based: gradient norm check"}

        if 0 < metrics.lr < self._fallback_lr_collapse:
            return {"assessment": "lr_collapse", "should_alert": True, "should_stop": True,
                    "severity": "critical", "dynamic_patience": 0,
                    "message": f"LR collapsed to {metrics.lr:.2e}",
                    "reasoning": "Rule-based: LR collapse check"}

        n_ov = self._fallback_overfitting_patience
        if len(val_losses) >= n_ov and len(train_losses) >= n_ov:
            if all(val_losses[i+1] > val_losses[i] + 0.005 for i in range(len(val_losses)-1)) \
                    and train_losses[-1] < train_losses[0]:
                return {"assessment": "overfitting", "should_alert": True, "should_stop": False,
                        "severity": "high", "dynamic_patience": 3,
                        "message": "Val loss increasing while train loss decreasing",
                        "reasoning": "Rule-based: overfitting window check"}

        n_pl = self._fallback_plateau_patience
        if len(maps) >= n_pl and (max(maps) - min(maps)) < self._fallback_plateau_threshold:
            return {"assessment": "plateau", "should_alert": True, "should_stop": True,
                    "severity": "medium", "dynamic_patience": 0,
                    "message": f"mAP50 plateaued at {metrics.map50:.4f} for {n_pl} epochs",
                    "reasoning": "Rule-based: plateau window check"}

        return {"assessment": "healthy", "should_alert": False, "should_stop": False,
                "severity": "low", "dynamic_patience": self._fallback_plateau_patience,
                "message": "", "reasoning": "Rule-based: no issues"}

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
        m.grad_norm = self._get_grad_norm(trainer)
        return m

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
        self._should_stop = False
        self.alerts.clear()

    def save_plots(self, run_id: str) -> None:
        if self.history:
            plot_training_curves(self.history, self.experiment_dir / run_id / "plots" / "training_curves.png")

    def run(self, *args, **kwargs):
        pass  # operates via callbacks
