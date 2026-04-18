from __future__ import annotations
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class DatasetProfile:
    """Characteristics of the dataset — injected once, used by all agents for context."""
    num_classes: int = 0
    class_names: list[str] = field(default_factory=list)
    total_images: int = 0
    total_instances: int = 0
    small_obj_ratio: float = 0.0       # fraction of objects < 32px
    class_imbalance_ratio: float = 1.0  # max_count / min_count
    domain: str = "general"            # e.g. "drone", "medical", "satellite"
    notes: str = ""                    # free-text domain hints for LLM

    def to_prompt_str(self) -> str:
        lines = [
            f"- Classes ({self.num_classes}): {', '.join(self.class_names[:10])}{'...' if len(self.class_names) > 10 else ''}",
            f"- Total images: {self.total_images}, instances: {self.total_instances}",
            f"- Small object ratio: {self.small_obj_ratio:.1%}",
            f"- Class imbalance ratio: {self.class_imbalance_ratio:.1f}x",
            f"- Domain: {self.domain}",
        ]
        if self.notes:
            lines.append(f"- Notes: {self.notes}")
        return "\n".join(lines)


@dataclass
class IterationSummary:
    iteration: int
    run_id: str
    map50: float
    map50_95: float
    num_weaknesses: int
    top_weaknesses: list[str]
    config_changes: dict
    alerts: list[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class SharedContext:
    """
    Passed through every agent in the pipeline each iteration.
    Accumulates cross-iteration state, enabling each agent to reason
    with full awareness of what happened before.
    """
    # Static
    dataset: DatasetProfile = field(default_factory=DatasetProfile)
    experiment_dir: str = "experiments"
    max_iterations: int = 10

    # Dynamic — updated each iteration
    iteration: int = 0
    best_map50: float = 0.0
    best_run_id: str = ""
    total_experiments: int = 0
    consecutive_no_improvement: int = 0

    # History
    iteration_history: list[IterationSummary] = field(default_factory=list)
    all_alerts: list[dict] = field(default_factory=list)
    all_weakness_types: list[str] = field(default_factory=list)  # deduplicated across iters

    # Current iteration state (reset each iteration)
    current_run_id: str = ""
    current_config: dict = field(default_factory=dict)
    current_alerts: list[dict] = field(default_factory=list)
    current_weaknesses: list[dict] = field(default_factory=list)
    monitor_decision: dict = field(default_factory=dict)
    analyzer_decision: dict = field(default_factory=dict)
    memory_decision: dict = field(default_factory=dict)
    trainer_decision: dict = field(default_factory=dict)

    def start_iteration(self, iteration: int, run_id: str, config: dict) -> None:
        self.iteration = iteration
        self.current_run_id = run_id
        self.current_config = config
        self.current_alerts = []
        self.current_weaknesses = []
        self.monitor_decision = {}
        self.analyzer_decision = {}
        self.memory_decision = {}
        self.trainer_decision = {}

    def end_iteration(self, map50: float, map50_95: float, config_changes: dict) -> None:
        if map50 > self.best_map50:
            self.best_map50 = map50
            self.best_run_id = self.current_run_id
            self.consecutive_no_improvement = 0
        else:
            self.consecutive_no_improvement += 1

        self.total_experiments += 1

        summary = IterationSummary(
            iteration=self.iteration,
            run_id=self.current_run_id,
            map50=map50,
            map50_95=map50_95,
            num_weaknesses=len(self.current_weaknesses),
            top_weaknesses=[w.get("message", "") for w in self.current_weaknesses[:3]],
            config_changes=config_changes,
            alerts=[a.get("message", "") for a in self.current_alerts],
        )
        self.iteration_history.append(summary)

        for w in self.current_weaknesses:
            wtype = w.get("type", "")
            if wtype and wtype not in self.all_weakness_types:
                self.all_weakness_types.append(wtype)

        self.all_alerts.extend(self.current_alerts)

    def trend_summary(self) -> str:
        """Human-readable trend for LLM prompts."""
        if len(self.iteration_history) < 2:
            return "First iteration — no trend data yet."
        last3 = self.iteration_history[-3:]
        maps = [s.map50 for s in last3]
        trend = "improving" if maps[-1] > maps[0] else "plateauing" if abs(maps[-1] - maps[0]) < 0.005 else "degrading"
        return (
            f"Last {len(last3)} iterations mAP50: {' → '.join(f'{m:.4f}' for m in maps)} ({trend}). "
            f"Best ever: {self.best_map50:.4f} @ {self.best_run_id}. "
            f"Consecutive no-improvement: {self.consecutive_no_improvement}."
        )

    def to_prompt_block(self) -> str:
        return "\n".join([
            "## Shared Context",
            f"- Iteration: {self.iteration}/{self.max_iterations}",
            f"- Best mAP50: {self.best_map50:.4f}",
            f"- Consecutive no-improvement: {self.consecutive_no_improvement}",
            f"- Recurring weakness types: {', '.join(self.all_weakness_types) or 'none yet'}",
            f"- Trend: {self.trend_summary()}",
            "",
            "## Dataset Profile",
            self.dataset.to_prompt_str(),
        ])

    def save(self, path: Optional[Path] = None) -> None:
        out = path or Path(self.experiment_dir) / "shared_context.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "iteration": self.iteration,
            "best_map50": self.best_map50,
            "best_run_id": self.best_run_id,
            "total_experiments": self.total_experiments,
            "consecutive_no_improvement": self.consecutive_no_improvement,
            "all_weakness_types": self.all_weakness_types,
            "iteration_history": [
                {
                    "iteration": s.iteration,
                    "run_id": s.run_id,
                    "map50": s.map50,
                    "map50_95": s.map50_95,
                    "num_weaknesses": s.num_weaknesses,
                    "top_weaknesses": s.top_weaknesses,
                    "config_changes": s.config_changes,
                    "alerts": s.alerts,
                    "timestamp": s.timestamp,
                }
                for s in self.iteration_history
            ],
        }
        with open(out, "w") as f:
            json.dump(data, f, indent=2)
