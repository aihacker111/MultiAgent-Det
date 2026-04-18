from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agents.monitor_agent import MonitorAgent
from agents.analyzer_agent import AnalyzerAgent
from agents.planner_agent import PlannerAgent
from agents.trainer_agent import TrainerAgent
from agents.memory_agent import MemoryAgent
from core.event_bus import EventBus, EventType, Event
from core.shared_context import SharedContext, DatasetProfile
from utils.logger import get_logger
from utils.metrics import AnalysisReport, ConfigDelta

console = Console()


class Orchestrator:
    """
    Coordinates all 5 LLM-based agents through a closed feedback loop.
    SharedContext flows through every agent — each agent reasons with full
    awareness of the dataset, history, and what other agents decided.

    Loop per iteration:
    1. TrainerAgent (LLM) evaluates proposals -> schedules runs -> spawns training
       MonitorAgent (LLM) watches training in real-time via callbacks
    2. AnalyzerAgent (LLM) decides what to analyze -> interprets patterns
    3. MemoryAgent (LLM) curates recalled memories -> synthesizes for Planner
    4. PlannerAgent (LLM CoT) reasons with analysis + memory -> proposes config deltas
    5. MemoryAgent stores outcome
    6. SharedContext updated -> next iteration
    """

    def __init__(
        self,
        config: dict,
        experiment_dir: Path,
        dataset_profile: Optional[DatasetProfile] = None,
    ) -> None:
        self.config = config
        self.experiment_dir = experiment_dir
        experiment_dir.mkdir(parents=True, exist_ok=True)
        (experiment_dir / "logs").mkdir(exist_ok=True)

        self.logger = get_logger(
            "Orchestrator",
            log_file=experiment_dir / "logs" / "orchestrator.log",
            level=config.get("system", {}).get("log_level", "INFO"),
        )

        # Build shared context
        self.shared_context = SharedContext(
            dataset=dataset_profile or DatasetProfile(),
            experiment_dir=str(experiment_dir),
            max_iterations=config.get("system", {}).get("max_iterations", 10),
        )

        self.event_bus = EventBus()
        self._setup_agents()
        self._register_event_handlers()

        self._pending_proposals: list[ConfigDelta] = []
        self._early_stop: bool = False

    def _setup_agents(self) -> None:
        kwargs = dict(
            config=self.config,
            event_bus=self.event_bus,
            experiment_dir=self.experiment_dir,
            shared_context=self.shared_context,
        )
        self.monitor  = MonitorAgent(**kwargs)
        self.analyzer = AnalyzerAgent(**kwargs)
        self.memory   = MemoryAgent(**kwargs)
        self.planner  = PlannerAgent(**kwargs)
        self.trainer  = TrainerAgent(**kwargs)
        self.logger.info("All 5 LLM agents initialized with shared context")

    def _register_event_handlers(self) -> None:
        self.event_bus.subscribe(EventType.MONITOR_ALERT, self._on_monitor_alert)
        self.event_bus.subscribe(EventType.MONITOR_EARLY_STOP, self._on_early_stop)
        self.event_bus.subscribe(EventType.TRAINING_COMPLETE, self._on_training_complete)
        self.event_bus.subscribe(EventType.TRAINING_FAILED, self._on_training_failed)

    def run(self, dataset_profile: Optional[DatasetProfile] = None) -> dict:
        if dataset_profile:
            self.shared_context.dataset = dataset_profile

        console.print(Panel(
            f"[bold cyan]Multi-Agent Object Detection System[/bold cyan]\n"
            f"All 5 agents: LLM-backed via OpenRouter | model: {self.config.get('llm', {}).get('model', 'N/A')}\n"
            f"Max iterations: {self.shared_context.max_iterations} | "
            f"Experiment dir: {self.experiment_dir}",
            title="System Start",
            border_style="cyan",
        ))

        current_config = dict(self.config)
        iteration = 0

        while iteration < self.shared_context.max_iterations:
            iteration += 1
            self.shared_context.start_iteration(
                iteration=iteration,
                run_id=f"iter_{iteration:03d}_pending",
                config=current_config,
            )

            self.logger.info(f"\n{'='*60}\nITERATION {iteration}/{self.shared_context.max_iterations}\n{'='*60}")
            self.monitor.reset()

            # ── 1. Trainer: evaluate proposals + spawn training ───────────────
            proposals_to_run = (
                [ConfigDelta(changes={}, rationale="Initial baseline run", confidence=1.0, priority=1)]
                if iteration == 1
                else self._pending_proposals
            )

            run_ids = self.trainer.run(
                proposals=proposals_to_run,
                base_config=current_config,
                monitor_agent=self.monitor,
            )

            if not run_ids:
                self.logger.warning("No runs spawned — terminating")
                break

            last_run_id = run_ids[-1]
            self.shared_context.current_run_id = last_run_id

            # ── 2. Analyzer: diagnose the completed run ───────────────────────
            report = self.analyzer.run(
                run_id=last_run_id,
                results_path=self.experiment_dir / last_run_id / "train" / "results.csv",
            )

            # ── 3. Memory: recall + LLM-curated synthesis ─────────────────────
            memory_result = self.memory.run(weaknesses=report.weaknesses)

            # ── 4. Planner: CoT reasoning -> config delta proposals ───────────
            proposals = self.planner.run(
                report=report,
                current_config=current_config,
                memory_recalls=memory_result.get("curated_memories", []),
                memory_synthesis=memory_result.get("synthesis", ""),
                run_id=last_run_id,
            )
            self._pending_proposals = proposals

            # ── 5. Memory: store outcome ──────────────────────────────────────
            for proposal in proposals:
                if memory_result.get("should_store_current", True):
                    self.memory.store(
                        weaknesses=report.weaknesses,
                        config_delta=proposal,
                        actual_map_improvement=proposal.expected_map_improvement,
                        run_id=last_run_id,
                    )

            # ── 6. Update shared context + config ────────────────────────────
            if proposals:
                changes = proposals[0].changes
                current_config = proposals[0].apply_to(current_config)
            else:
                changes = {}

            self.shared_context.end_iteration(
                map50=report.overall_map50,
                map50_95=report.overall_map50_95,
                config_changes=changes,
            )
            self.shared_context.save(self.experiment_dir / "shared_context.json")

            self._print_iteration_summary(report, proposals, memory_result)

            if self._early_stop:
                self.logger.info("[yellow]Early stop triggered — exiting loop[/yellow]")
                break

            if not proposals and iteration > 1:
                self.logger.info("No proposals generated — converged")
                break

        return self._finalize()

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_monitor_alert(self, event: Event) -> None:
        self.logger.warning(f"Monitor: {event.data.get('message', '')}")

    def _on_early_stop(self, event: Event) -> None:
        self.logger.warning("Early stop requested by MonitorAgent")
        self._early_stop = True

    def _on_training_complete(self, event: Event) -> None:
        data = event.data
        self.logger.info(
            f"Training complete: {data.get('run_id')} | "
            f"mAP50={data.get('metrics', {}).get('map50', 0):.4f}"
        )

    def _on_training_failed(self, event: Event) -> None:
        self.logger.error(f"Training failed: {event.data.get('run_id')} — {event.data.get('error')}")

    # ── Display ───────────────────────────────────────────────────────────────

    def _print_iteration_summary(
        self,
        report: AnalysisReport,
        proposals: list[ConfigDelta],
        memory_result: dict,
    ) -> None:
        ctx = self.shared_context
        table = Table(title=f"Iteration {ctx.iteration} Summary", show_header=True)
        table.add_column("", style="cyan")
        table.add_column("", style="white")

        table.add_row("Run ID", report.run_id)
        table.add_row("mAP50", f"{report.overall_map50:.4f}")
        table.add_row("Best mAP50", f"{ctx.best_map50:.4f}")
        table.add_row("Trend", ctx.trend_summary()[:70])
        table.add_row("Weaknesses", str(len(report.weaknesses)))
        table.add_row("Memory recalled", str(memory_result.get("raw_count", 0)))
        table.add_row("Proposals", str(len(proposals)))
        console.print(table)

        if report.weaknesses:
            console.print("[bold yellow]Top Weaknesses:[/bold yellow]")
            for w in report.weaknesses[:3]:
                sev = w.get("severity", "?").upper()
                col = {"CRITICAL": "red", "HIGH": "orange3", "MEDIUM": "yellow"}.get(sev, "white")
                console.print(f"  [{col}][{sev}][/{col}] {w.get('message', '')} | fix: {w.get('fix_category', '')}")

        if memory_result.get("synthesis"):
            console.print(f"[bold blue]Memory synthesis:[/bold blue] {memory_result['synthesis'][:120]}")

        if proposals:
            console.print("[bold green]Next Proposals:[/bold green]")
            for p in proposals[:2]:
                console.print(f"  • {p.rationale} (Δ={p.expected_map_improvement:+.3f}, conf={p.confidence:.2f})")

    def _finalize(self) -> dict:
        ctx = self.shared_context
        summary = {
            "total_iterations": ctx.iteration,
            "best_map50": ctx.best_map50,
            "best_run_id": ctx.best_run_id,
            "iteration_history": [
                {"iteration": s.iteration, "map50": s.map50, "run_id": s.run_id}
                for s in ctx.iteration_history
            ],
            "registry_summary": self.trainer.get_summary(),
            "memory_stats": self.memory.get_stats(),
        }

        out = self.experiment_dir / "final_summary.json"
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)

        console.print(Panel(
            f"[bold green]Done[/bold green] | "
            f"Iterations: {ctx.iteration} | Best mAP50: {ctx.best_map50:.4f}",
            border_style="green",
        ))
        return summary
