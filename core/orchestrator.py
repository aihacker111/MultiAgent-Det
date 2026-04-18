from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
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
    Closed-loop 5-agent pipeline. Each iteration:

      [STAGE 1] TrainerAgent (LLM) — evaluates proposals → schedules → spawns training
                MonitorAgent (LLM) — watches via callbacks → may set trainer.stop_training
      [STAGE 2] AnalyzerAgent (LLM) — diagnoses completed run
      [STAGE 3] MemoryAgent  (LLM) — curates recalled memories
      [STAGE 4] PlannerAgent (LLM) — CoT reasoning → config delta proposals
      [STAGE 5] MemoryAgent        — stores outcome for future recall

    NOTE: Stages 2-5 run AFTER training completes (model.train() is blocking).
    The Planner output feeds the NEXT iteration's Trainer.
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
        self.event_bus.subscribe(EventType.MONITOR_ALERT,      self._on_monitor_alert)
        self.event_bus.subscribe(EventType.MONITOR_EARLY_STOP, self._on_early_stop)
        self.event_bus.subscribe(EventType.TRAINING_COMPLETE,  self._on_training_complete)
        self.event_bus.subscribe(EventType.TRAINING_FAILED,    self._on_training_failed)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self, dataset_profile: Optional[DatasetProfile] = None) -> dict:
        if dataset_profile:
            self.shared_context.dataset = dataset_profile

        llm_model = self.config.get("llm", {}).get("model", "N/A")
        console.print(Panel(
            f"[bold cyan]Multi-Agent Object Detection Training System[/bold cyan]\n"
            f"LLM: [yellow]{llm_model}[/yellow] (all 5 agents)\n"
            f"Max iterations: {self.shared_context.max_iterations} | "
            f"Experiment dir: {self.experiment_dir}\n\n"
            f"[dim]Pipeline per iteration:[/dim]\n"
            f"[dim]  TRAIN (blocking) → ANALYZE → MEMORY → PLAN → (next iter TRAIN)[/dim]",
            title="System Start",
            border_style="cyan",
        ))

        current_config = dict(self.config)
        iteration = 0

        while iteration < self.shared_context.max_iterations:
            iteration += 1
            self._early_stop = False

            self.shared_context.start_iteration(
                iteration=iteration,
                run_id=f"iter_{iteration:03d}_pending",
                config=current_config,
            )

            console.print(Rule(
                f"[bold]ITERATION {iteration}/{self.shared_context.max_iterations}[/bold]",
                style="cyan"
            ))

            self.monitor.reset()

            # ─────────────────────────────────────────────────────────────────
            # STAGE 1: TrainerAgent decides which proposals to run, then trains
            # ─────────────────────────────────────────────────────────────────
            proposals_to_run = (
                [ConfigDelta(changes={}, rationale="Initial baseline run", confidence=1.0, priority=1)]
                if iteration == 1
                else self._pending_proposals
            )

            if not proposals_to_run:
                self.logger.warning("No proposals to run — skipping iteration")
                break

            console.print(f"\n[bold cyan]▶ STAGE 1[/bold cyan] — TrainerAgent scheduling + training "
                          f"({len(proposals_to_run)} proposals)")

            run_ids = self.trainer.run(
                proposals=proposals_to_run,
                base_config=current_config,
                monitor_agent=self.monitor,
            )

            if not run_ids:
                self.logger.warning("No runs were spawned (all proposals rejected/duplicate) — stopping")
                break

            last_run_id = run_ids[-1]
            self.shared_context.current_run_id = last_run_id
            stopped_early = self.monitor.should_stop()

            if stopped_early:
                console.print(f"[yellow]  MonitorAgent triggered early stop at this run[/yellow]")

            # ─────────────────────────────────────────────────────────────────
            # STAGE 2: AnalyzerAgent diagnoses the completed run
            # ─────────────────────────────────────────────────────────────────
            console.print(f"\n[bold cyan]▶ STAGE 2[/bold cyan] — AnalyzerAgent diagnosing [bold]{last_run_id}[/bold]")

            results_csv = self.experiment_dir / last_run_id / "train" / "results.csv"
            report = self.analyzer.run(
                run_id=last_run_id,
                results_path=results_csv if results_csv.exists() else None,
            )

            # ─────────────────────────────────────────────────────────────────
            # STAGE 3: MemoryAgent recalls + LLM-curates relevant past fixes
            # ─────────────────────────────────────────────────────────────────
            console.print(f"\n[bold cyan]▶ STAGE 3[/bold cyan] — MemoryAgent recalling past experiences")

            memory_result = self.memory.run(weaknesses=report.weaknesses)
            console.print(
                f"  Recalled {memory_result.get('raw_count', 0)} raw → "
                f"{len(memory_result.get('curated_memories', []))} curated"
            )

            # ─────────────────────────────────────────────────────────────────
            # STAGE 4: PlannerAgent reasons via LLM CoT → config delta proposals
            # ─────────────────────────────────────────────────────────────────
            console.print(f"\n[bold cyan]▶ STAGE 4[/bold cyan] — PlannerAgent reasoning (LLM CoT)")

            proposals = self.planner.run(
                report=report,
                current_config=current_config,
                memory_recalls=memory_result.get("curated_memories", []),
                memory_synthesis=memory_result.get("synthesis", ""),
                run_id=last_run_id,
            )

            if proposals:
                console.print(f"  [green]Generated {len(proposals)} proposal(s):[/green]")
                for i, p in enumerate(proposals, 1):
                    console.print(
                        f"    [{i}] {p.rationale}\n"
                        f"        changes={p.changes} | "
                        f"conf={p.confidence:.2f} | Δ={p.expected_map_improvement:+.3f}"
                    )
                # Apply top proposal to config for next training run
                current_config = proposals[0].apply_to(current_config)
            else:
                console.print("  [yellow]No proposals generated — model may have converged[/yellow]")

            self._pending_proposals = proposals

            # ─────────────────────────────────────────────────────────────────
            # STAGE 5: MemoryAgent stores outcome for future iterations
            # ─────────────────────────────────────────────────────────────────
            console.print(f"\n[bold cyan]▶ STAGE 5[/bold cyan] — MemoryAgent storing outcome")

            if memory_result.get("should_store_current", True):
                for proposal in proposals:
                    self.memory.store(
                        weaknesses=report.weaknesses,
                        config_delta=proposal,
                        actual_map_improvement=proposal.expected_map_improvement,
                        run_id=last_run_id,
                    )

            # Update shared context
            self.shared_context.end_iteration(
                map50=report.overall_map50,
                map50_95=report.overall_map50_95,
                config_changes=proposals[0].changes if proposals else {},
            )
            self.shared_context.save(self.experiment_dir / "shared_context.json")

            self._print_iteration_summary(report, proposals)

            # Stop conditions
            if self._early_stop and not proposals:
                self.logger.info("Early stop + no proposals — terminating")
                break

            if not proposals and iteration > 1:
                self.logger.info("No proposals — model converged, terminating")
                break

        return self._finalize()

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_monitor_alert(self, event: Event) -> None:
        alert = event.data
        self.logger.warning(f"[MonitorAgent] {alert.get('message', '')}")

    def _on_early_stop(self, event: Event) -> None:
        self._early_stop = True
        self.logger.warning("[MonitorAgent] Early stop flag set — training will stop after current epoch")

    def _on_training_complete(self, event: Event) -> None:
        d = event.data
        self.logger.info(
            f"[TrainerAgent] Training complete: {d.get('run_id')} | "
            f"mAP50={d.get('metrics', {}).get('map50', 0):.4f}"
        )

    def _on_training_failed(self, event: Event) -> None:
        self.logger.error(
            f"[TrainerAgent] Training failed: {event.data.get('run_id')} — {event.data.get('error', '')}"
        )

    # ── Display ───────────────────────────────────────────────────────────────

    def _print_iteration_summary(self, report: AnalysisReport, proposals: list[ConfigDelta]) -> None:
        ctx = self.shared_context
        table = Table(title=f"Iteration {ctx.iteration} Complete", show_header=False, box=None)
        table.add_column("", style="cyan", width=25)
        table.add_column("", style="white")
        table.add_row("Run ID", report.run_id)
        table.add_row("mAP50", f"{report.overall_map50:.4f}")
        table.add_row("mAP50-95", f"{report.overall_map50_95:.4f}")
        table.add_row("Best mAP50 ever", f"{ctx.best_map50:.4f} ({ctx.best_run_id})")
        table.add_row("Trend", ctx.trend_summary()[:80])
        table.add_row("Weaknesses found", str(len(report.weaknesses)))
        table.add_row("Proposals for next iter", str(len(proposals)))
        console.print(table)

        if report.weaknesses:
            console.print("\n[bold yellow]Top Weaknesses:[/bold yellow]")
            for w in report.weaknesses[:3]:
                sev = w.get("severity", "?").upper()
                col = {"CRITICAL": "red", "HIGH": "orange3", "MEDIUM": "yellow"}.get(sev, "white")
                console.print(
                    f"  [{col}][{sev}][/{col}] {w.get('message', '')} "
                    f"| root: {w.get('root_cause', '')} | fix: {w.get('fix_category', '')}"
                )

    def _finalize(self) -> dict:
        ctx = self.shared_context
        summary = {
            "total_iterations": ctx.iteration,
            "best_map50": ctx.best_map50,
            "best_run_id": ctx.best_run_id,
            "iteration_history": [
                {"iteration": s.iteration, "map50": s.map50, "run_id": s.run_id,
                 "config_changes": s.config_changes}
                for s in ctx.iteration_history
            ],
            "registry_summary": self.trainer.get_summary(),
            "memory_stats": self.memory.get_stats(),
        }
        out = self.experiment_dir / "final_summary.json"
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)

        console.print(Panel(
            f"[bold green]Pipeline Complete[/bold green]\n"
            f"Iterations: {ctx.iteration} | Best mAP50: [green]{ctx.best_map50:.4f}[/green] "
            f"@ {ctx.best_run_id}\n"
            f"Total experiments: {summary['registry_summary'].get('total', 0)} | "
            f"Memory entries: {summary['memory_stats'].get('total', 0)}",
            border_style="green",
        ))
        return summary
