#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
from pathlib import Path

import click
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()
console = Console()

sys.path.insert(0, str(Path(__file__).parent))


def load_config(config_path: Path, overrides: dict) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    for key, value in overrides.items():
        parts = key.split(".")
        node = config
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return config


@click.group()
def cli():
    """Multi-Agent Object Detection Training System"""
    pass


@cli.command()
@click.option("--config", "-c", default="config/default_config.yaml", type=click.Path(exists=True), help="Config file path")
@click.option("--data", "-d", required=True, type=click.Path(exists=True), help="Path to data.yaml")
@click.option("--model", "-m", default="yolov8n.pt", help="Base YOLO model")
@click.option("--epochs", "-e", default=100, type=int, help="Epochs per training run")
@click.option("--iterations", "-i", default=5, type=int, help="Max agent loop iterations")
@click.option("--imgsz", default=640, type=int, help="Image size")
@click.option("--batch", default=16, type=int, help="Batch size")
@click.option("--gpu-memory", default=16.0, type=float, help="Available GPU memory in GB")
@click.option("--experiment-dir", default="experiments", type=click.Path(), help="Experiment output directory")
@click.option("--no-llm", is_flag=True, help="Disable LLM Planner, use rule-based fallback")
@click.option("--demo", is_flag=True, help="Run in demo mode (no real training, mock results)")
def train(config, data, model, epochs, iterations, imgsz, batch, gpu_memory, experiment_dir, no_llm, demo):
    """Start the multi-agent training loop."""
    from core.orchestrator import Orchestrator

    cfg = load_config(Path(config), {
        "training.data": data,
        "training.model": model,
        "training.epochs": epochs,
        "training.imgsz": imgsz,
        "training.batch": batch,
        "system.max_iterations": iterations,
        "system.gpu_memory_gb": gpu_memory,
    })

    if no_llm:
        os.environ.pop("OPENROUTER_API_KEY", None)
        console.print("[yellow]LLM disabled — using rule-based planner[/yellow]")

    if demo:
        cfg["_demo_mode"] = True
        console.print("[cyan]Running in demo mode — no real training[/cyan]")

    exp_dir = Path(experiment_dir)
    console.print(Panel(
        f"[bold]Configuration[/bold]\n"
        f"Model: {model} | Data: {data}\n"
        f"Epochs: {epochs} | Iterations: {iterations}\n"
        f"Image size: {imgsz} | Batch: {batch}\n"
        f"GPU memory: {gpu_memory}GB\n"
        f"Experiments: {exp_dir.absolute()}",
        border_style="blue",
    ))

    orchestrator = Orchestrator(cfg, exp_dir)
    result = orchestrator.run()

    console.print(f"\n[bold green]Best mAP50 achieved: {result['best_map50']:.4f}[/bold green]")
    console.print(f"Completed {result['total_iterations']} iterations")


@cli.command()
@click.option("--data", "-d", required=True, type=click.Path(exists=True), help="Path to data.yaml")
@click.option("--results-dir", required=True, type=click.Path(exists=True), help="Path to existing Ultralytics run results")
@click.option("--config", "-c", default="config/default_config.yaml", type=click.Path(exists=True))
@click.option("--experiment-dir", default="experiments", type=click.Path())
def analyze(data, results_dir, config, experiment_dir):
    """Run Analyzer Agent on an existing training run."""
    from core.event_bus import EventBus
    from agents.analyzer_agent import AnalyzerAgent

    cfg = load_config(Path(config), {"training.data": data})
    bus = EventBus()
    exp_dir = Path(experiment_dir)

    agent = AnalyzerAgent(cfg, bus, exp_dir)
    run_id = Path(results_dir).name
    results_path = Path(results_dir) / "results.csv"

    report = agent.run(run_id=run_id, results_path=results_path)

    console.print(f"\n[bold]Analysis Report for[/bold] {run_id}")
    console.print(f"mAP50: {report.overall_map50:.4f} | mAP50-95: {report.overall_map50_95:.4f}")
    console.print(f"Weaknesses found: {len(report.weaknesses)}")

    for w in report.weaknesses[:5]:
        sev = w.get("severity", "?").upper()
        color = {"CRITICAL": "red", "HIGH": "orange3", "MEDIUM": "yellow"}.get(sev, "white")
        console.print(f"  [{color}][{sev}][/{color}] {w.get('message')}")


@cli.command()
@click.option("--weakness-desc", "-w", required=True, help="Describe the weakness (free text)")
@click.option("--config", "-c", default="config/default_config.yaml", type=click.Path(exists=True))
@click.option("--experiment-dir", default="experiments", type=click.Path())
@click.option("--current-map", default=0.0, type=float, help="Current mAP50")
def plan(weakness_desc, config, experiment_dir, current_map):
    """Run Planner Agent for a given weakness description."""
    import uuid
    from core.event_bus import EventBus
    from agents.planner_agent import PlannerAgent
    from agents.memory_agent import MemoryAgent
    from utils.metrics import AnalysisReport

    cfg = load_config(Path(config), {})
    bus = EventBus()
    exp_dir = Path(experiment_dir)

    memory = MemoryAgent(cfg, bus, exp_dir)
    planner = PlannerAgent(cfg, bus, exp_dir)

    weakness = [{"type": "user_defined", "severity": "high", "message": weakness_desc}]
    recalls = memory.recall(weakness)

    report = AnalysisReport(
        run_id=f"plan_{uuid.uuid4().hex[:6]}",
        epoch=0,
        overall_map50=current_map,
        weaknesses=weakness,
    )

    proposals = planner.run(
        report=report,
        current_config=cfg,
        memory_recalls=recalls,
        run_id=report.run_id,
    )

    console.print(f"\n[bold cyan]Generated {len(proposals)} proposals:[/bold cyan]")
    for i, p in enumerate(proposals, 1):
        console.print(f"\n[bold]Proposal {i}:[/bold] {p.rationale}")
        console.print(f"  Config changes: {p.changes}")
        console.print(f"  Expected ΔmAP50: {p.expected_map_improvement:+.3f}")
        console.print(f"  Confidence: {p.confidence:.2f}")


@cli.command()
@click.option("--experiment-dir", default="experiments", type=click.Path(exists=True))
def status(experiment_dir):
    """Show experiment registry status."""
    from core.experiment_registry import ExperimentRegistry

    exp_dir = Path(experiment_dir)
    db_path = exp_dir / "registry.db"
    if not db_path.exists():
        console.print("[yellow]No experiment registry found[/yellow]")
        return

    registry = ExperimentRegistry(db_path)
    summary = registry.summary()

    table = Table(title="Experiment Registry")
    table.add_column("Status")
    table.add_column("Count", justify="right")

    for status_name, count in summary.get("by_status", {}).items():
        table.add_row(status_name, str(count))

    console.print(table)
    console.print(f"Total experiments: {summary['total']}")
    console.print(f"Best mAP50: {summary['best_map50']:.4f}")

    best_runs = registry.get_best(top_n=5)
    if best_runs:
        console.print("\n[bold]Top 5 Runs:[/bold]")
        for run in best_runs:
            console.print(f"  {run.run_id} | mAP50={run.final_map50:.4f} | {run.status}")


@cli.command()
@click.option("--experiment-dir", default="experiments", type=click.Path(exists=True))
def memory_stats(experiment_dir):
    """Show Memory Agent statistics."""
    from core.event_bus import EventBus
    from agents.memory_agent import MemoryAgent
    import yaml

    cfg_path = Path("config/default_config.yaml")
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}

    bus = EventBus()
    exp_dir = Path(experiment_dir)
    memory = MemoryAgent(cfg, bus, exp_dir)
    stats = memory.get_stats()

    console.print("[bold]Memory Agent Statistics[/bold]")
    for k, v in stats.items():
        console.print(f"  {k}: {v}")


if __name__ == "__main__":
    cli()
