from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from agents.base_agent import BaseAgent
from core.event_bus import EventBus, Event, EventType
from core.shared_context import SharedContext
from utils.metrics import AnalysisReport, ClassMetrics, SizeMetrics, AnchorMetrics
from utils.visualization import plot_confusion_matrix, plot_per_class_ap, plot_size_breakdown

# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a computer vision diagnostic expert agent. Your role is to analyze YOLO training results and identify what is actually wrong — not just report numbers.

You receive raw metrics from a completed training run. You must:
1. Decide WHICH analyses are actually needed for this specific situation (not always run all)
2. Interpret patterns holistically — consider dataset profile and training history
3. Build a prioritized weakness list with root cause diagnosis, not just symptoms
4. Assign accurate severity accounting for dataset size and domain

Key diagnostic rules:
- Low AP on a class with few instances is expected — distinguish data scarcity from model failure
- FP type matters: background FP -> augmentation fix; class confusion FP -> data or loss fix
- Small object weakness with low anchor assignment ratio -> imgsz fix, not augmentation
- Class imbalance > 10x -> resampling strategy, not hyperparameter tuning
- Recurring weaknesses across iterations mean previous fixes were insufficient — escalate severity

Be concise and actionable. Prioritize root causes over symptoms."""

OUTPUT_SCHEMA = """{
  "run_needed_analyses": ["per_class_ap", "confusion_matrix", "size_breakdown", "anchor_analysis", "fp_fn_breakdown"],
  "skip_reasons": {"analysis_name": "reason to skip"},
  "weakness_interpretations": [
    {
      "type": "weak_class | class_confusion | small_object_weakness | anchor_failure | data_scarcity | imbalance",
      "severity": "critical | high | medium | low",
      "root_cause": "one sentence diagnosis",
      "affected": "class name or 'all' or 'small objects'",
      "fix_category": "augmentation | regularization | imgsz | resampling | data_collection | loss_weight",
      "message": "human-readable description",
      "is_recurring": true | false
    }
  ],
  "overall_assessment": "brief summary of the model state",
  "suggested_priority_focus": "what the Planner should focus on most"
}"""
# ─────────────────────────────────────────────────────────────────────────────


class AnalyzerAgent(BaseAgent):
    """
    Post-training diagnostic agent.
    LLM decides which analyses to run, interprets patterns holistically,
    and produces a root-cause weakness list — not just metric summaries.
    """

    AGENT_NAME = "AnalyzerAgent"
    SYSTEM_PROMPT = SYSTEM_PROMPT

    def _setup(self) -> None:
        cfg = self.config.get("analyzer", {})
        self.weak_ap_threshold: float = cfg.get("weak_class_ap_threshold", 0.3)
        self.small_size: int = cfg.get("small_obj_size", 32)
        self.medium_size: int = cfg.get("medium_obj_size", 96)
        self.anchor_min_ratio: float = cfg.get("anchor_assignment_min_ratio", 0.5)

    def run(
        self,
        run_id: str,
        trainer=None,
        validator=None,
        results_path: Path | None = None,
    ) -> AnalysisReport:
        self.logger.info(f"[cyan]Analyzing[/cyan] [bold]{run_id}[/bold]")

        # Step 1: collect raw metrics (deterministic tools)
        report = self._collect_raw_metrics(run_id, trainer, results_path)

        # Step 2: perceive — build context string for LLM
        perception = self._build_perception(report)

        # Step 3: reason — LLM decides what to analyze and how to interpret
        decision = self.reason(perception, OUTPUT_SCHEMA)

        if not decision:
            decision = self._fallback_decision(report)

        # Step 4: act — apply LLM interpretation to finalize report
        report.weaknesses = self._apply_decision(report, decision)

        self._save_report(report, decision)
        self._generate_plots(report, decision.get("run_needed_analyses", []))

        self.event_bus.publish(Event(
            type=EventType.ANALYSIS_COMPLETE,
            source=self.AGENT_NAME,
            data=report,
        ))

        if self.shared_context:
            self.shared_context.current_weaknesses = report.weaknesses

        self.logger.info(
            f"Analysis done — {len(report.weaknesses)} weaknesses | "
            f"mAP50={report.overall_map50:.4f} | "
            f"focus: {decision.get('suggested_priority_focus', 'N/A')}"
        )
        return report

    # ── Perception builder ────────────────────────────────────────────────────

    def _build_perception(self, report: AnalysisReport) -> str:
        ctx_block = self.shared_context.to_prompt_block() if self.shared_context else ""

        lines = [
            ctx_block,
            "",
            "## Raw Metrics — Current Run",
            f"- mAP50: {report.overall_map50:.4f} | mAP50-95: {report.overall_map50_95:.4f}",
        ]

        if report.class_metrics:
            lines += ["", "## Per-class AP50 (sorted by AP, worst first)"]
            for m in sorted(report.class_metrics, key=lambda x: x.ap50)[:10]:
                lines.append(
                    f"  {m.class_name}: AP50={m.ap50:.3f}, P={m.precision:.2f}, "
                    f"R={m.recall:.2f}, n={m.num_instances}, "
                    f"FP_bg={m.fp_background}, FP_conf={m.fp_class_confusion}, FN={m.fn_missed}"
                )

        sm = report.size_metrics
        if sm.small_count + sm.medium_count + sm.large_count > 0:
            lines += [
                "",
                "## Object Size Performance",
                f"  Small (<{self.small_size}px):  mAP50={sm.small_map50:.3f} (n={sm.small_count})",
                f"  Medium:                        mAP50={sm.medium_map50:.3f} (n={sm.medium_count})",
                f"  Large:                         mAP50={sm.large_map50:.3f} (n={sm.large_count})",
            ]

        am = report.anchor_metrics
        if am.total_objects > 0:
            lines += [
                "",
                "## Anchor Assignment",
                f"  Assignment ratio: {am.assignment_ratio:.1%} ({am.unassigned_objects} unassigned / {am.total_objects} total)",
            ]

        if report.confusion_matrix is not None and report.class_names:
            nc = len(report.class_names)
            matrix = report.confusion_matrix
            confusions = []
            for i in range(min(nc, matrix.shape[0])):
                for j in range(min(nc, matrix.shape[1])):
                    if i != j and matrix[i, j] > 0.15:
                        confusions.append(
                            f"  '{report.class_names[i]}' -> '{report.class_names[j]}': {matrix[i,j]:.1%}"
                        )
            if confusions:
                lines += ["", "## Notable Confusions"] + confusions

        if self.shared_context and self.shared_context.all_weakness_types:
            lines += [
                "",
                f"## Recurring weakness types across iterations: "
                f"{', '.join(self.shared_context.all_weakness_types)}",
            ]

        return "\n".join(lines)

    # ── Decision application ──────────────────────────────────────────────────

    def _apply_decision(self, report: AnalysisReport, decision: dict) -> list[dict]:
        interpretations = decision.get("weakness_interpretations", [])
        if not interpretations:
            return self._rule_based_weaknesses(report)

        weaknesses = []
        for i, interp in enumerate(interpretations):
            w = {
                "type": interp.get("type", "unknown"),
                "severity": interp.get("severity", "medium"),
                "root_cause": interp.get("root_cause", ""),
                "affected": interp.get("affected", ""),
                "fix_category": interp.get("fix_category", ""),
                "message": interp.get("message", ""),
                "is_recurring": interp.get("is_recurring", False),
                "priority": i + 1,
                "source": "llm",
            }
            weaknesses.append(w)

        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        weaknesses.sort(key=lambda w: severity_order.get(w["severity"], 3))
        for i, w in enumerate(weaknesses):
            w["priority"] = i + 1

        return weaknesses

    def _fallback_decision(self, report: AnalysisReport) -> dict:
        weaknesses = self._rule_based_weaknesses(report)
        return {
            "run_needed_analyses": ["per_class_ap", "confusion_matrix", "size_breakdown", "anchor_analysis"],
            "weakness_interpretations": weaknesses,
            "overall_assessment": f"mAP50={report.overall_map50:.4f}",
            "suggested_priority_focus": weaknesses[0]["type"] if weaknesses else "general_improvement",
        }

    def _rule_based_weaknesses(self, report: AnalysisReport) -> list[dict]:
        weaknesses = []
        for cm in report.class_metrics:
            if cm.ap50 < self.weak_ap_threshold:
                is_recurring = (
                    self.shared_context and
                    "weak_class" in self.shared_context.all_weakness_types
                )
                weaknesses.append({
                    "type": "weak_class",
                    "severity": "critical" if cm.ap50 < 0.15 else "high",
                    "root_cause": f"Class '{cm.class_name}' AP50={cm.ap50:.3f}",
                    "affected": cm.class_name,
                    "fix_category": "augmentation" if cm.fp_background > cm.fp_class_confusion else "data_collection",
                    "message": f"Class '{cm.class_name}' AP50={cm.ap50:.3f} below threshold",
                    "is_recurring": is_recurring,
                    "source": "rule",
                })

        sm = report.size_metrics
        if sm.small_count > 0 and sm.small_map50 < self.weak_ap_threshold:
            weaknesses.append({
                "type": "small_object_weakness",
                "severity": "high",
                "root_cause": "Small objects underperforming — likely imgsz too small",
                "affected": "small objects",
                "fix_category": "imgsz",
                "message": f"Small objects mAP50={sm.small_map50:.3f} (n={sm.small_count})",
                "is_recurring": False,
                "source": "rule",
            })

        am = report.anchor_metrics
        if am.assignment_ratio < self.anchor_min_ratio:
            weaknesses.append({
                "type": "anchor_failure",
                "severity": "critical",
                "root_cause": f"Only {am.assignment_ratio:.1%} objects assigned — anchors mismatched",
                "affected": "all",
                "fix_category": "imgsz",
                "message": f"Anchor assignment ratio {am.assignment_ratio:.1%}",
                "is_recurring": False,
                "source": "rule",
            })

        if report.confusion_matrix is not None and report.class_names:
            nc = len(report.class_names)
            matrix = report.confusion_matrix
            for i in range(min(nc, matrix.shape[0])):
                for j in range(min(nc, matrix.shape[1])):
                    if i != j and matrix[i, j] > 0.2:
                        weaknesses.append({
                            "type": "class_confusion",
                            "severity": "medium",
                            "root_cause": f"'{report.class_names[i]}' confused as '{report.class_names[j]}'",
                            "affected": report.class_names[i],
                            "fix_category": "augmentation",
                            "message": f"Confusion {report.class_names[i]}->{report.class_names[j]}: {matrix[i,j]:.1%}",
                            "is_recurring": False,
                            "source": "rule",
                        })

        sev_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        weaknesses.sort(key=lambda w: sev_order.get(w.get("severity", "low"), 3))
        for i, w in enumerate(weaknesses):
            w["priority"] = i + 1
        return weaknesses

    # ── Raw metric collection (deterministic) ─────────────────────────────────

    def _collect_raw_metrics(
        self, run_id: str, trainer, results_path: Path | None
    ) -> AnalysisReport:
        if trainer is not None:
            return self._from_trainer(run_id, trainer)
        if results_path is not None and results_path.exists():
            return self._from_results_file(run_id, results_path)
        return self._mock_report(run_id)

    def _from_trainer(self, run_id: str, trainer) -> AnalysisReport:
        metrics_dict = getattr(trainer, "metrics", {}) or {}
        epoch = getattr(trainer, "epoch", 0)
        report = AnalysisReport(
            run_id=run_id, epoch=epoch,
            overall_map50=float(metrics_dict.get("metrics/mAP50(B)", 0.0)),
            overall_map50_95=float(metrics_dict.get("metrics/mAP50-95(B)", 0.0)),
        )
        validator = getattr(trainer, "validator", None)
        if validator:
            report.class_names = self._get_class_names(validator)
            report.class_metrics = self._extract_class_metrics(validator, report.class_names)
            report.confusion_matrix = self._extract_confusion_matrix(validator)
            report.size_metrics = self._extract_size_metrics(validator)
            report.anchor_metrics = self._analyze_anchor_assignment(validator)
        return report

    def _from_results_file(self, run_id: str, results_path: Path) -> AnalysisReport:
        import csv
        report = AnalysisReport(run_id=run_id, epoch=0)
        with open(results_path) as f:
            rows = list(csv.DictReader(f))
        if rows:
            last = rows[-1]
            report.epoch = len(rows)
            report.overall_map50 = float(last.get("metrics/mAP50(B)", 0))
            report.overall_map50_95 = float(last.get("metrics/mAP50-95(B)", 0))
        return report

    def _get_class_names(self, validator) -> list[str]:
        names = getattr(validator, "names", None) or getattr(
            getattr(validator, "model", None), "names", None
        )
        if isinstance(names, dict):
            return [names[i] for i in sorted(names)]
        return list(names) if names else []

    def _extract_class_metrics(self, validator, class_names) -> list[ClassMetrics]:
        result = []
        try:
            stats = getattr(validator, "stats", None) or getattr(validator, "metrics", None)
            if stats is None:
                return result
            nc = len(class_names)
            ap50 = np.array(getattr(stats, "ap50", np.zeros(nc)))
            ap = np.array(getattr(stats, "ap", np.zeros(nc)))
            p = np.array(getattr(stats, "p", np.zeros(nc)))
            r = np.array(getattr(stats, "r", np.zeros(nc)))
            for i, name in enumerate(class_names):
                result.append(ClassMetrics(
                    class_name=name,
                    ap50=float(ap50[i]) if i < len(ap50) else 0.0,
                    ap50_95=float(ap[i]) if i < len(ap) else 0.0,
                    precision=float(p[i]) if i < len(p) else 0.0,
                    recall=float(r[i]) if i < len(r) else 0.0,
                ))
        except Exception as exc:
            self.logger.warning(f"Class metrics extraction: {exc}")
        return result

    def _extract_confusion_matrix(self, validator) -> np.ndarray | None:
        try:
            cm_obj = getattr(validator, "confusion_matrix", None)
            matrix = getattr(cm_obj, "matrix", None) if cm_obj else None
            return np.array(matrix, dtype=float) if matrix is not None else None
        except Exception:
            return None

    def _extract_size_metrics(self, validator) -> SizeMetrics:
        return SizeMetrics()

    def _analyze_anchor_assignment(self, validator) -> AnchorMetrics:
        am = AnchorMetrics()
        try:
            dataset = getattr(validator, "dataset", None)
            labels = getattr(dataset, "labels", None) if dataset else None
            if labels:
                am.total_objects = sum(len(l) for l in labels)
                am.assigned_objects = am.total_objects
        except Exception:
            pass
        return am

    def _mock_report(self, run_id: str) -> AnalysisReport:
        report = AnalysisReport(
            run_id=run_id, epoch=50,
            overall_map50=0.45, overall_map50_95=0.28,
            class_names=["person", "car", "bicycle", "dog", "cat"],
        )
        for name, ap50, ap, p, r, n, fp_bg, fp_c, fn in [
            ("person",  0.62, 0.41, 0.71, 0.65, 1200, 8,  20, 15),
            ("car",     0.78, 0.55, 0.82, 0.74, 800,  3,  10, 8),
            ("bicycle", 0.21, 0.12, 0.35, 0.42, 150,  15, 30, 45),
            ("dog",     0.18, 0.09, 0.28, 0.36, 90,   20, 25, 50),
            ("cat",     0.43, 0.27, 0.58, 0.55, 120,  12, 18, 22),
        ]:
            report.class_metrics.append(ClassMetrics(
                class_name=name, ap50=ap50, ap50_95=ap,
                precision=p, recall=r, num_instances=n,
                fp_background=fp_bg, fp_class_confusion=fp_c, fn_missed=fn,
            ))
        nc = 5
        matrix = np.eye(nc) * 0.6
        matrix[2, 4] = 0.25
        matrix[3, 4] = 0.30
        report.confusion_matrix = matrix
        report.size_metrics = SizeMetrics(
            small_map50=0.18, medium_map50=0.42, large_map50=0.71,
            small_count=180, medium_count=420, large_count=760,
        )
        report.anchor_metrics = AnchorMetrics(
            total_objects=1360, assigned_objects=1090, unassigned_objects=270
        )
        return report

    # ── Persistence & plots ───────────────────────────────────────────────────

    def _save_report(self, report: AnalysisReport, decision: dict) -> None:
        out_dir = self.experiment_dir / report.run_id / "analysis"
        out_dir.mkdir(parents=True, exist_ok=True)
        data = report.to_dict()
        data["llm_decision"] = {
            "overall_assessment": decision.get("overall_assessment", ""),
            "suggested_priority_focus": decision.get("suggested_priority_focus", ""),
            "ran_analyses": decision.get("run_needed_analyses", []),
            "skip_reasons": decision.get("skip_reasons", {}),
        }
        with open(out_dir / "weakness_report.json", "w") as f:
            json.dump(data, f, indent=2)

    def _generate_plots(self, report: AnalysisReport, analyses: list[str]) -> None:
        plot_dir = self.experiment_dir / report.run_id / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        if "per_class_ap" in analyses and report.class_metrics:
            plot_per_class_ap(report, plot_dir / "per_class_ap.png")
        if "confusion_matrix" in analyses and report.confusion_matrix is not None:
            plot_confusion_matrix(report.confusion_matrix, report.class_names, plot_dir / "confusion_matrix.png")
        if "size_breakdown" in analyses:
            plot_size_breakdown(report, plot_dir / "size_breakdown.png")
