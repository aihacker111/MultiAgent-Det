from __future__ import annotations
import json
import os
import re
from pathlib import Path
from openai import OpenAI, APIError, APIConnectionError, RateLimitError
from agents.base_agent import BaseAgent, OPENROUTER_BASE_URL
from core.event_bus import EventBus, Event, EventType
from utils.metrics import AnalysisReport, ConfigDelta

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

COT_SYSTEM_PROMPT = """You are an expert computer vision engineer specializing in YOLO object detection optimization.
Your job is to analyze training weaknesses and propose concrete hyperparameter configurations to fix them.

You MUST follow this exact reasoning structure for each proposal:
1. OBSERVATION: What specific metrics/patterns indicate a problem?
2. DIAGNOSIS: What is the root cause of this weakness?
3. HYPOTHESIS: How will the proposed change fix this root cause?
4. CONFIG DELTA: Only the parameters that change (JSON format).
5. EXPECTED DELTA: Estimated mAP50 improvement (e.g. +0.05).
6. CONFIDENCE: 0.0-1.0 score for this proposal.

Rules:
- Output CONFIG DELTA as minimal JSON — only changed parameters vs the current config.
- Never change more than 4 parameters per proposal to avoid confounding.
- Rank proposals by expected impact x confidence.
- Consider past successful fixes when available (from memory).

Valid config parameters: imgsz, batch, lr0, lrf, momentum, weight_decay, warmup_epochs,
warmup_bias_lr, box, cls, dfl, hsv_h, hsv_s, hsv_v, degrees, translate, scale, shear,
perspective, flipud, fliplr, mosaic, mixup, copy_paste, dropout, epochs

Output format (strict JSON array, no markdown):
[
  {
    "observation": "...",
    "diagnosis": "...",
    "hypothesis": "...",
    "config_delta": {"param": value},
    "expected_map_improvement": 0.05,
    "confidence": 0.75,
    "priority": 1,
    "rationale": "one-sentence summary"
  }
]"""


class PlannerAgent(BaseAgent):
    """
    LLM-driven hyperparameter reasoning agent via OpenRouter.
    Uses chain-of-thought to diagnose weaknesses and propose config deltas.
    Integrates with MemoryAgent for retrieval-augmented planning.
    Falls back to rule-based logic when OPENROUTER_API_KEY is not set.
    """

    def _setup(self) -> None:
        cfg = self.config.get("planner", {})
        self.model: str = cfg.get("model", "anthropic/claude-sonnet-4-5")
        self.max_tokens: int = cfg.get("max_tokens", 2048)
        self.temperature: float = cfg.get("temperature", 0.2)
        self.max_suggestions: int = cfg.get("max_suggestions", 3)
        self.confidence_threshold: float = cfg.get("confidence_threshold", 0.6)

        api_key = os.environ.get("OPENROUTER_API_KEY")
        site_url = os.environ.get("OPENROUTER_SITE_URL", "")
        site_name = os.environ.get("OPENROUTER_SITE_NAME", "MultiAgent-YOLOTrainer")

        if not api_key:
            self.logger.warning("OPENROUTER_API_KEY not set — Planner will use rule-based fallback")
            self._client = None
            return

        extra_headers = {"X-Title": site_name}
        if site_url:
            extra_headers["HTTP-Referer"] = site_url

        self._client = OpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            default_headers=extra_headers,
        )
        self.logger.info(f"OpenRouter client ready | model: [bold]{self.model}[/bold]")

    def run(
        self,
        report: AnalysisReport,
        current_config: dict,
        memory_recalls: list[dict] | None = None,
        memory_synthesis: str = "",
        run_id: str = "",
    ) -> list[ConfigDelta]:
        self.logger.info(
            f"[magenta]Planning next experiments[/magenta] for run [bold]{run_id}[/bold] "
            f"({len(report.weaknesses)} weaknesses)"
        )

        if not report.weaknesses:
            self.logger.info("No weaknesses found — no changes needed")
            return []

        if self._client is not None:
            proposals = self._plan_with_llm(report, current_config, memory_recalls or [], memory_synthesis)
        else:
            self.logger.warning("Using rule-based fallback planner (no API key)")
            proposals = self._plan_rule_based(report, current_config)

        proposals = [p for p in proposals if p.confidence >= self.confidence_threshold]
        proposals = proposals[: self.max_suggestions]
        proposals.sort(key=lambda p: p.priority)

        self._save_proposals(proposals, report.run_id)

        self.event_bus.publish(Event(
            type=EventType.PLAN_READY,
            source="PlannerAgent",
            data={"run_id": report.run_id, "proposals": [p.to_dict() for p in proposals]},
        ))

        for i, p in enumerate(proposals):
            self.logger.info(
                f"  Proposal {i+1}: {p.rationale} "
                f"(conf={p.confidence:.2f}, expected D={p.expected_map_improvement:+.3f})"
            )

        return proposals

    def _plan_with_llm(
        self,
        report: AnalysisReport,
        current_config: dict,
        memory_recalls: list[dict],
        memory_synthesis: str = "",
    ) -> list[ConfigDelta]:
        prompt = self._build_prompt(report, current_config, memory_recalls, memory_synthesis)

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": COT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            raw_text = response.choices[0].message.content or ""

            usage = response.usage
            if usage:
                self.logger.debug(
                    f"Tokens — prompt: {usage.prompt_tokens}, "
                    f"completion: {usage.completion_tokens}, "
                    f"total: {usage.total_tokens}"
                )

            self._save_llm_response(raw_text, report.run_id)
            return self._parse_llm_response(raw_text)

        except RateLimitError as exc:
            self.logger.error(f"OpenRouter rate limit exceeded: {exc}")
            return self._plan_rule_based(report, current_config)
        except APIConnectionError as exc:
            self.logger.error(f"OpenRouter connection error: {exc}")
            return self._plan_rule_based(report, current_config)
        except APIError as exc:
            self.logger.error(f"OpenRouter API error (status={exc.status_code}): {exc.message}")
            return self._plan_rule_based(report, current_config)
        except Exception as exc:
            self.logger.error(f"Unexpected error calling OpenRouter: {exc}")
            return self._plan_rule_based(report, current_config)

    def _build_prompt(
        self,
        report: AnalysisReport,
        current_config: dict,
        memory_recalls: list[dict],
        memory_synthesis: str = "",
    ) -> str:
        training_cfg = current_config.get("training", current_config)
        relevant_keys = [
            "imgsz", "batch", "lr0", "lrf", "momentum", "weight_decay",
            "mosaic", "mixup", "copy_paste", "dropout", "hsv_h", "hsv_s", "hsv_v",
            "degrees", "translate", "scale", "fliplr", "flipud",
        ]
        current_params = {k: training_cfg.get(k) for k in relevant_keys if k in training_cfg}

        ctx_block = self.shared_context.to_prompt_block() if self.shared_context else ""
        prompt_parts = [
            ctx_block,
            "",
            "## Current Training Configuration",
            f"```json\n{json.dumps(current_params, indent=2)}\n```",
            "",
            "## Training Results",
            f"- Overall mAP50: {report.overall_map50:.4f}",
            f"- Overall mAP50-95: {report.overall_map50_95:.4f}",
            "",
            "## Detected Weaknesses (prioritized)",
        ]

        for w in report.weaknesses[:8]:
            prompt_parts.append(f"- [{w.get('severity', '?').upper()}] {w.get('message', '')}")

        if report.class_metrics:
            prompt_parts += ["", "## Per-class AP50"]
            weak = sorted(report.class_metrics, key=lambda m: m.ap50)[:6]
            for m in weak:
                prompt_parts.append(
                    f"- {m.class_name}: AP50={m.ap50:.3f}, "
                    f"P={m.precision:.2f}, R={m.recall:.2f}, "
                    f"FP_bg={m.fp_background}, FP_conf={m.fp_class_confusion}, FN={m.fn_missed}"
                )

        sm = report.size_metrics
        analyzer_cfg = self.config.get("analyzer", {})
        if any([sm.small_count, sm.medium_count, sm.large_count]):
            prompt_parts += [
                "",
                "## Object Size Performance",
                f"- Small  (<{analyzer_cfg.get('small_obj_size', 32)}px): mAP50={sm.small_map50:.3f} (n={sm.small_count})",
                f"- Medium: mAP50={sm.medium_map50:.3f} (n={sm.medium_count})",
                f"- Large:  mAP50={sm.large_map50:.3f} (n={sm.large_count})",
            ]

        am = report.anchor_metrics
        if am.total_objects > 0:
            prompt_parts += [
                "",
                "## Anchor Assignment",
                f"- Assignment ratio: {am.assignment_ratio:.1%} ({am.unassigned_objects} unassigned)",
            ]

        if memory_synthesis:
            prompt_parts += ["", f"## Memory Agent Synthesis", f"{memory_synthesis}"]
        if memory_recalls:
            prompt_parts += ["", "## Top Recalled Past Fixes"]
            for rec in memory_recalls[:3]:
                warning = " [WARNING: this fix worsened performance]" if rec.get("is_warning") else ""
                prompt_parts.append(
                    f"- conf={rec.get('adjusted_confidence', 0):.2f} | "
                    f"Fix: {json.dumps(rec.get('config_delta_changes', rec.get('config_delta', {}).get('changes', {})))} | "
                    f"Actual D mAP50: {rec['actual_map_improvement']:+.4f}{warning}"
                )

        prompt_parts += [
            "",
            f"Generate up to {self.max_suggestions} proposals to address the top weaknesses.",
            "Output strict JSON array only.",
        ]

        return "\n".join(prompt_parts)

    def _parse_llm_response(self, text: str) -> list[ConfigDelta]:
        try:
            text = re.sub(r"```(?:json)?\s*", "", text).strip()
            json_match = re.search(r"\[[\s\S]*\]", text)
            if not json_match:
                self.logger.warning("No JSON array found in LLM response")
                return []

            proposals_raw = json.loads(json_match.group(0))
            deltas = []
            for i, raw in enumerate(proposals_raw):
                delta = ConfigDelta(
                    changes=raw.get("config_delta", {}),
                    rationale=raw.get("rationale", ""),
                    confidence=float(raw.get("confidence", 0.5)),
                    expected_map_improvement=float(raw.get("expected_map_improvement", 0.0)),
                    priority=int(raw.get("priority", i + 1)),
                )
                deltas.append(delta)
            return deltas

        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            self.logger.error(f"Failed to parse LLM response: {exc}\nRaw:\n{text[:500]}")
            return []

    def _plan_rule_based(
        self, report: AnalysisReport, current_config: dict
    ) -> list[ConfigDelta]:
        """Deterministic fallback — no API required."""
        proposals = []
        training_cfg = current_config.get("training", current_config)
        weakness_types = {w.get("type") for w in report.weaknesses}

        if "small_object_weakness" in weakness_types or "anchor_assignment_failure" in weakness_types:
            current_imgsz = training_cfg.get("imgsz", 640)
            proposals.append(ConfigDelta(
                changes={"imgsz": min(current_imgsz * 2, 1280), "mosaic": 1.0},
                rationale="Increase image size and mosaic for small objects",
                confidence=0.80,
                expected_map_improvement=0.06,
                priority=1,
            ))

        has_overfit = any(w.get("type") == "overfitting" for w in report.weaknesses)
        if has_overfit or "weak_class" in weakness_types:
            current_lr = training_cfg.get("lr0", 0.01)
            current_wd = training_cfg.get("weight_decay", 0.0005)
            proposals.append(ConfigDelta(
                changes={
                    "lr0": round(current_lr * 0.5, 6),
                    "weight_decay": round(current_wd * 2, 6),
                    "dropout": 0.1,
                },
                rationale="Reduce LR and increase regularization to combat overfitting",
                confidence=0.72,
                expected_map_improvement=0.04,
                priority=2,
            ))

        if "class_confusion" in weakness_types:
            proposals.append(ConfigDelta(
                changes={"mixup": 0.15, "copy_paste": 0.1, "hsv_s": 0.9},
                rationale="Add mixup/copy-paste augmentation for confused classes",
                confidence=0.65,
                expected_map_improvement=0.03,
                priority=3,
            ))

        return proposals

    def _save_proposals(self, proposals: list[ConfigDelta], run_id: str) -> None:
        out_dir = self.experiment_dir / run_id / "planner"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "proposals.json", "w") as f:
            json.dump([p.to_dict() for p in proposals], f, indent=2)

    def _save_llm_response(self, text: str, run_id: str) -> None:
        out_dir = self.experiment_dir / run_id / "planner"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "llm_cot_response.txt", "w") as f:
            f.write(text)
