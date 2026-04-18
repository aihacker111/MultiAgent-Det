from __future__ import annotations
import json
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI, APIError, APIConnectionError, RateLimitError

from core.event_bus import EventBus
from core.shared_context import SharedContext
from utils.logger import get_logger

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Per-agent max_tokens key in config["llm"]
_AGENT_TOKEN_KEYS = {
    "MonitorAgent":  "monitor_max_tokens",
    "AnalyzerAgent": "analyzer_max_tokens",
    "MemoryAgent":   "memory_max_tokens",
    "TrainerAgent":  "trainer_max_tokens",
    "PlannerAgent":  "planner_max_tokens",
}


class BaseAgent(ABC):
    """
    Base class for all LLM-backed agents.
    Perceive → Reason (LLM) → Act

    Per-agent max_tokens read from config["llm"]["<agent>_max_tokens"].
    All LLM responses logged to experiments/<run_id>/reasoning/ for debugging.
    """

    SYSTEM_PROMPT: str = ""
    AGENT_NAME: str = "BaseAgent"

    def __init__(
        self,
        config: dict,
        event_bus: EventBus,
        experiment_dir: Path,
        shared_context: Optional[SharedContext] = None,
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self.experiment_dir = experiment_dir
        self.shared_context = shared_context

        self.logger = get_logger(
            self.AGENT_NAME,
            log_file=experiment_dir / "logs" / f"{self.AGENT_NAME.lower()}.log",
            level=config.get("system", {}).get("log_level", "INFO"),
        )

        llm_cfg = config.get("llm", {})
        self.model: str = llm_cfg.get("model", "anthropic/claude-sonnet-4-5")
        self.temperature: float = llm_cfg.get("temperature", 0.1)

        # Per-agent token budget — critical: Analyzer needs 3000+, Monitor only 512
        token_key = _AGENT_TOKEN_KEYS.get(self.AGENT_NAME, "monitor_max_tokens")
        self.max_tokens: int = llm_cfg.get(token_key, 1024)

        api_key = os.environ.get("OPENROUTER_API_KEY")
        site_name = os.environ.get("OPENROUTER_SITE_NAME", "MultiAgent-YOLOTrainer")
        site_url = os.environ.get("OPENROUTER_SITE_URL", "")

        if api_key:
            extra_headers = {"X-Title": site_name}
            if site_url:
                extra_headers["HTTP-Referer"] = site_url
            self._llm = OpenAI(
                api_key=api_key,
                base_url=OPENROUTER_BASE_URL,
                default_headers=extra_headers,
            )
            self.logger.debug(
                f"{self.AGENT_NAME} | model={self.model} | max_tokens={self.max_tokens}"
            )
        else:
            self._llm = None
            self.logger.warning(
                f"{self.AGENT_NAME}: OPENROUTER_API_KEY not set — fallback logic will be used"
            )

        self._setup()

    def _setup(self) -> None:
        pass

    def reason(self, perception: str, output_schema: str) -> dict:
        """
        Core LLM reasoning call.
        Returns parsed dict or {} on failure (triggers deterministic fallback).
        """
        if self._llm is None:
            return {}

        user_prompt = "\n\n".join([
            perception,
            f"Respond with a single JSON object matching this schema:\n{output_schema}",
            "Output STRICT JSON ONLY — no markdown fences, no explanation outside the JSON object.",
        ])

        try:
            response = self._llm.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason

            # Detect truncation before trying to parse
            if finish_reason == "length":
                self.logger.warning(
                    f"{self.AGENT_NAME}: response truncated (finish_reason=length, "
                    f"max_tokens={self.max_tokens}) — attempting partial JSON recovery"
                )
                raw = self._attempt_json_recovery(raw)

            self._log_reasoning(raw, finish_reason)
            result = self._parse_json(raw)
            if not result:
                self.logger.debug(f"{self.AGENT_NAME}: empty parse result → fallback")
            return result

        except RateLimitError as e:
            self.logger.warning(f"Rate limit — using fallback: {e}")
        except APIConnectionError as e:
            self.logger.warning(f"Connection error — using fallback: {e}")
        except APIError as e:
            self.logger.warning(f"API error {e.status_code} — using fallback: {e.message}")
        except Exception as e:
            self.logger.warning(f"Unexpected LLM error — using fallback: {e}")
        return {}

    def _parse_json(self, text: str) -> dict:
        """Parse LLM JSON response. Tries multiple strategies before giving up."""
        # Strip markdown fences
        text = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()

        # Strategy 1: direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: find first {...} block
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # Strategy 3: if text ends mid-string, try truncating to last complete field
        # This handles the common case where JSON is cut off inside a string value
        truncated = self._truncate_to_valid_json(text)
        if truncated:
            try:
                return json.loads(truncated)
            except json.JSONDecodeError:
                pass

        self.logger.warning(f"Could not parse LLM JSON:\n{text[:400]}")
        return {}

    def _truncate_to_valid_json(self, text: str) -> str:
        """
        Attempt to salvage a truncated JSON object by closing unclosed
        arrays/objects. Handles the most common truncation pattern where
        a string field value is cut off mid-sentence.
        """
        text = text.strip()
        if not text.startswith("{"):
            idx = text.find("{")
            if idx == -1:
                return ""
            text = text[idx:]

        # Close any unclosed string by truncating at last complete field
        # Find last comma followed by a complete "key": value pair
        # Simple heuristic: truncate at last '}' or last complete array item
        depth_brace = 0
        depth_bracket = 0
        in_string = False
        escape = False
        last_safe = 0

        for i, ch in enumerate(text):
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth_brace += 1
            elif ch == "}":
                depth_brace -= 1
                if depth_brace == 0:
                    last_safe = i + 1
                    break
            elif ch == "[":
                depth_bracket += 1
            elif ch == "]":
                depth_bracket -= 1

        if last_safe > 0:
            return text[:last_safe]

        # Try to close unclosed structure
        suffix = ("]" * depth_bracket) + ("}" * depth_brace)
        if suffix:
            # Strip trailing comma before closing
            candidate = re.sub(r",\s*$", "", text.rstrip()) + suffix
            return candidate
        return ""

    def _attempt_json_recovery(self, raw: str) -> str:
        """Pre-process truncated response before parsing."""
        raw = raw.strip()
        # If ends mid-string, truncate to last complete line that ends with a value
        lines = raw.split("\n")
        # Remove last line if it looks incomplete (no closing quote/bracket/brace)
        while lines and not re.search(r'["\d\]}\w][\s,]*$', lines[-1].rstrip()):
            lines.pop()
        return "\n".join(lines)

    def _log_reasoning(self, raw: str, finish_reason: str = "") -> None:
        if not self.shared_context or not self.shared_context.current_run_id:
            return
        log_dir = self.experiment_dir / self.shared_context.current_run_id / "reasoning"
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"{self.AGENT_NAME.lower()}_reasoning.txt"
        with open(path, "a") as f:
            f.write(
                f"\n{'='*60}\n"
                f"Iteration {self.shared_context.iteration} | finish={finish_reason}\n"
                f"{'='*60}\n"
            )
            f.write(raw)
            f.write("\n")

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        raise NotImplementedError
