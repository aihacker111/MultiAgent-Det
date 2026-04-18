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


class BaseAgent(ABC):
    """
    Base class for all LLM-backed agents.

    Each agent follows the Perceive -> Reason -> Act loop:
      - perceive(): collect relevant signals from env + shared context
      - reason():   call LLM with structured system prompt to decide what to do
      - act():      execute the decided actions using deterministic tools
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
        self.max_tokens: int = llm_cfg.get("max_tokens", 1024)
        self.temperature: float = llm_cfg.get("temperature", 0.1)

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
        else:
            self._llm = None
            self.logger.warning(
                f"{self.AGENT_NAME}: OPENROUTER_API_KEY not set — fallback logic will be used"
            )

        self._setup()

    def _setup(self) -> None:
        """Override for agent-specific init beyond LLM client."""
        pass

    def reason(self, perception: str, output_schema: str) -> dict:
        """
        Core LLM reasoning call.
        perception:    formatted string of what the agent currently observes
        output_schema: JSON schema description for the expected output
        Returns parsed dict, or {} on failure (triggers deterministic fallback).
        """
        if self._llm is None:
            return {}

        user_prompt = "\n\n".join([
            perception,
            f"Respond with a single JSON object matching this schema:\n{output_schema}",
            "Output strict JSON only — no markdown, no explanation outside the JSON.",
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
            self._log_reasoning(raw)
            return self._parse_json(raw)

        except RateLimitError as e:
            self.logger.warning(f"Rate limit hit — using fallback: {e}")
        except APIConnectionError as e:
            self.logger.warning(f"Connection error — using fallback: {e}")
        except APIError as e:
            self.logger.warning(f"API error {e.status_code} — using fallback: {e.message}")
        except Exception as e:
            self.logger.warning(f"Unexpected LLM error — using fallback: {e}")
        return {}

    def _parse_json(self, text: str) -> dict:
        text = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
        self.logger.warning(f"Could not parse LLM JSON:\n{text[:300]}")
        return {}

    def _log_reasoning(self, raw: str) -> None:
        if not self.shared_context or not self.shared_context.current_run_id:
            return
        log_dir = (
            self.experiment_dir / self.shared_context.current_run_id / "reasoning"
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"{self.AGENT_NAME.lower()}_reasoning.txt"
        with open(path, "a") as f:
            f.write(
                f"\n{'='*60}\nIteration {self.shared_context.iteration}\n{'='*60}\n"
            )
            f.write(raw)
            f.write("\n")

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        raise NotImplementedError
