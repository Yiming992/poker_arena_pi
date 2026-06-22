"""Shared base for LLM-backed poker agents.

Handles:
- Prompt construction (via prompts module).
- API call retry with exponential backoff (1s, 3s) on transient failures.
- Parse + single re-prompt repair loop (the orchestrator handles benching).

On total API failure the agent raises APIFailure so the orchestrator can apply
its auto-check/fold policy. On parse/validation failure it raises AgentDecisionError
with the repaired view info so the orchestrator can re-prompt or default.
"""
from __future__ import annotations

import asyncio
from typing import Optional, Tuple

from poker.models import Action
from . import prompts
from .base import AgentGameView, PokerAgent
from game import validator


class APIFailure(Exception):
    """All API retries exhausted."""


class AgentDecisionError(Exception):
    """Model produced an unparseable or illegal action after repair."""


class LLMAgent(PokerAgent):
    """Base class; subclasses implement _call_model()."""

    RETRY_DELAYS = (1.0, 3.0)

    def __init__(self, name: str, model: str, timeout: float = 30.0) -> None:
        self.name = name
        self.model = model
        self.timeout = timeout

    async def _call_model(self, system: str, user: str) -> str:
        """Provider-specific single API call returning raw text. Override."""
        raise NotImplementedError

    async def _call_with_retry(self, system: str, user: str) -> str:
        last_exc: Optional[Exception] = None
        for attempt in range(len(self.RETRY_DELAYS) + 1):
            try:
                return await asyncio.wait_for(
                    self._call_model(system, user), self.timeout
                )
            except Exception as exc:  # noqa: BLE001 - provider errors vary widely
                last_exc = exc
                if attempt < len(self.RETRY_DELAYS):
                    await asyncio.sleep(self.RETRY_DELAYS[attempt])
        raise APIFailure(str(last_exc))

    async def decide(self, view: AgentGameView) -> Tuple[Action, str]:
        """Single attempt: returns (action, reasoning) or raises.

        The orchestrator owns the repair re-prompt and benching; this method
        does one API round-trip and one parse/validate pass.
        """
        system = prompts.SYSTEM_PROMPT
        user = prompts.build_user_prompt(view)
        text = await self._call_with_retry(system, user)
        try:
            parsed = validator.parse_and_validate(text, view)
        except (validator.ParseError, validator.ValidationError) as exc:
            raise AgentDecisionError(str(exc)) from exc
        return parsed.action, parsed.reasoning

    async def decide_with_repair(
        self, view: AgentGameView, error: str
    ) -> Tuple[Action, str]:
        """Re-prompt once including the error and the valid action list."""
        system = prompts.SYSTEM_PROMPT
        user = prompts.build_user_prompt(view)
        user += (
            f"\n\nYOUR PREVIOUS RESPONSE WAS INVALID: {error}\n"
            f"Valid actions are: {', '.join(view.valid_actions)}. "
            "Respond again and make sure the final line is exactly "
            "'ACTION: <verb> AMOUNT: <int>'."
        )
        text = await self._call_with_retry(system, user)
        parsed = validator.parse_and_validate(text, view)
        return parsed.action, parsed.reasoning
