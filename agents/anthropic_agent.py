"""Anthropic agent — Claude 3.5 / 4 via the anthropic SDK."""
from __future__ import annotations

from typing import Optional

from .llm_base import LLMAgent


class AnthropicAgent(LLMAgent):
    agent_type = "anthropic"

    def __init__(
        self,
        name: str,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(name, model, timeout)
        from anthropic import AsyncAnthropic

        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        self._client = AsyncAnthropic(**kwargs)

    async def _call_model(self, system: str, user: str) -> str:
        resp = await self._client.messages.create(
            model=self.model,
            system=system,
            max_tokens=700,
            temperature=0.8,
            messages=[{"role": "user", "content": user}],
        )
        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        return "".join(parts)
