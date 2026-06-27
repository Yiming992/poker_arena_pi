"""OpenAI agent — GPT-4o / GPT-4o-mini via the openai SDK."""
from __future__ import annotations

from typing import Optional

from .llm_base import LLMAgent


class OpenAIAgent(LLMAgent):
    agent_type = "openai"

    def __init__(
        self,
        name: str,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(name, model, timeout)
        from openai import AsyncOpenAI

        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**kwargs)

    async def _call_model(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        # Some newer models (e.g. gpt-5.x reasoning models) reject non-default
        # temperature and use max_completion_tokens instead of max_tokens. Try
        # the rich call first, then fall back to a minimal one on a 400 that
        # mentions those params.
        try:
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.8,
                max_tokens=600,
            )
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            if "temperature" in msg or "max_tokens" in msg or "max_completion" in msg:
                resp = await self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=800,
                )
            else:
                raise
        return resp.choices[0].message.content or ""
