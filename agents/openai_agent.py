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
        resp = await self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.8,
            max_tokens=600,
        )
        return resp.choices[0].message.content or ""
