"""Google agent — Gemini 2.0 / 2.5 via the google-genai SDK."""
from __future__ import annotations

from typing import Optional

from .llm_base import LLMAgent


class GoogleAgent(LLMAgent):
    agent_type = "google"

    def __init__(
        self,
        name: str,
        model: str = "gemini-2.5-pro",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(name, model, timeout)
        from google import genai

        self._genai = genai
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()

    async def _call_model(self, system: str, user: str) -> str:
        from google.genai import types

        resp = await self._client.aio.models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.8,
                max_output_tokens=700,
            ),
        )
        return resp.text or ""
