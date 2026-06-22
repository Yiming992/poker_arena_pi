"""NVIDIA NIM agent — OpenAI-compatible endpoint, reuses the openai SDK.

Models served via NIM (Llama, Mistral, Qwen, etc.) may not support function
calling, so we rely on the strict text template enforced by the validator's
parse+repair loop (same as every other agent).
"""
from __future__ import annotations

from typing import Optional

from .openai_agent import OpenAIAgent

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"


class NvidiaAgent(OpenAIAgent):
    agent_type = "nvidia"

    def __init__(
        self,
        name: str,
        model: str = "meta/llama-3.3-70b-instruct",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 45.0,
    ) -> None:
        super().__init__(
            name=name,
            model=model,
            api_key=api_key,
            base_url=base_url or DEFAULT_BASE_URL,
            timeout=timeout,
        )
