"""NVIDIA Inference API agent.

A single NVIDIA inference API key gives access to many models behind two
request schemas on the same host (https://inference-api.nvidia.com):

- OpenAI-compatible:   POST /v1/chat/completions   (e.g. openai/openai/gpt-5.5,
                                                     meta/llama-*, mistral, qwen)
- Anthropic-compatible: POST /v1/messages           (e.g. aws/anthropic/
                                                      bedrock-claude-opus-4-8)

The model string itself doesn't tell us which schema to use, so the config
selects it via an `api` field: "chat" (default) or "messages".

    - { name: "GPT-5.5", agent: "nvidia", model: "openai/openai/gpt-5.5" }
    - { name: "Opus",    agent: "nvidia", model: "aws/anthropic/bedrock-claude-opus-4-8", api: "messages" }

Both paths reuse the strict ACTION/AMOUNT text template + parse/repair loop, so
no function calling is required.
"""
from __future__ import annotations

from typing import Optional

from .anthropic_agent import AnthropicAgent
from .openai_agent import OpenAIAgent

# Host that serves both /v1/chat/completions and /v1/messages.
DEFAULT_HOST = "https://inference-api.nvidia.com"
DEFAULT_CHAT_BASE_URL = f"{DEFAULT_HOST}/v1"


class _NvidiaChatAgent(OpenAIAgent):
    """OpenAI-compatible NVIDIA models via /v1/chat/completions."""

    agent_type = "nvidia"

    def __init__(
        self,
        name: str,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 45.0,
    ) -> None:
        super().__init__(
            name=name,
            model=model,
            api_key=api_key,
            base_url=base_url or DEFAULT_CHAT_BASE_URL,
            timeout=timeout,
        )


class _NvidiaMessagesAgent(AnthropicAgent):
    """Anthropic-compatible NVIDIA models via /v1/messages.

    Reuses the anthropic SDK pointed at the NVIDIA host (the SDK appends
    /v1/messages to the base_url).
    """

    agent_type = "nvidia"

    def __init__(
        self,
        name: str,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 45.0,
    ) -> None:
        from anthropic import AsyncAnthropic

        # Initialise LLMAgent state without building the default client.
        self.name = name
        self.model = model
        self.timeout = timeout
        # The anthropic SDK appends /v1/messages, so the base_url must be the
        # bare host. Strip a trailing /v1 if the shared provider config supplied
        # the chat-style base_url.
        host = (base_url or DEFAULT_HOST).rstrip("/")
        if host.endswith("/v1"):
            host = host[: -len("/v1")]
        kwargs = {"base_url": host}
        if api_key:
            kwargs["api_key"] = api_key
        self._client = AsyncAnthropic(**kwargs)


def NvidiaAgent(
    name: str,
    model: str = "meta/llama-3.3-70b-instruct",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 45.0,
    api: str = "chat",
):
    """Factory returning the right NVIDIA agent for the requested schema."""
    if api == "messages":
        return _NvidiaMessagesAgent(
            name=name, model=model, api_key=api_key, base_url=base_url, timeout=timeout
        )
    return _NvidiaChatAgent(
        name=name, model=model, api_key=api_key, base_url=base_url, timeout=timeout
    )
