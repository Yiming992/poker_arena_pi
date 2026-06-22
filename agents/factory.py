"""Agent factory — builds PokerAgent instances from config entries."""
from __future__ import annotations

from typing import Optional

from .base import PokerAgent
from .rule_based_agent import RuleBasedAgent


def build_agent(
    name: str,
    agent_type: str,
    model: Optional[str] = None,
    providers: Optional[dict] = None,
) -> PokerAgent:
    providers = providers or {}

    if agent_type == "rule_based":
        return RuleBasedAgent(name=name, model=model or "TAG-rules")

    if agent_type == "openai":
        from .openai_agent import OpenAIAgent

        cfg = providers.get("openai", {})
        return OpenAIAgent(
            name=name,
            model=model or "gpt-4o",
            api_key=cfg.get("api_key"),
            base_url=cfg.get("base_url"),
        )

    if agent_type == "anthropic":
        from .anthropic_agent import AnthropicAgent

        cfg = providers.get("anthropic", {})
        return AnthropicAgent(
            name=name, model=model or "claude-sonnet-4-20250514",
            api_key=cfg.get("api_key"),
        )

    if agent_type == "google":
        from .google_agent import GoogleAgent

        cfg = providers.get("google", {})
        return GoogleAgent(
            name=name, model=model or "gemini-2.5-pro", api_key=cfg.get("api_key")
        )

    if agent_type == "nvidia":
        from .nvidia_agent import NvidiaAgent

        cfg = providers.get("nvidia", {})
        return NvidiaAgent(
            name=name,
            model=model or "meta/llama-3.3-70b-instruct",
            api_key=cfg.get("api_key"),
            base_url=cfg.get("base_url"),
        )

    raise ValueError(f"Unknown agent type: {agent_type!r}")
