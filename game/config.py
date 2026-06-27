"""Config loading with ${ENV_VAR} expansion."""
from __future__ import annotations

import os
import re
from typing import Any

import yaml

from .orchestrator import SessionConfig

_ENV_RE = re.compile(r"\$\{([A-Z0-9_]+)\}")

# Convenience aliases: if the primary var is unset, fall back to these.
_ENV_ALIASES = {
    "NVIDIA_API_KEY": ["NVIDIA_INFERENCE_API_KEY", "NVIDIA_NIM_API_KEY", "NGC_API_KEY"],
    "GOOGLE_API_KEY": ["GEMINI_API_KEY"],
}


def _lookup_env(name: str) -> str:
    val = os.environ.get(name)
    if val:
        return val
    for alias in _ENV_ALIASES.get(name, []):
        val = os.environ.get(alias)
        if val:
            return val
    return ""


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        def repl(m):
            return _lookup_env(m.group(1))
        return _ENV_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def load_config(path: str) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _expand(raw)


def session_config_from(raw: dict) -> SessionConfig:
    g = raw.get("game", {})
    return SessionConfig(
        starting_stack=g.get("starting_stack", 1000),
        small_blind=g.get("small_blind", 5),
        big_blind=g.get("big_blind", 10),
        max_hands=g.get("max_hands", 100),
        human_starting_stack=g.get("human_starting_stack", 1000),
        human_action_timeout=g.get("human_action_timeout", 60),
        between_hand_delay=g.get("between_hand_delay", 2.0),
        action_delay=g.get("action_delay", 1.0),
        memory_window=g.get("memory_window", 10),
        casual_mode=g.get("casual_mode", True),
    )
