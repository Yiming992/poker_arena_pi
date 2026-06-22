"""ActionValidator — parses LLM responses into actions and validates legality.

Implements the invalid-action and parse-repair policy from the design:
- Parse the strict ACTION/AMOUNT template (with regex fallbacks).
- Validate the action against the engine's legal moves for the current view.
- On failure, the orchestrator re-prompts once; if that fails, default to
  check (if legal) otherwise fold.

This module is pure and deterministic so it can be golden-tested with recorded
response fixtures (no live API calls).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from agents.base import AgentGameView
from poker.models import Action, ActionType


class ParseError(Exception):
    pass


class ValidationError(Exception):
    pass


@dataclass
class ParsedResponse:
    action: Action
    reasoning: str


# Primary template: ACTION: <verb> AMOUNT: <int>
_ACTION_RE = re.compile(
    r"ACTION:\s*(fold|check|call|raise|all[_\s-]?in)\b"
    r"(?:[^\n]*?AMOUNT:\s*\$?([0-9][0-9,]*))?",
    re.IGNORECASE,
)
# Fallback: a verb possibly followed by a number anywhere late in the text.
_FALLBACK_RE = re.compile(
    r"\b(fold|check|call|raise|bet|all[_\s-]?in)\b\s*\$?([0-9][0-9,]*)?",
    re.IGNORECASE,
)


def _to_int(raw: Optional[str]) -> int:
    if not raw:
        return 0
    return int(raw.replace(",", ""))


def parse_response(text: str) -> ParsedResponse:
    """Extract an Action and reasoning from raw model text.

    Reasoning is the full text (the panel shows the model's whole monologue);
    the action is extracted from the template or a fallback scan.
    """
    if not text or not text.strip():
        raise ParseError("Empty response")

    reasoning = text.strip()

    match = _ACTION_RE.search(text)
    if match:
        verb = match.group(1).lower().replace(" ", "_").replace("-", "_")
        amount = _to_int(match.group(2))
        return ParsedResponse(_verb_to_action(verb, amount), reasoning)

    # Fallback: scan from the end for the last action-like token.
    matches = list(_FALLBACK_RE.finditer(text))
    if matches:
        m = matches[-1]
        verb = m.group(1).lower().replace(" ", "_").replace("-", "_")
        if verb == "bet":
            verb = "raise"
        amount = _to_int(m.group(2))
        return ParsedResponse(_verb_to_action(verb, amount), reasoning)

    raise ParseError("No recognizable action in response")


def _verb_to_action(verb: str, amount: int) -> Action:
    if verb == "fold":
        return Action(ActionType.FOLD)
    if verb == "check":
        return Action(ActionType.CHECK)
    if verb == "call":
        return Action(ActionType.CALL)
    if verb == "all_in":
        return Action(ActionType.ALL_IN, amount=amount)
    if verb == "raise":
        return Action(ActionType.RAISE, amount=amount)
    raise ParseError(f"Unknown verb: {verb}")


def validate(action: Action, view: AgentGameView) -> Action:
    """Return a legal Action or raise ValidationError with a helpful message.

    Normalizes near-misses: a 'raise' to the max becomes all-in; a 'call' with
    nothing to call becomes a check.
    """
    valid = set(view.valid_actions)

    if action.type == ActionType.FOLD:
        if "fold" not in valid:
            raise ValidationError("Fold is not available")
        return action

    if action.type == ActionType.CHECK:
        if "check" not in valid:
            raise ValidationError(
                f"Cannot check; you must call {view.call_amount} or fold/raise"
            )
        return action

    if action.type == ActionType.CALL:
        if "call" not in valid:
            # Nothing to call — interpret as a check if legal.
            if "check" in valid:
                return Action(ActionType.CHECK)
            raise ValidationError("Nothing to call")
        return action

    if action.type in (ActionType.RAISE, ActionType.ALL_IN):
        if "raise" not in valid:
            raise ValidationError(
                "You cannot raise here; valid actions are "
                f"{sorted(valid)}"
            )
        target = action.amount
        if action.type == ActionType.ALL_IN or target >= view.max_raise_to:
            return Action(ActionType.ALL_IN, amount=view.max_raise_to)
        if target < view.min_raise_to:
            raise ValidationError(
                f"Raise must reach at least {view.min_raise_to} "
                f"(you said {target})"
            )
        if target > view.max_raise_to:
            raise ValidationError(
                f"Raise of {target} exceeds your stack "
                f"(max {view.max_raise_to})"
            )
        return Action(ActionType.RAISE, amount=target)

    raise ValidationError(f"Unknown action type: {action.type}")


def parse_and_validate(text: str, view: AgentGameView) -> ParsedResponse:
    parsed = parse_response(text)
    legal = validate(parsed.action, view)
    return ParsedResponse(legal, parsed.reasoning)


def fallback_action(view: AgentGameView) -> Action:
    """Safe default when all parse/repair attempts fail: check if legal else fold."""
    if "check" in view.valid_actions:
        return Action(ActionType.CHECK)
    return Action(ActionType.FOLD)
