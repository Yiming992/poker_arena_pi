"""Shared prompt templates for LLM poker agents.

V1 ships in "casual mode": hand-strength hints are included to reduce
hallucination and produce more entertaining, readable play.
"""
from __future__ import annotations

from poker.models import SUIT_SYMBOLS
from .base import AgentGameView

SYSTEM_PROMPT = """You are a skilled, opinionated No-Limit Texas Hold'em player \
competing in an AI poker arena against other AI models and possibly a human. \
You have a distinct playing style and you commit to it. Your goal is to win chips.

For every decision you must:
1. Assess your hand strength and how it connects with the board.
2. Consider your position, the pot odds, and the bet you're facing.
3. Read your opponents based on their actions this hand and recent history.
4. Decide on an action and justify the sizing.

Be concise but reveal your genuine thinking — this is a spectator sport and \
people are reading your reasoning. Show personality.

You MUST end your response with a single action line in EXACTLY this format:
ACTION: <fold|check|call|raise> AMOUNT: <integer>

Rules for the action line:
- For fold/check/call, AMOUNT must be 0.
- For raise, AMOUNT is the TOTAL number of chips you want your bet to reach \
this round (not the additional amount). It must be between min_raise and \
max_raise shown below.
- Only choose an action from the listed valid actions."""


def _fmt_cards(cards) -> str:
    if not cards:
        return "(none)"
    return " ".join(f"{c.rank}{SUIT_SYMBOLS[c.suit]}" for c in cards)


def build_user_prompt(view: AgentGameView) -> str:
    lines = []
    lines.append(f"=== Hand #{view.hand_number} — {view.stage.upper()} ===")
    lines.append(f"You are {view.your_name} in seat {view.your_seat}.")
    lines.append(
        f"Blinds: {view.small_blind}/{view.big_blind}. "
        f"Dealer button is at seat {view.dealer_seat}."
    )
    lines.append("")
    lines.append(f"Your hole cards: {_fmt_cards(view.your_hole_cards)}")
    lines.append(f"Community cards: {_fmt_cards(view.community_cards)}")
    if view.hand_hint:
        lines.append(f"Hand read (hint): {view.hand_hint}")
    lines.append("")
    lines.append(f"Pot: {view.pot} chips")
    lines.append(f"Your stack: {view.your_stack} chips")
    lines.append(f"Your chips in this round: {view.your_current_bet}")
    lines.append(
        f"Highest bet this round: {view.current_bet} "
        f"(it costs you {view.call_amount} to call)"
    )
    lines.append("")
    lines.append("Opponents:")
    for o in view.opponents:
        status = []
        if o.has_folded:
            status.append("folded")
        if o.is_all_in:
            status.append("all-in")
        if o.is_human:
            status.append("HUMAN")
        tag = f" [{', '.join(status)}]" if status else ""
        lines.append(
            f"  - {o.name} (seat {o.seat}): stack {o.stack}, "
            f"bet this round {o.current_bet}{tag}"
        )
    lines.append("")

    if view.betting_history:
        lines.append("Action so far this hand:")
        for h in view.betting_history:
            amt = f" {h['amount']}" if h.get("amount") else ""
            lines.append(f"  [{h['stage']}] {h['player']} {h['action']}{amt}")
        lines.append("")

    if view.recent_hand_summaries:
        lines.append("Recent hands (for opponent reads):")
        for s in view.recent_hand_summaries:
            lines.append(f"  - {s}")
        lines.append("")

    lines.append(f"Valid actions: {', '.join(view.valid_actions)}")
    if "call" in view.valid_actions:
        lines.append(f"Call costs: {view.call_amount}")
    if "raise" in view.valid_actions:
        lines.append(
            f"Raise: total bet between {view.min_raise_to} (min) and "
            f"{view.max_raise_to} (max / all-in)"
        )
    lines.append("")
    lines.append(
        "Explain your thinking, then end with the action line "
        "(ACTION: ... AMOUNT: ...)."
    )
    return "\n".join(lines)
