"""Deterministic Tight-Aggressive (TAG) agent.

Two purposes:
1. First-class cheap/fast opponent (config agent: "rule_based").
2. Benching fallback when an LLM agent accumulates too many failures.

Uses no LLM calls. Preflop strength comes from a Sklansky-style group lookup;
postflop strength comes from the treys evaluator against the board. Reasoning
output is a short deterministic string so the panel stays populated.
"""
from __future__ import annotations

from typing import Tuple

from poker.evaluator import rank_percentage
from poker.models import Action, ActionType, Card
from .base import AgentGameView, PokerAgent

# Sklansky-Malmuth-ish preflop groups -> strength tier (0=premium..3=weak).
# Keyed by canonical hand code: rank-pair, or "XYs"/"XYo" (suited/offsuit),
# ranks ordered high-to-low.
_PREMIUM = {"AA", "KK", "QQ", "JJ", "AKs", "AKo", "AQs", "TT"}
_STRONG = {
    "AQo", "AJs", "ATs", "KQs", "99", "88", "AJo", "KJs", "KQo", "ATo",
    "KTs", "QJs", "77", "QTs", "JTs",
}
_PLAYABLE = {
    "A9s", "A8s", "A7s", "A6s", "A5s", "A4s", "A3s", "A2s", "KJo", "QJo",
    "JTo", "K9s", "Q9s", "J9s", "T9s", "98s", "66", "55", "44", "33", "22",
    "KTo", "QTo", "T8s", "87s", "76s", "65s", "54s", "A9o", "A8o",
}

_RANK_ORDER = "23456789TJQKA"


def _preflop_code(cards: Tuple[Card, Card]) -> str:
    a, b = cards
    if _RANK_ORDER.index(a.rank) < _RANK_ORDER.index(b.rank):
        a, b = b, a
    if a.rank == b.rank:
        return a.rank + b.rank
    suited = "s" if a.suit == b.suit else "o"
    return f"{a.rank}{b.rank}{suited}"


def _preflop_tier(cards: Tuple[Card, Card]) -> int:
    code = _preflop_code(cards)
    if code in _PREMIUM:
        return 0
    if code in _STRONG:
        return 1
    if code in _PLAYABLE:
        return 2
    return 3


def _postflop_tier(pct: float) -> int:
    if pct >= 0.90:
        return 0  # premium (top 10%)
    if pct >= 0.75:
        return 1  # strong (top 25%)
    if pct >= 0.50:
        return 2  # playable (top 50%)
    return 3      # weak


_TIER_NAME = {0: "premium", 1: "strong", 2: "playable", 3: "weak"}


class RuleBasedAgent(PokerAgent):
    agent_type = "rule_based"

    def __init__(self, name: str, model: str = "TAG-rules") -> None:
        self.name = name
        self.model = model

    async def decide(self, view: AgentGameView) -> Tuple[Action, str]:
        action, reason = self._decide_sync(view)
        return action, reason

    def _decide_sync(self, view: AgentGameView) -> Tuple[Action, str]:
        preflop = view.stage == "preflop" or not view.community_cards
        if preflop:
            tier = _preflop_tier(view.your_hole_cards)
            pct_str = f"preflop group {tier}"
        else:
            pct = rank_percentage(view.your_hole_cards, view.community_cards)
            tier = _postflop_tier(pct)
            pct_str = f"{pct * 100:.0f}th percentile"

        in_position = self._in_position(view)
        facing_bet = view.call_amount > 0
        can_raise = "raise" in view.valid_actions
        pot_odds = self._pot_odds(view)

        action, rationale = self._strategy(
            view, tier, preflop, in_position, facing_bet, can_raise, pot_odds
        )
        reasoning = (
            f"Hand strength: {_TIER_NAME[tier]} ({pct_str}). "
            f"{'In position. ' if in_position else 'Out of position. '}"
            f"{f'Pot odds {pot_odds:.1f}:1. ' if facing_bet and pot_odds else ''}"
            f"{rationale}"
        )
        return action, reasoning

    def _in_position(self, view: AgentGameView) -> bool:
        active_seats = [view.your_seat] + [
            o.seat for o in view.opponents if not o.has_folded
        ]
        # Crude: in position if you act late relative to the dealer.
        return view.your_seat >= view.dealer_seat

    def _pot_odds(self, view: AgentGameView) -> float:
        if view.call_amount <= 0:
            return 0.0
        return view.pot / view.call_amount

    def _strategy(
        self, view, tier, preflop, in_position, facing_bet, can_raise, pot_odds
    ) -> Tuple[Action, str]:
        bb = view.big_blind
        if preflop:
            if tier == 0:
                target = self._raise_target(view, view.current_bet * 3 if view.current_bet else 3 * bb)
                if can_raise:
                    return Action(ActionType.RAISE, target), "Premium hand: raising for value."
                if facing_bet:
                    return Action(ActionType.CALL), "Premium hand, can't raise: calling."
                return Action(ActionType.CHECK), "Premium hand: checking option."
            if tier == 1:
                target = self._raise_target(view, view.current_bet + 2 * bb if view.current_bet else 2 * bb)
                if can_raise:
                    return Action(ActionType.RAISE, target), "Strong hand: raising."
                if facing_bet:
                    return Action(ActionType.CALL), "Strong hand: calling."
                return Action(ActionType.CHECK), "Strong hand: checking."
            if tier == 2:
                if not facing_bet:
                    return Action(ActionType.CHECK), "Playable hand: taking a free look."
                if in_position and view.call_amount <= 3 * bb:
                    return Action(ActionType.CALL), "Playable in position: calling cheaply."
                return self._fold_or_check(view), "Playable out of position vs raise: folding."
            # Weak
            if not facing_bet:
                return Action(ActionType.CHECK), "Weak hand: checking."
            return self._fold_or_check(view), "Weak hand: folding."

        # Postflop
        pot = view.pot
        if tier == 0:
            if facing_bet and can_raise:
                target = self._raise_target(view, view.current_bet + int(2.5 * view.call_amount))
                return Action(ActionType.RAISE, target), "Premium made hand: raising for value."
            if not facing_bet and can_raise:
                target = self._raise_target(view, max(view.min_raise_to, int(pot * 0.66)))
                return Action(ActionType.RAISE, target), "Premium made hand: betting 2/3 pot."
            if facing_bet:
                return Action(ActionType.CALL), "Premium hand, can't raise: calling."
            return Action(ActionType.CHECK), "Premium hand: checking to trap."
        if tier == 1:
            if not facing_bet and can_raise:
                target = self._raise_target(view, max(view.min_raise_to, int(pot * 0.5)))
                return Action(ActionType.RAISE, target), "Strong hand: betting 1/2 pot."
            if facing_bet:
                return Action(ActionType.CALL), "Strong hand: calling."
            return Action(ActionType.CHECK), "Strong hand: checking."
        if tier == 2:
            if not facing_bet:
                return Action(ActionType.CHECK), "Marginal hand: checking."
            if pot_odds >= 3.0:
                return Action(ActionType.CALL), "Marginal hand with good pot odds: calling."
            return self._fold_or_check(view), "Marginal hand, poor odds: folding."
        # Weak
        if not facing_bet:
            return Action(ActionType.CHECK), "Weak hand: checking."
        return self._fold_or_check(view), "Weak hand facing a bet: folding."

    def _raise_target(self, view: AgentGameView, desired: int) -> int:
        target = max(view.min_raise_to, desired)
        target = min(target, view.max_raise_to)
        return target

    def _fold_or_check(self, view: AgentGameView) -> Action:
        if "check" in view.valid_actions:
            return Action(ActionType.CHECK)
        return Action(ActionType.FOLD)
