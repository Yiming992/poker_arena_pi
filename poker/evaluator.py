"""Hand evaluation wrapper around the `treys` library.

Lower score == stronger hand (treys convention). We expose helpers to compare
players at showdown and to produce human-readable hand classes for the UI and
for the casual-mode prompt hints.
"""
from __future__ import annotations

from typing import List, Sequence

from treys import Card as TreysCard
from treys import Evaluator as TreysEvaluator

from .models import Card

_evaluator = TreysEvaluator()


def _to_treys(cards: Sequence[Card]) -> List[int]:
    return [TreysCard.new(c.code) for c in cards]


def evaluate(hole_cards: Sequence[Card], community: Sequence[Card]) -> int:
    """Return the treys hand score (1=best, 7462=worst). Needs 5-7 total cards."""
    board = _to_treys(community)
    hand = _to_treys(hole_cards)
    return _evaluator.evaluate(board, hand)


def hand_class(hole_cards: Sequence[Card], community: Sequence[Card]) -> str:
    """Human-readable class, e.g. 'Two Pair', 'Flush'."""
    score = evaluate(hole_cards, community)
    rank_class = _evaluator.get_rank_class(score)
    return _evaluator.class_to_string(rank_class)


def rank_percentage(hole_cards: Sequence[Card], community: Sequence[Card]) -> float:
    """Fraction of all hands this hand beats (0..1, higher is stronger)."""
    score = evaluate(hole_cards, community)
    return 1.0 - _evaluator.get_five_card_rank_percentage(score)
