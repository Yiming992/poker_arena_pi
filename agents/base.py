"""Agent interface and the filtered game view passed to agents.

HARD RULE: An AgentGameView NEVER contains opponents' hole cards. The
orchestrator builds it from the canonical GameState before calling decide().
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from poker.models import Action, Card


@dataclass
class OpponentView:
    name: str
    seat: int
    stack: int
    current_bet: int
    total_committed: int
    has_folded: bool
    is_all_in: bool
    is_human: bool


@dataclass
class AgentGameView:
    """Everything an agent is allowed to see when deciding."""
    your_name: str
    your_seat: int
    your_stack: int
    your_hole_cards: Tuple[Card, Card]
    community_cards: List[Card]
    pot: int
    current_bet: int
    your_current_bet: int
    call_amount: int          # chips to add to call
    min_raise_to: int         # minimum total bet for a raise
    max_raise_to: int         # maximum total bet (all-in)
    valid_actions: List[str]
    stage: str
    hand_number: int
    dealer_seat: int
    big_blind: int
    small_blind: int
    opponents: List[OpponentView]
    # Action log for the current hand (player_name, action, amount, stage).
    betting_history: List[dict] = field(default_factory=list)
    # Rolling summaries of recent finished hands for opponent modeling.
    recent_hand_summaries: List[str] = field(default_factory=list)
    # Casual-mode hand-strength hint (v1 default). None in hard mode.
    hand_hint: Optional[str] = None
    # Number of players still in the hand.
    players_in_hand: int = 0


class PokerAgent(ABC):
    """Abstract base class all agents implement."""

    name: str
    model: str
    agent_type: str

    @abstractmethod
    async def decide(self, view: AgentGameView) -> Tuple[Action, str]:
        """Return (action, reasoning_text) for the given filtered view."""
        ...
