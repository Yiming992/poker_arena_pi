"""Core data models for Poker Arena.

These are pure data containers with no game logic. The engine owns mutation;
these structures describe the canonical game state and the messages that flow
between engine, orchestrator, and the projection/websocket layers.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

RANKS = "23456789TJQKA"
SUITS = "hdcs"
SUIT_SYMBOLS = {"h": "\u2665", "d": "\u2666", "c": "\u2663", "s": "\u2660"}


@dataclass(frozen=True)
class Card:
    rank: str  # '2'-'9', 'T', 'J', 'Q', 'K', 'A'
    suit: str  # 'h', 'd', 'c', 's'

    def __post_init__(self) -> None:
        if self.rank not in RANKS:
            raise ValueError(f"Invalid rank: {self.rank!r}")
        if self.suit not in SUITS:
            raise ValueError(f"Invalid suit: {self.suit!r}")

    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"

    @property
    def code(self) -> str:
        """Two-char code understood by the treys library, e.g. 'Ah'."""
        return f"{self.rank}{self.suit}"

    @classmethod
    def from_code(cls, code: str) -> "Card":
        return cls(rank=code[0], suit=code[1])


class ActionType(str, Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    RAISE = "raise"
    ALL_IN = "all_in"


class Stage(str, Enum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"


class ViewMode(str, Enum):
    OBSERVER = "observer"  # sees all cards + all reasoning (god-view)
    PLAYER = "player"      # sees only own cards, reasoning after action


@dataclass
class Action:
    """A decided action returned by an agent / accepted by the engine."""
    type: ActionType
    amount: int = 0  # total chips the player is moving to (target bet for raise)

    def __str__(self) -> str:
        if self.type in (ActionType.RAISE, ActionType.ALL_IN) and self.amount:
            return f"{self.type.value} {self.amount}"
        return self.type.value


@dataclass
class PlayerState:
    player_id: str          # unique identifier (e.g. "claude-1", "human")
    name: str               # display name (unique at the table)
    agent_type: str         # 'openai','anthropic','google','nvidia','rule_based','human'
    stack: int
    seat: int               # 1-9
    hole_cards: Optional[Tuple[Card, Card]] = None
    is_active: bool = True          # seated and dealt into the hand
    current_bet: int = 0            # chips committed this betting round
    total_committed: int = 0        # chips committed this whole hand (for side pots)
    has_folded: bool = False
    is_all_in: bool = False
    is_human: bool = False
    has_acted: bool = False         # acted at least once this betting round
    benched: bool = False           # LLM swapped for rule-based fallback
    failure_count: int = 0          # invalid-action failures this session
    hands_won: int = 0

    @property
    def in_hand(self) -> bool:
        """Still contesting the pot (dealt, not folded)."""
        return self.is_active and not self.has_folded

    @property
    def can_act(self) -> bool:
        """Able to make a betting decision (in hand and has chips)."""
        return self.in_hand and not self.is_all_in


@dataclass
class ActionRecord:
    player_name: str
    action: str            # 'fold','check','call','raise','all_in','post_blind'
    amount: Optional[int]
    reasoning: str         # empty string for human players
    stage: str
    hand_number: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class Pot:
    """A (side) pot with the set of players eligible to win it."""
    amount: int
    eligible_player_ids: List[str]


@dataclass
class HandResult:
    """Outcome of a finished hand, produced at showdown / early termination."""
    hand_number: int
    winners: List[str]                 # player_ids that won chips
    pot_awards: dict                   # player_id -> chips awarded
    best_hands: dict                   # player_id -> human-readable hand class
    showdown: bool                     # True if cards were revealed at showdown
    community_cards: List[Card]


@dataclass
class GameState:
    """Canonical single source of truth for a poker game."""
    players: List[PlayerState] = field(default_factory=list)
    community_cards: List[Card] = field(default_factory=list)
    pots: List[Pot] = field(default_factory=list)
    current_bet: int = 0               # highest committed bet this round
    last_raise_size: int = 0           # for min-raise enforcement
    dealer_position: int = 0           # index into players list
    current_player_index: int = 0
    stage: Stage = Stage.PREFLOP
    hand_number: int = 0
    small_blind: int = 5
    big_blind: int = 10
    action_history: List[ActionRecord] = field(default_factory=list)
    pending_joins: List[str] = field(default_factory=list)
    pending_leaves: List[str] = field(default_factory=list)
    hand_in_progress: bool = False
    last_result: Optional[HandResult] = None

    @property
    def total_pot(self) -> int:
        return sum(p.amount for p in self.pots)

    def player_by_id(self, player_id: str) -> Optional[PlayerState]:
        for p in self.players:
            if p.player_id == player_id:
                return p
        return None

    def players_in_hand(self) -> List[PlayerState]:
        return [p for p in self.players if p.in_hand]

    def players_can_act(self) -> List[PlayerState]:
        return [p for p in self.players if p.can_act]
