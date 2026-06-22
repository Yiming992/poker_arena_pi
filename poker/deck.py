"""Standard 52-card deck with shuffle/deal and integrity checks."""
from __future__ import annotations

import random
from typing import List, Optional

from .models import RANKS, SUITS, Card


class Deck:
    def __init__(self, rng: Optional[random.Random] = None) -> None:
        self._rng = rng or random.Random()
        self.cards: List[Card] = []
        self.reset()

    def reset(self) -> None:
        self.cards = [Card(r, s) for s in SUITS for r in RANKS]

    def shuffle(self) -> None:
        self._rng.shuffle(self.cards)

    def deal(self, n: int = 1) -> List[Card]:
        if n > len(self.cards):
            raise ValueError("Not enough cards left in the deck to deal")
        dealt, self.cards = self.cards[:n], self.cards[n:]
        return dealt

    def deal_one(self) -> Card:
        return self.deal(1)[0]

    def verify_integrity(self) -> None:
        """Misdeal protection: ensure no duplicates and a full 52-card universe."""
        if len(set(self.cards)) != len(self.cards):
            raise ValueError("Deck integrity error: duplicate cards detected")
        full = {Card(r, s) for s in SUITS for r in RANKS}
        if not set(self.cards).issubset(full):
            raise ValueError("Deck integrity error: unknown cards detected")

    def __len__(self) -> int:
        return len(self.cards)
