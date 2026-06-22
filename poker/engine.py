"""Core No-Limit Texas Hold'em engine.

Pure game logic with no AI knowledge. All chip movement goes through engine
methods that recompute pots after every action; the orchestrator never mutates
chip counts directly.

House rules (v1):
- Dead button rule (button may land on / pass an empty seat conceptually; we
  only ever rotate among seated players but never re-seat folded players mid-hand)
- Min raise = size of the last raise (or big blind preflop)
- All-in players are eligible only for pots up to their contribution level
- Heads-up: dealer posts small blind, opponent posts big blind
- No antes; cards always shown at showdown (no mucking)
"""
from __future__ import annotations

import random
from typing import List, Optional

from .deck import Deck
from .evaluator import evaluate, hand_class
from .models import (
    Action,
    ActionRecord,
    ActionType,
    Card,
    GameState,
    HandResult,
    PlayerState,
    Pot,
    Stage,
)


class IllegalActionError(Exception):
    """Raised when an action violates the rules of the current game state."""


class PokerEngine:
    """Owns and mutates a single canonical GameState."""

    MAX_SEATS = 9

    def __init__(
        self,
        small_blind: int = 5,
        big_blind: int = 10,
        rng: Optional[random.Random] = None,
    ) -> None:
        self._rng = rng or random.Random()
        self.state = GameState(small_blind=small_blind, big_blind=big_blind)
        self._deck = Deck(self._rng)

    # ------------------------------------------------------------------ #
    # Seating (dynamic, applied between hands by the orchestrator)
    # ------------------------------------------------------------------ #
    def open_seats(self) -> List[int]:
        taken = {p.seat for p in self.state.players}
        return [s for s in range(1, self.MAX_SEATS + 1) if s not in taken]

    def seat_player(self, player: PlayerState) -> None:
        if self.state.hand_in_progress:
            raise IllegalActionError("Cannot seat a player mid-hand")
        if len(self.state.players) >= self.MAX_SEATS:
            raise IllegalActionError("Table is full")
        if any(p.player_id == player.player_id for p in self.state.players):
            raise IllegalActionError(f"Player {player.player_id} already seated")
        if any(p.name == player.name for p in self.state.players):
            raise IllegalActionError(f"Name {player.name!r} already taken")
        if player.seat in {p.seat for p in self.state.players}:
            raise IllegalActionError(f"Seat {player.seat} occupied")
        self.state.players.append(player)
        self.state.players.sort(key=lambda p: p.seat)

    def remove_player(self, player_id: str) -> None:
        if self.state.hand_in_progress:
            raise IllegalActionError("Cannot remove a player mid-hand")
        self.state.players = [p for p in self.state.players if p.player_id != player_id]

    def seated_count(self) -> int:
        return len(self.state.players)

    # ------------------------------------------------------------------ #
    # Hand lifecycle
    # ------------------------------------------------------------------ #
    def can_start_hand(self) -> bool:
        return (
            not self.state.hand_in_progress
            and len([p for p in self.state.players if p.stack > 0]) >= 2
        )

    def start_hand(self) -> None:
        s = self.state
        if s.hand_in_progress:
            raise IllegalActionError("A hand is already in progress")
        active = [p for p in s.players if p.stack > 0]
        if len(active) < 2:
            raise IllegalActionError("Need at least 2 players with chips to start")

        # Reset per-hand player state for everyone seated with chips.
        for p in s.players:
            p.hole_cards = None
            p.current_bet = 0
            p.total_committed = 0
            p.has_folded = False
            p.is_all_in = False
            p.has_acted = False
            p.is_active = p.stack > 0

        s.community_cards = []
        s.pots = [Pot(amount=0, eligible_player_ids=[])]
        s.current_bet = 0
        s.last_raise_size = s.big_blind
        s.stage = Stage.PREFLOP
        s.hand_number += 1
        s.action_history = [
            r for r in s.action_history if r.hand_number > s.hand_number - 6
        ]  # keep a rolling tail for memory/UI
        s.last_result = None
        s.hand_in_progress = True

        # Fresh shuffled deck with integrity check (misdeal protection).
        self._deck.reset()
        self._deck.verify_integrity()
        self._deck.shuffle()

        self._advance_button(active)
        self._post_blinds(active)
        self._deal_hole_cards(active)
        self._set_first_to_act_preflop(active)
        self._rebuild_pots()

    def _active_players(self) -> List[PlayerState]:
        return [p for p in self.state.players if p.is_active]

    def _advance_button(self, active: List[PlayerState]) -> None:
        """Move the dealer button to the next active seat."""
        s = self.state
        n = len(s.players)
        # Find current button player; advance to next *active* seat.
        start = s.dealer_position
        for offset in range(1, n + 1):
            idx = (start + offset) % n
            if s.players[idx].is_active:
                s.dealer_position = idx
                return
        s.dealer_position = s.players.index(active[0])

    def _seat_order_from(self, start_index: int) -> List[int]:
        """Indices of active players in clockwise order starting after start_index."""
        s = self.state
        n = len(s.players)
        order = []
        for offset in range(1, n + 1):
            idx = (start_index + offset) % n
            if s.players[idx].is_active:
                order.append(idx)
        return order

    def _post_blinds(self, active: List[PlayerState]) -> None:
        s = self.state
        order = self._seat_order_from(s.dealer_position)
        if len(active) == 2:
            # Heads-up: dealer posts SB, the other posts BB.
            sb_idx = s.dealer_position
            bb_idx = order[0]
        else:
            sb_idx = order[0]
            bb_idx = order[1]
        self._commit_blind(s.players[sb_idx], s.small_blind)
        self._commit_blind(s.players[bb_idx], s.big_blind)
        s.current_bet = s.big_blind
        s.last_raise_size = s.big_blind
        self._blind_indices = (sb_idx, bb_idx)

    def _commit_blind(self, player: PlayerState, amount: int) -> None:
        posted = min(amount, player.stack)
        player.stack -= posted
        player.current_bet += posted
        player.total_committed += posted
        if player.stack == 0:
            player.is_all_in = True
        self.state.action_history.append(
            ActionRecord(
                player_name=player.name,
                action="post_blind",
                amount=posted,
                reasoning="",
                stage=self.state.stage.value,
                hand_number=self.state.hand_number,
            )
        )

    def _deal_hole_cards(self, active: List[PlayerState]) -> None:
        # Deal one card at a time, two rounds, starting left of button (poker style).
        order = self._seat_order_from(self.state.dealer_position)
        for _ in range(2):
            for idx in order:
                p = self.state.players[idx]
                card = self._deck.deal_one()
                if p.hole_cards is None:
                    p.hole_cards = (card,)  # type: ignore[assignment]
                else:
                    p.hole_cards = (p.hole_cards[0], card)

    def _set_first_to_act_preflop(self, active: List[PlayerState]) -> None:
        s = self.state
        sb_idx, bb_idx = self._blind_indices
        if len(active) == 2:
            # Heads-up preflop: dealer/SB acts first.
            first = sb_idx
        else:
            # Player left of big blind (UTG).
            order = self._seat_order_from(bb_idx)
            first = order[0]
        s.current_player_index = first

    # ------------------------------------------------------------------ #
    # Betting
    # ------------------------------------------------------------------ #
    def current_player(self) -> Optional[PlayerState]:
        if not self.state.hand_in_progress:
            return None
        p = self.state.players[self.state.current_player_index]
        return p

    def call_amount(self, player: PlayerState) -> int:
        """Chips the player must add to match the current bet (capped at stack)."""
        return min(self.state.current_bet - player.current_bet, player.stack)

    def min_raise_to(self, player: PlayerState) -> int:
        """The minimum total bet a raise must reach."""
        return self.state.current_bet + self.state.last_raise_size

    def max_raise_to(self, player: PlayerState) -> int:
        """Max total bet = everything the player has committed + their stack."""
        return player.current_bet + player.stack

    def valid_actions(self, player: PlayerState) -> List[str]:
        actions: List[str] = ["fold"]
        to_call = self.call_amount(player)
        if to_call == 0:
            actions.append("check")
        else:
            actions.append("call")
        # Can raise/bet if the player has more chips than just calling.
        if player.stack > to_call:
            actions.append("raise")
        return actions

    def apply_action(self, action: Action, reasoning: str = "") -> None:
        """Validate and apply an action for the current player, then advance."""
        s = self.state
        if not s.hand_in_progress:
            raise IllegalActionError("No hand in progress")
        player = self.current_player()
        if player is None or not player.can_act:
            raise IllegalActionError("It is not this player's turn to act")

        to_call = self.call_amount(player)

        if action.type == ActionType.FOLD:
            player.has_folded = True

        elif action.type == ActionType.CHECK:
            if to_call != 0:
                raise IllegalActionError("Cannot check facing a bet")

        elif action.type == ActionType.CALL:
            if to_call == 0:
                raise IllegalActionError("Nothing to call; use check")
            self._move_chips(player, to_call)

        elif action.type in (ActionType.RAISE, ActionType.ALL_IN):
            target = action.amount
            max_to = self.max_raise_to(player)
            min_to = self.min_raise_to(player)
            if action.type == ActionType.ALL_IN:
                target = max_to
            if target > max_to:
                raise IllegalActionError(
                    f"Raise to {target} exceeds stack (max {max_to})"
                )
            # A raise must reach at least the min-raise, UNLESS it's an all-in
            # for less (a short all-in is legal but does not reopen betting).
            is_all_in = target >= max_to
            if target <= s.current_bet:
                raise IllegalActionError("Raise must exceed the current bet")
            if not is_all_in and target < min_to:
                raise IllegalActionError(
                    f"Raise to {target} is below the minimum ({min_to})"
                )
            add = target - player.current_bet
            raise_increment = target - s.current_bet
            self._move_chips(player, add)
            # Only a full (>= min) raise reopens action and updates last_raise_size.
            if target >= min_to or s.current_bet == 0:
                s.last_raise_size = raise_increment
                # Reopen action: everyone else must act again.
                for other in s.players:
                    if other is not player and other.can_act:
                        other.has_acted = False
            s.current_bet = max(s.current_bet, player.current_bet)
        else:
            raise IllegalActionError(f"Unknown action type {action.type}")

        player.has_acted = True
        self._record_action(player, action, reasoning)
        self._rebuild_pots()
        self._advance_turn_or_street()

    def _move_chips(self, player: PlayerState, amount: int) -> None:
        amount = min(amount, player.stack)
        player.stack -= amount
        player.current_bet += amount
        player.total_committed += amount
        if player.stack == 0:
            player.is_all_in = True

    def _record_action(self, player: PlayerState, action: Action, reasoning: str) -> None:
        act_name = action.type.value
        amount: Optional[int] = None
        if action.type == ActionType.CALL:
            amount = self.call_amount(player) if False else player.current_bet
        if action.type in (ActionType.RAISE, ActionType.ALL_IN):
            amount = player.current_bet
            if player.is_all_in:
                act_name = "all_in"
        self.state.action_history.append(
            ActionRecord(
                player_name=player.name,
                action=act_name,
                amount=amount,
                reasoning=reasoning,
                stage=self.state.stage.value,
                hand_number=self.state.hand_number,
            )
        )

    # ------------------------------------------------------------------ #
    # Pot management (main + side pots)
    # ------------------------------------------------------------------ #
    def _rebuild_pots(self) -> None:
        """Recompute main + side pots from each player's total_committed.

        Standard layered side-pot algorithm: sort by committed amount, peel off
        layers at each distinct contribution level.
        """
        s = self.state
        contributions = {
            p.player_id: p.total_committed for p in s.players if p.total_committed > 0
        }
        if not contributions:
            s.pots = [Pot(amount=0, eligible_player_ids=[])]
            return

        # Players still able to win (in hand). Folded players' chips stay in pot
        # but they are not eligible to win.
        eligible_ids = {p.player_id for p in s.players if p.in_hand}

        pots: List[Pot] = []
        remaining = dict(contributions)
        while any(v > 0 for v in remaining.values()):
            positive = [v for v in remaining.values() if v > 0]
            level = min(positive)
            contributors = [pid for pid, v in remaining.items() if v > 0]
            amount = level * len(contributors)
            layer_eligible = [pid for pid in contributors if pid in eligible_ids]
            pots.append(Pot(amount=amount, eligible_player_ids=layer_eligible))
            for pid in contributors:
                remaining[pid] -= level

        # Merge consecutive pots with identical eligibility sets for cleanliness.
        merged: List[Pot] = []
        for pot in pots:
            if merged and set(merged[-1].eligible_player_ids) == set(
                pot.eligible_player_ids
            ):
                merged[-1].amount += pot.amount
            else:
                merged.append(pot)
        s.pots = merged or [Pot(amount=0, eligible_player_ids=[])]

    # ------------------------------------------------------------------ #
    # Turn / street advancement
    # ------------------------------------------------------------------ #
    def _betting_round_complete(self) -> bool:
        s = self.state
        in_hand = s.players_in_hand()
        if len(in_hand) <= 1:
            return True
        actors = s.players_can_act()
        if not actors:
            return True
        # Everyone who can act must have acted AND matched the current bet.
        for p in actors:
            if not p.has_acted:
                return False
            if p.current_bet != s.current_bet:
                return False
        return True

    def _next_actor_index(self) -> Optional[int]:
        s = self.state
        n = len(s.players)
        for offset in range(1, n + 1):
            idx = (s.current_player_index + offset) % n
            if s.players[idx].can_act:
                return idx
        return None

    def _advance_turn_or_street(self) -> None:
        s = self.state
        # If only one player remains in the hand, award and end immediately.
        if len(s.players_in_hand()) <= 1:
            self._end_hand()
            return

        if self._betting_round_complete():
            self._advance_street()
            return

        nxt = self._next_actor_index()
        if nxt is None:
            self._advance_street()
        else:
            s.current_player_index = nxt

    def _advance_street(self) -> None:
        s = self.state
        # Reset per-round betting state.
        for p in s.players:
            p.current_bet = 0
            p.has_acted = False
        s.current_bet = 0
        s.last_raise_size = s.big_blind

        # If <=1 players can still act (rest all-in), run out the board to showdown.
        actionable = len(s.players_can_act())

        if s.stage == Stage.PREFLOP:
            s.stage = Stage.FLOP
            self._deck.deal_one()  # burn
            s.community_cards.extend(self._deck.deal(3))
        elif s.stage == Stage.FLOP:
            s.stage = Stage.TURN
            self._deck.deal_one()
            s.community_cards.extend(self._deck.deal(1))
        elif s.stage == Stage.TURN:
            s.stage = Stage.RIVER
            self._deck.deal_one()
            s.community_cards.extend(self._deck.deal(1))
        elif s.stage == Stage.RIVER:
            self._end_hand()
            return
        else:
            self._end_hand()
            return

        # If nobody can act anymore, keep dealing until the river then showdown.
        if actionable < 2:
            self._advance_street()
            return

        # First to act postflop is the first active player left of the button.
        order = self._seat_order_from(s.dealer_position)
        order = [idx for idx in order if s.players[idx].can_act]
        if order:
            s.current_player_index = order[0]
        else:
            self._advance_street()

    # ------------------------------------------------------------------ #
    # Showdown / pot award
    # ------------------------------------------------------------------ #
    def _end_hand(self) -> None:
        s = self.state
        in_hand = s.players_in_hand()
        pot_awards: dict = {}
        best_hands: dict = {}
        winners: List[str] = []

        if len(in_hand) == 1:
            # Everyone else folded; award entire pot, no showdown reveal needed.
            winner = in_hand[0]
            total = s.total_pot
            winner.stack += total
            winner.hands_won += 1
            pot_awards[winner.player_id] = total
            winners.append(winner.player_id)
            showdown = False
        else:
            showdown = True
            # Evaluate every remaining player's best 5-card hand.
            scores = {
                p.player_id: evaluate(list(p.hole_cards), s.community_cards)
                for p in in_hand
            }
            for p in in_hand:
                best_hands[p.player_id] = hand_class(list(p.hole_cards), s.community_cards)

            # Award each pot to eligible player(s) with the best (lowest) score.
            for pot in s.pots:
                if pot.amount == 0:
                    continue
                contenders = [
                    pid for pid in pot.eligible_player_ids if pid in scores
                ]
                if not contenders:
                    # Every eligible contributor for this (side) pot folded.
                    # The chips can't vanish: award them to the remaining
                    # in-hand players via normal showdown evaluation.
                    contenders = list(scores.keys())
                best = min(scores[pid] for pid in contenders)
                pot_winners = [pid for pid in contenders if scores[pid] == best]
                self._split_pot(pot.amount, pot_winners, pot_awards)
                for pid in pot_winners:
                    if pid not in winners:
                        winners.append(pid)
            for pid in winners:
                p = s.player_by_id(pid)
                if p:
                    p.hands_won += 1

        s.last_result = HandResult(
            hand_number=s.hand_number,
            winners=winners,
            pot_awards=pot_awards,
            best_hands=best_hands,
            showdown=showdown,
            community_cards=list(s.community_cards),
        )
        s.stage = Stage.SHOWDOWN
        s.hand_in_progress = False

    def _split_pot(self, amount: int, winner_ids: List[str], awards: dict) -> None:
        """Split a pot evenly; odd chips go to first player left of the button."""
        s = self.state
        n = len(winner_ids)
        share = amount // n
        remainder = amount - share * n
        for pid in winner_ids:
            p = s.player_by_id(pid)
            if p:
                p.stack += share
            awards[pid] = awards.get(pid, 0) + share
        # Distribute odd chips starting from the first seat left of the button.
        if remainder > 0:
            order = self._seat_order_from(s.dealer_position)
            ordered_winners = [
                s.players[idx].player_id
                for idx in order
                if s.players[idx].player_id in winner_ids
            ]
            i = 0
            while remainder > 0 and ordered_winners:
                pid = ordered_winners[i % len(ordered_winners)]
                p = s.player_by_id(pid)
                if p:
                    p.stack += 1
                awards[pid] = awards.get(pid, 0) + 1
                remainder -= 1
                i += 1
