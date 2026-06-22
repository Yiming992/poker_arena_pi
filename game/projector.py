"""StateProjector — builds observer/player views and the agent view.

The engine maintains a single canonical GameState. This module projects it
into JSON-serializable dicts for clients, and into the filtered AgentGameView
for LLM/rule agents. The engine never knows about client concerns.
"""
from __future__ import annotations

from typing import List, Optional

from agents.base import AgentGameView, OpponentView
from poker.engine import PokerEngine
from poker.evaluator import hand_class
from poker.models import Card, GameState, PlayerState, Stage


def card_dict(c: Card) -> dict:
    return {"rank": c.rank, "suit": c.suit, "code": c.code}


def _player_public(p: PlayerState) -> dict:
    return {
        "player_id": p.player_id,
        "name": p.name,
        "agent_type": p.agent_type,
        "model": getattr(p, "model", ""),
        "stack": p.stack,
        "seat": p.seat,
        "current_bet": p.current_bet,
        "has_folded": p.has_folded,
        "is_all_in": p.is_all_in,
        "is_active": p.is_active,
        "is_human": p.is_human,
        "benched": p.benched,
        "hands_won": p.hands_won,
    }


def _action_history(state: GameState) -> List[dict]:
    return [
        {
            "player": r.player_name,
            "action": r.action,
            "amount": r.amount,
            "reasoning": r.reasoning,
            "stage": r.stage,
            "hand_number": r.hand_number,
            "timestamp": r.timestamp,
        }
        for r in state.action_history
    ]


def _base_state(state: GameState) -> dict:
    return {
        "hand_number": state.hand_number,
        "stage": state.stage.value,
        "pot": state.total_pot,
        "current_bet": state.current_bet,
        "dealer_seat": state.players[state.dealer_position].seat
        if state.players
        else 0,
        "small_blind": state.small_blind,
        "big_blind": state.big_blind,
        "hand_in_progress": state.hand_in_progress,
        "community_cards": [card_dict(c) for c in state.community_cards],
        "players": [_player_public(p) for p in state.players],
        "pending_joins": list(state.pending_joins),
        "pending_leaves": list(state.pending_leaves),
        "current_player_id": (
            state.players[state.current_player_index].player_id
            if state.hand_in_progress and state.players
            else None
        ),
    }


def _result_dict(state: GameState) -> Optional[dict]:
    r = state.last_result
    if not r:
        return None
    return {
        "hand_number": r.hand_number,
        "winners": r.winners,
        "pot_awards": r.pot_awards,
        "best_hands": r.best_hands,
        "showdown": r.showdown,
        "community_cards": [card_dict(c) for c in r.community_cards],
    }


def observer_view(state: GameState) -> dict:
    """God-view: every player's hole cards and reasoning."""
    base = _base_state(state)
    base["type"] = "game_state"
    base["view"] = "observer"
    all_hole = {}
    show_at_showdown = state.stage == Stage.SHOWDOWN or not state.hand_in_progress
    for p in state.players:
        if p.hole_cards:
            all_hole[p.player_id] = [card_dict(c) for c in p.hole_cards]
        else:
            all_hole[p.player_id] = None
    base["all_hole_cards"] = all_hole
    base["action_history"] = _action_history(state)
    base["last_result"] = _result_dict(state)
    return base


def player_view(state: GameState, engine: PokerEngine, human_id: str) -> dict:
    """Player view: own cards only; opponent cards hidden until showdown;
    reasoning only for actions completed in the current street."""
    base = _base_state(state)
    base["type"] = "game_state"
    base["view"] = "player"
    me = state.player_by_id(human_id)

    base["your_hole_cards"] = (
        [card_dict(c) for c in me.hole_cards] if me and me.hole_cards else None
    )

    show_all = state.stage == Stage.SHOWDOWN or not state.hand_in_progress
    opp_cards = {}
    for p in state.players:
        if p.player_id == human_id:
            continue
        if show_all and p.hole_cards and p.in_hand:
            opp_cards[p.player_id] = [card_dict(c) for c in p.hole_cards]
        else:
            opp_cards[p.player_id] = None
    base["opponent_hole_cards"] = opp_cards

    # Reasoning only for actions completed in the current street.
    cur_stage = state.stage.value
    cur_hand = state.hand_number
    completed = {}
    history = []
    for r in state.action_history:
        rec = {
            "player": r.player_name,
            "action": r.action,
            "amount": r.amount,
            "stage": r.stage,
            "hand_number": r.hand_number,
            # Reasoning only for current street of current hand (or showdown).
            "reasoning": (
                r.reasoning
                if (show_all or (r.stage == cur_stage and r.hand_number == cur_hand))
                else ""
            ),
            "timestamp": r.timestamp,
        }
        history.append(rec)
    base["action_history"] = history
    base["last_result"] = _result_dict(state)

    # Turn / action affordances.
    your_turn = (
        state.hand_in_progress
        and me is not None
        and state.players[state.current_player_index].player_id == human_id
        and me.can_act
    )
    base["your_turn"] = your_turn
    if your_turn and me is not None:
        base["valid_actions"] = engine.valid_actions(me)
        base["call_amount"] = engine.call_amount(me)
        base["min_raise"] = engine.min_raise_to(me)
        base["max_raise"] = engine.max_raise_to(me)
        base["your_current_bet"] = me.current_bet
        base["your_stack"] = me.stack
    return base


def build_agent_view(
    engine: PokerEngine,
    player_id: str,
    recent_summaries: List[str],
    hand_hint: Optional[str],
) -> AgentGameView:
    """Filtered view for an agent — NEVER contains opponents' hole cards."""
    state = engine.state
    me = state.player_by_id(player_id)
    assert me is not None and me.hole_cards is not None

    opponents = [
        OpponentView(
            name=p.name,
            seat=p.seat,
            stack=p.stack,
            current_bet=p.current_bet,
            total_committed=p.total_committed,
            has_folded=p.has_folded,
            is_all_in=p.is_all_in,
            is_human=p.is_human,
        )
        for p in state.players
        if p.player_id != player_id
    ]

    history = [
        {
            "player": r.player_name,
            "action": r.action,
            "amount": r.amount,
            "stage": r.stage,
        }
        for r in state.action_history
        if r.hand_number == state.hand_number
    ]

    return AgentGameView(
        your_name=me.name,
        your_seat=me.seat,
        your_stack=me.stack,
        your_hole_cards=me.hole_cards,
        community_cards=list(state.community_cards),
        pot=state.total_pot,
        current_bet=state.current_bet,
        your_current_bet=me.current_bet,
        call_amount=engine.call_amount(me),
        min_raise_to=engine.min_raise_to(me),
        max_raise_to=engine.max_raise_to(me),
        valid_actions=engine.valid_actions(me),
        stage=state.stage.value,
        hand_number=state.hand_number,
        dealer_seat=state.players[state.dealer_position].seat,
        big_blind=state.big_blind,
        small_blind=state.small_blind,
        opponents=opponents,
        betting_history=history,
        recent_hand_summaries=recent_summaries,
        hand_hint=hand_hint,
        players_in_hand=len(state.players_in_hand()),
    )


def make_hand_hint(engine: PokerEngine, player_id: str) -> Optional[str]:
    """Casual-mode hint describing the player's current best hand."""
    state = engine.state
    me = state.player_by_id(player_id)
    if not me or not me.hole_cards:
        return None
    if not state.community_cards:
        # Preflop: describe the holding.
        a, b = me.hole_cards
        if a.rank == b.rank:
            return f"a pocket pair of {a.rank}s"
        suited = "suited" if a.suit == b.suit else "offsuit"
        return f"{a.rank}{b.rank} {suited}"
    try:
        return "you currently have " + hand_class(
            list(me.hole_cards), state.community_cards
        )
    except Exception:
        return None
