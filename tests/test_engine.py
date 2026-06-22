import random

import pytest

from poker.engine import IllegalActionError, PokerEngine
from poker.models import Action, ActionType, Card, PlayerState, Stage


def make_player(pid, seat, stack=1000, agent="rule_based"):
    return PlayerState(
        player_id=pid, name=pid, agent_type=agent, stack=stack, seat=seat
    )


def fresh_engine(n=4, stack=1000, seed=42):
    eng = PokerEngine(small_blind=5, big_blind=10, rng=random.Random(seed))
    for i in range(n):
        eng.seat_player(make_player(f"p{i}", seat=i + 1, stack=stack))
    return eng


# ---------------------------------------------------------------------- #
# Seating
# ---------------------------------------------------------------------- #
def test_seating_and_open_seats():
    eng = fresh_engine(3)
    assert eng.seated_count() == 3
    assert eng.open_seats() == [4, 5, 6, 7, 8, 9]


def test_cannot_seat_duplicate_name():
    eng = PokerEngine()
    eng.seat_player(make_player("a", 1))
    with pytest.raises(IllegalActionError):
        eng.seat_player(PlayerState("b", "a", "rule_based", 1000, 2))


def test_table_full():
    eng = PokerEngine()
    for i in range(9):
        eng.seat_player(make_player(f"p{i}", i + 1))
    with pytest.raises(IllegalActionError):
        eng.seat_player(make_player("extra", 1))


# ---------------------------------------------------------------------- #
# Blinds & hand start
# ---------------------------------------------------------------------- #
def test_start_hand_posts_blinds():
    eng = fresh_engine(4)
    eng.start_hand()
    s = eng.state
    assert s.hand_in_progress
    assert s.total_pot == 15  # SB 5 + BB 10
    # Every player dealt two cards.
    for p in s.players:
        assert p.hole_cards is not None and len(p.hole_cards) == 2
    assert s.current_bet == 10


def test_need_two_players():
    eng = fresh_engine(1)
    assert not eng.can_start_hand()
    with pytest.raises(IllegalActionError):
        eng.start_hand()


def test_heads_up_button_posts_sb():
    eng = fresh_engine(2)
    eng.start_hand()
    s = eng.state
    button = s.players[s.dealer_position]
    # In heads-up, button posts the small blind.
    assert button.current_bet == s.small_blind


# ---------------------------------------------------------------------- #
# Betting flow
# ---------------------------------------------------------------------- #
def test_everyone_folds_to_one():
    eng = fresh_engine(4)
    eng.start_hand()
    # Three players fold preflop -> last one wins.
    for _ in range(3):
        eng.apply_action(Action(ActionType.FOLD))
    assert not eng.state.hand_in_progress
    res = eng.state.last_result
    assert res is not None
    assert len(res.winners) == 1
    assert not res.showdown


def test_check_down_to_showdown():
    eng = fresh_engine(2, seed=7)
    eng.start_hand()
    # Heads up: SB (button) acts first preflop. Call then check to river.
    safety = 0
    while eng.state.hand_in_progress and safety < 50:
        safety += 1
        p = eng.current_player()
        to_call = eng.call_amount(p)
        if to_call > 0:
            eng.apply_action(Action(ActionType.CALL))
        else:
            eng.apply_action(Action(ActionType.CHECK))
    assert not eng.state.hand_in_progress
    assert eng.state.stage == Stage.SHOWDOWN
    assert len(eng.state.community_cards) == 5
    res = eng.state.last_result
    assert res.showdown


def test_chip_conservation():
    eng = fresh_engine(4, stack=1000, seed=11)
    total_before = sum(p.stack for p in eng.state.players)
    eng.start_hand()
    safety = 0
    while eng.state.hand_in_progress and safety < 200:
        safety += 1
        p = eng.current_player()
        to_call = eng.call_amount(p)
        if to_call > 0:
            eng.apply_action(Action(ActionType.CALL))
        else:
            eng.apply_action(Action(ActionType.CHECK))
    total_after = sum(p.stack for p in eng.state.players)
    assert total_before == total_after == 4000


def test_illegal_check_facing_bet():
    eng = fresh_engine(3)
    eng.start_hand()
    # UTG faces the big blind; cannot check.
    with pytest.raises(IllegalActionError):
        eng.apply_action(Action(ActionType.CHECK))


def test_raise_below_min_rejected():
    eng = fresh_engine(3)
    eng.start_hand()
    p = eng.current_player()
    # current_bet is 10, min raise to is 20. Raising to 15 is illegal.
    with pytest.raises(IllegalActionError):
        eng.apply_action(Action(ActionType.RAISE, amount=15))


def test_raise_exceeds_stack_rejected():
    eng = fresh_engine(3, stack=100)
    eng.start_hand()
    p = eng.current_player()
    with pytest.raises(IllegalActionError):
        eng.apply_action(Action(ActionType.RAISE, amount=99999))


# ---------------------------------------------------------------------- #
# Side pots
# ---------------------------------------------------------------------- #
def test_side_pot_formation():
    eng = PokerEngine(small_blind=5, big_blind=10, rng=random.Random(3))
    eng.seat_player(make_player("short", 1, stack=100))
    eng.seat_player(make_player("mid", 2, stack=500))
    eng.seat_player(make_player("big", 3, stack=500))
    eng.start_hand()
    # Drive everyone all-in by repeated raises/calls.
    safety = 0
    while eng.state.hand_in_progress and safety < 50:
        safety += 1
        p = eng.current_player()
        if p is None:
            break
        max_to = eng.max_raise_to(p)
        # Push all-in if possible, else call.
        if "raise" in eng.valid_actions(p):
            eng.apply_action(Action(ActionType.ALL_IN, amount=max_to))
        else:
            eng.apply_action(Action(ActionType.CALL))
    # Pot should be fully distributed; chips conserved.
    assert sum(p.stack for p in eng.state.players) == 1100
    assert not eng.state.hand_in_progress


def test_all_in_short_player_eligibility():
    eng = PokerEngine(small_blind=5, big_blind=10, rng=random.Random(99))
    eng.seat_player(make_player("short", 1, stack=50))
    eng.seat_player(make_player("a", 2, stack=1000))
    eng.seat_player(make_player("b", 3, stack=1000))
    eng.start_hand()
    safety = 0
    while eng.state.hand_in_progress and safety < 80:
        safety += 1
        p = eng.current_player()
        if p is None:
            break
        to_call = eng.call_amount(p)
        if to_call > 0:
            eng.apply_action(Action(ActionType.CALL))
        else:
            eng.apply_action(Action(ActionType.CHECK))
    assert sum(p.stack for p in eng.state.players) == 2050


# ---------------------------------------------------------------------- #
# Button rotation across hands
# ---------------------------------------------------------------------- #
def test_button_rotates():
    eng = fresh_engine(4)
    eng.start_hand()
    first_button = eng.state.dealer_position
    # Fold everyone to end the hand quickly.
    while eng.state.hand_in_progress:
        eng.apply_action(Action(ActionType.FOLD))
    eng.start_hand()
    assert eng.state.dealer_position != first_button
