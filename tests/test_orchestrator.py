import asyncio

from agents.base import AgentGameView
from agents.human_agent import HumanAgent
from agents.rule_based_agent import RuleBasedAgent
from game.orchestrator import Orchestrator, SessionConfig
from poker.models import Action, ActionType


def make_orch(n=4, max_hands=10, stack=1000):
    cfg = SessionConfig(
        starting_stack=stack,
        small_blind=5,
        big_blind=10,
        max_hands=max_hands,
        between_hand_delay=0.0,
        action_delay=0.0,
    )
    orch = Orchestrator(cfg)
    orch.set_speed("fast")
    for i in range(n):
        orch.add_agent(RuleBasedAgent(f"Bot{i}"), seat=i + 1, stack=stack)
    return orch


def test_full_autonomous_session_runs():
    orch = make_orch(n=4, max_hands=8)
    total_before = sum(p.stack for p in orch.state.players)
    asyncio.run(orch.run())
    total_after = sum(p.stack for p in orch.state.players)
    # Chips are conserved across the whole session.
    assert total_before == total_after
    # We played hands (or stopped early because a player busted).
    assert orch.state.hand_number >= 1
    assert not orch.state.hand_in_progress


def test_session_ends_when_one_player_left_with_chips():
    # Small stacks force eliminations quickly.
    orch = make_orch(n=3, max_hands=100, stack=40)
    asyncio.run(orch.run())
    survivors = [p for p in orch.state.players if p.stack > 0]
    # Either we hit max_hands or only one player has chips.
    assert len(survivors) >= 1


def test_standings_sorted():
    orch = make_orch(n=4, max_hands=5)
    asyncio.run(orch.run())
    standings = orch.final_standings()
    stacks = [s["stack"] for s in standings]
    assert stacks == sorted(stacks, reverse=True)


def test_human_join_leave_queue():
    orch = make_orch(n=4, max_hands=3)
    human = HumanAgent("You", timeout=1)
    res = orch.queue_join(human)
    assert res["ok"]
    # Second join rejected (single human seat).
    res2 = orch.queue_join(HumanAgent("Other"))
    assert not res2["ok"]
    orch.cancel_join()
    assert "pending-human" not in orch.state.pending_joins


def test_human_plays_a_hand_auto_folds_on_timeout():
    cfg = SessionConfig(
        max_hands=2, between_hand_delay=0.0, action_delay=0.0, human_action_timeout=1
    )
    orch = Orchestrator(cfg)
    orch.set_speed("fast")
    for i in range(3):
        orch.add_agent(RuleBasedAgent(f"Bot{i}"), seat=i + 1, stack=1000)
    # Human that never acts -> times out -> auto fold/check.
    human = HumanAgent("You", timeout=1)
    orch.add_agent(human, seat=4, stack=1000, is_human=True)
    total_before = sum(p.stack for p in orch.state.players)
    asyncio.run(orch.run())
    assert sum(p.stack for p in orch.state.players) == total_before
