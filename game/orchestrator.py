"""Game orchestrator — drives a complete poker session.

Responsibilities:
- Seat agents, run hands one action at a time in turn order.
- Build filtered AgentGameViews and call agent.decide().
- Apply the invalid-action / API-failure / benching policies.
- Process join/leave queue between hands.
- Broadcast state to clients via an injected broadcast callback.
- Maintain a rolling window of recent hand summaries for opponent modeling.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional

from agents.base import PokerAgent
from agents.human_agent import HumanAgent
from agents.llm_base import AgentDecisionError, APIFailure, LLMAgent
from agents.rule_based_agent import RuleBasedAgent
from poker.engine import IllegalActionError, PokerEngine
from poker.models import (
    Action,
    ActionType,
    GameState,
    PlayerState,
    Stage,
)
from . import projector, validator

BroadcastFn = Callable[[], Awaitable[None]]
NoticeFn = Callable[[str], Awaitable[None]]

MAX_FAILURES_BEFORE_BENCH = 3


@dataclass
class SessionConfig:
    starting_stack: int = 1000
    small_blind: int = 5
    big_blind: int = 10
    max_hands: int = 100
    human_starting_stack: object = 1000  # int or "average"
    human_action_timeout: int = 60
    between_hand_delay: float = 2.0
    action_delay: float = 1.0           # readability pause after each AI action
    memory_window: int = 10
    casual_mode: bool = True            # include hand-strength hints


@dataclass
class SeatedAgent:
    agent: PokerAgent
    player_id: str
    fallback: Optional[PokerAgent] = None  # rule-based replacement when benched


class Orchestrator:
    def __init__(
        self,
        config: SessionConfig,
        broadcast: Optional[BroadcastFn] = None,
        notice: Optional[NoticeFn] = None,
        on_complete: Optional[Callable[[List[dict]], Awaitable[None]]] = None,
    ) -> None:
        self.config = config
        self.engine = PokerEngine(config.small_blind, config.big_blind)
        self.agents: Dict[str, SeatedAgent] = {}
        self._broadcast = broadcast or self._noop
        self._notice = notice or self._noop_notice
        self._on_complete = on_complete
        self.recent_summaries: List[str] = []
        self._running = False
        self._paused = False
        self._stop = False
        self.human_id: Optional[str] = None
        self.human_agent: Optional[HumanAgent] = None
        self._speed = "normal"  # slow | normal | fast

    # ------------------------------------------------------------------ #
    async def _noop(self) -> None:
        return

    async def _noop_notice(self, msg: str) -> None:
        return

    @property
    def state(self) -> GameState:
        return self.engine.state

    def set_speed(self, speed: str) -> None:
        if speed in ("slow", "normal", "fast"):
            self._speed = speed

    def _action_pause(self) -> float:
        return {"slow": 2.0, "normal": self.config.action_delay, "fast": 0.0}[
            self._speed
        ]

    def _hand_pause(self) -> float:
        return {"slow": 4.0, "normal": self.config.between_hand_delay, "fast": 0.2}[
            self._speed
        ]

    # ------------------------------------------------------------------ #
    # Seating
    # ------------------------------------------------------------------ #
    def add_agent(self, agent: PokerAgent, seat: int, stack: int, is_human=False) -> str:
        player_id = self._unique_id(agent.name)
        player = PlayerState(
            player_id=player_id,
            name=agent.name,
            agent_type=agent.agent_type,
            stack=stack,
            seat=seat,
            is_human=is_human,
        )
        # attach model for display
        setattr(player, "model", getattr(agent, "model", ""))
        self.engine.seat_player(player)
        fallback = None
        if isinstance(agent, LLMAgent):
            fallback = RuleBasedAgent(name=agent.name, model="TAG-fallback")
        self.agents[player_id] = SeatedAgent(agent, player_id, fallback)
        if is_human:
            self.human_id = player_id
            if isinstance(agent, HumanAgent):
                self.human_agent = agent
        return player_id

    def _unique_id(self, name: str) -> str:
        base = name.lower().replace(" ", "-")
        pid = base
        i = 1
        existing = {p.player_id for p in self.state.players}
        while pid in existing:
            i += 1
            pid = f"{base}-{i}"
        return pid

    # ------------------------------------------------------------------ #
    # Main session loop
    # ------------------------------------------------------------------ #
    async def run(self) -> None:
        self._running = True
        self._stop = False
        try:
            while not self._stop and self.state.hand_number < self.config.max_hands:
                self._apply_join_leave_queue()
                if not self.engine.can_start_hand():
                    await self._notice("Not enough players with chips. Game paused.")
                    break
                await self._play_hand()
                await self._broadcast()
                if self._stop:
                    break
                await asyncio.sleep(self._hand_pause())
            await self._notice("Session complete.")
            if self._on_complete and not self._stop:
                await self._on_complete(self.final_standings())
        finally:
            self._running = False

    def stop(self) -> None:
        self._stop = True

    async def _play_hand(self) -> None:
        self.engine.start_hand()
        await self._broadcast()
        await asyncio.sleep(self._action_pause())

        safety = 0
        while self.state.hand_in_progress and safety < 1000:
            safety += 1
            player = self.engine.current_player()
            if player is None or not player.can_act:
                # Shouldn't happen, but guard against stalls.
                break
            await self._take_turn(player)
            await self._broadcast()
            if not isinstance(self.agents[player.player_id].agent, HumanAgent):
                await asyncio.sleep(self._action_pause())

        self._summarize_hand()

    # ------------------------------------------------------------------ #
    # A single player's turn (with all the failure policies)
    # ------------------------------------------------------------------ #
    async def _take_turn(self, player: PlayerState) -> None:
        seated = self.agents[player.player_id]
        agent = seated.fallback if player.benched and seated.fallback else seated.agent

        hint = (
            projector.make_hand_hint(self.engine, player.player_id)
            if self.config.casual_mode
            else None
        )
        view = projector.build_agent_view(
            self.engine, player.player_id, self.recent_summaries, hint
        )

        action: Action
        reasoning: str

        if isinstance(agent, HumanAgent):
            action, reasoning = await agent.decide(view)
        elif isinstance(agent, LLMAgent):
            action, reasoning = await self._llm_turn(agent, seated, player, view)
        else:
            # rule-based or other deterministic agent
            try:
                action, reasoning = await agent.decide(view)
                action = validator.validate(action, view)
            except Exception:
                action = validator.fallback_action(view)
                reasoning = "Defaulted to safe action."

        self._safe_apply(action, reasoning, player, view)

    async def _llm_turn(self, agent, seated, player, view):
        try:
            action, reasoning = await agent.decide(view)
            return action, reasoning
        except APIFailure as exc:
            await self._notice(
                f"{player.name}: API error ({exc}). Auto-acting to keep the game moving."
            )
            return validator.fallback_action(view), f"[API failure: {exc}]"
        except AgentDecisionError as exc:
            # Re-prompt once with the error.
            player.failure_count += 1
            try:
                action, reasoning = await agent.decide_with_repair(view, str(exc))
                return action, reasoning
            except (APIFailure, AgentDecisionError, validator.ValidationError,
                    validator.ParseError) as exc2:
                player.failure_count += 1
                await self._maybe_bench(player)
                return (
                    validator.fallback_action(view),
                    f"[Invalid action twice: {exc2}; defaulted]",
                )

    async def _maybe_bench(self, player: PlayerState) -> None:
        if player.failure_count >= MAX_FAILURES_BEFORE_BENCH and not player.benched:
            player.benched = True
            await self._notice(
                f"{player.name} has been benched after "
                f"{player.failure_count} invalid actions. "
                "Rule-based agent taking over its seat."
            )

    def _safe_apply(self, action, reasoning, player, view) -> None:
        try:
            self.engine.apply_action(action, reasoning)
        except IllegalActionError:
            # Last-resort: the engine still rejected it; apply a safe default.
            fb = validator.fallback_action(view)
            self.engine.apply_action(fb, reasoning or "[defaulted]")

    # ------------------------------------------------------------------ #
    # Join / leave queue
    # ------------------------------------------------------------------ #
    def queue_join(self, agent: PokerAgent) -> dict:
        if self.human_id is not None:
            return {"ok": False, "reason": "Human seat already taken."}
        open_seats = self.engine.open_seats()
        if not open_seats:
            return {"ok": False, "reason": "No open seats."}
        if "pending-human" in self.state.pending_joins:
            return {"ok": False, "reason": "Already queued."}
        self._pending_human_agent = agent
        self.state.pending_joins.append("pending-human")
        return {"ok": True}

    def cancel_join(self) -> None:
        if "pending-human" in self.state.pending_joins:
            self.state.pending_joins.remove("pending-human")
        self._pending_human_agent = None

    def queue_leave(self) -> None:
        if self.human_id and self.human_id not in self.state.pending_leaves:
            self.state.pending_leaves.append(self.human_id)

    def emergency_leave(self) -> None:
        """Human clicked Leave mid-hand: fold now, remove seat at hand end."""
        if self.human_agent:
            self.human_agent.force_fold()
        self.queue_leave()

    def _apply_join_leave_queue(self) -> None:
        # Leaves first (free seats), then joins.
        for pid in list(self.state.pending_leaves):
            self.engine.remove_player(pid)
            self.agents.pop(pid, None)
            if pid == self.human_id:
                self.human_id = None
                self.human_agent = None
        self.state.pending_leaves.clear()

        if "pending-human" in self.state.pending_joins:
            self.state.pending_joins.remove("pending-human")
            agent = getattr(self, "_pending_human_agent", None)
            if agent is not None:
                open_seats = self.engine.open_seats()
                if open_seats and self.human_id is None:
                    stack = self._human_stack()
                    self.add_agent(agent, open_seats[0], stack, is_human=True)
                self._pending_human_agent = None

    def _human_stack(self) -> int:
        cfg = self.config.human_starting_stack
        if isinstance(cfg, str) and cfg.lower() == "average":
            ai_stacks = [
                p.stack for p in self.state.players if not p.is_human
            ]
            if ai_stacks:
                return int(sum(ai_stacks) / len(ai_stacks))
            return self.config.starting_stack
        return int(cfg)

    # ------------------------------------------------------------------ #
    # Hand summaries (rolling memory window)
    # ------------------------------------------------------------------ #
    def _summarize_hand(self) -> None:
        r = self.state.last_result
        if not r:
            return
        names = {p.player_id: p.name for p in self.state.players}
        winner_names = [names.get(w, w) for w in r.winners]
        award_str = ", ".join(
            f"{names.get(pid, pid)} +{amt}" for pid, amt in r.pot_awards.items()
        )
        kind = "showdown" if r.showdown else "won uncontested"
        summary = (
            f"Hand #{r.hand_number}: {', '.join(winner_names)} {kind} ({award_str})"
        )
        self.recent_summaries.append(summary)
        if len(self.recent_summaries) > self.config.memory_window:
            self.recent_summaries = self.recent_summaries[-self.config.memory_window :]

    # ------------------------------------------------------------------ #
    # Human action injection (called by server on player_action message)
    # ------------------------------------------------------------------ #
    def submit_human_action(self, action: Action) -> bool:
        if self.human_agent:
            return self.human_agent.submit_action(action)
        return False

    # ------------------------------------------------------------------ #
    def final_standings(self) -> List[dict]:
        return sorted(
            [
                {
                    "name": p.name,
                    "stack": p.stack,
                    "hands_won": p.hands_won,
                    "agent_type": p.agent_type,
                }
                for p in self.state.players
            ],
            key=lambda d: d["stack"],
            reverse=True,
        )
