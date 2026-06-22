"""Human agent — bridges WebSocket input to the PokerAgent interface.

When it's the human's turn, the orchestrator calls decide(). This agent
publishes the request (so the server can prompt the UI), then awaits an action
submitted via submit_action(). If the human doesn't act within the timeout, it
auto-folds (or checks if legal) to keep the game moving.
"""
from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Optional, Tuple

from poker.models import Action, ActionType
from .base import AgentGameView, PokerAgent


class HumanAgent(PokerAgent):
    agent_type = "human"

    def __init__(
        self,
        name: str,
        timeout: int = 60,
        on_turn: Optional[Callable[[AgentGameView], Awaitable[None]]] = None,
    ) -> None:
        self.name = name
        self.model = "human"
        self.timeout = timeout
        self._on_turn = on_turn
        self._pending: Optional[asyncio.Future] = None
        self._force_fold = False

    def set_on_turn(self, cb: Callable[[AgentGameView], Awaitable[None]]) -> None:
        self._on_turn = cb

    async def decide(self, view: AgentGameView) -> Tuple[Action, str]:
        if self._force_fold:
            self._force_fold = False
            return self._safe_default(view), ""

        loop = asyncio.get_event_loop()
        self._pending = loop.create_future()
        if self._on_turn:
            await self._on_turn(view)
        try:
            action: Action = await asyncio.wait_for(self._pending, self.timeout)
            return action, ""
        except asyncio.TimeoutError:
            return self._safe_default(view), ""
        finally:
            self._pending = None

    def submit_action(self, action: Action) -> bool:
        """Called by the server when the human submits an action. Returns True
        if the action was accepted (i.e. we were waiting for one)."""
        if self._pending and not self._pending.done():
            self._pending.set_result(action)
            return True
        return False

    def force_fold(self) -> None:
        """Emergency leave mid-hand: fold immediately on next/active turn."""
        self._force_fold = True
        if self._pending and not self._pending.done():
            self._pending.set_result(Action(ActionType.FOLD))

    def _safe_default(self, view: AgentGameView) -> Action:
        if "check" in view.valid_actions:
            return Action(ActionType.CHECK)
        return Action(ActionType.FOLD)
