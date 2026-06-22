"""FastAPI entry point for Poker Arena.

Serves the web UI and a WebSocket endpoint. One server instance hosts one
session. The first client to send join_request claims the single human seat;
all others remain read-only observers (v1 single-session assumption).
"""
from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import Dict, List, Optional

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from agents.factory import build_agent
from agents.human_agent import HumanAgent
from game import config as cfg_mod
from game import projector
from game.orchestrator import Orchestrator, SessionConfig
from poker.models import Action, ActionType

load_dotenv(os.path.expanduser("~/.env"))
load_dotenv()  # also load local .env if present

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "web" / "static"
TEMPLATE_DIR = BASE_DIR / "web" / "templates"

app: FastAPI  # defined after STATE below


class ConnectionManager:
    """Tracks connected clients and their view mode."""

    def __init__(self) -> None:
        self.clients: List[WebSocket] = []
        self.human_client: Optional[WebSocket] = None

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.clients.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.clients:
            self.clients.remove(ws)
        if ws is self.human_client:
            self.human_client = None

    async def broadcast_state(self, orch: Orchestrator) -> None:
        observer_msg = projector.observer_view(orch.state)
        dead = []
        for ws in self.clients:
            try:
                if ws is self.human_client and orch.human_id:
                    msg = projector.player_view(
                        orch.state, orch.engine, orch.human_id
                    )
                else:
                    msg = observer_msg
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    async def notice(self, text: str) -> None:
        await self._send_all({"type": "notice", "text": text})

    async def session_complete(self, standings: list) -> None:
        await self._send_all({"type": "session_complete", "standings": standings})

    async def _send_all(self, payload: dict) -> None:
        dead = []
        for ws in self.clients:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


class App:
    """Holds the orchestrator and connection manager for the running session."""

    def __init__(self) -> None:
        self.manager = ConnectionManager()
        self.orch: Optional[Orchestrator] = None
        self.raw_config: dict = {}
        self.run_task: Optional[asyncio.Task] = None

    def build_orchestrator(self, raw: dict) -> Orchestrator:
        session_cfg = cfg_mod.session_config_from(raw)
        orch = Orchestrator(
            session_cfg,
            broadcast=lambda: self.manager.broadcast_state(self.orch),
            notice=self.manager.notice,
            on_complete=self.manager.session_complete,
        )
        providers = raw.get("providers", {})
        seat = 1
        for p in raw.get("players", []):
            agent = build_agent(
                name=p["name"],
                agent_type=p["agent"],
                model=p.get("model"),
                providers=providers,
            )
            orch.add_agent(agent, seat=seat, stack=session_cfg.starting_stack)
            seat += 1
        return orch

    async def start_session(self, raw: Optional[dict] = None) -> None:
        if self.run_task and not self.run_task.done():
            self.orch.stop()
            try:
                await asyncio.wait_for(self.run_task, timeout=5)
            except Exception:
                pass
        raw = raw or self.raw_config
        self.raw_config = raw
        self.orch = self.build_orchestrator(raw)
        self.run_task = asyncio.create_task(self.orch.run())


STATE = App()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    cfg_path = os.environ.get(
        "POKER_ARENA_CONFIG", str(BASE_DIR / "poker_arena.yaml")
    )
    raw = cfg_mod.load_config(cfg_path)
    STATE.raw_config = raw
    await STATE.start_session(raw)
    yield
    if STATE.orch:
        STATE.orch.stop()


app = FastAPI(title="Poker Arena", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (TEMPLATE_DIR / "index.html").read_text()


@app.get("/api/config")
async def get_config() -> dict:
    raw = STATE.raw_config
    return {
        "players": [
            {"name": p["name"], "agent": p["agent"], "model": p.get("model", "")}
            for p in raw.get("players", [])
        ],
        "game": raw.get("game", {}),
        "available_agents": ["openai", "anthropic", "google", "nvidia", "rule_based"],
    }


def _make_human_agent() -> HumanAgent:
    timeout = STATE.orch.config.human_action_timeout if STATE.orch else 60
    agent = HumanAgent("You", timeout=timeout)

    async def on_turn(view) -> None:
        # Trigger a fresh broadcast so the human sees their action prompt.
        await STATE.manager.broadcast_state(STATE.orch)

    agent.set_on_turn(on_turn)
    return agent


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await STATE.manager.connect(ws)
    try:
        # Send initial state immediately.
        if STATE.orch:
            await STATE.manager.broadcast_state(STATE.orch)
        while True:
            data = await ws.receive_json()
            await handle_message(ws, data)
    except WebSocketDisconnect:
        # If the human disconnects mid-hand, trigger emergency leave.
        if ws is STATE.manager.human_client and STATE.orch:
            STATE.orch.emergency_leave()
        STATE.manager.disconnect(ws)
    except Exception:
        STATE.manager.disconnect(ws)


async def handle_message(ws: WebSocket, data: dict) -> None:
    orch = STATE.orch
    if not orch:
        return
    mtype = data.get("type")

    if mtype == "join_request":
        if STATE.manager.human_client is not None:
            await ws.send_json(
                {"type": "join_rejected", "reason": "A human seat is already in use."}
            )
            return
        agent = _make_human_agent()
        res = orch.queue_join(agent)
        if res.get("ok"):
            STATE.manager.human_client = ws
            STATE._pending_ws_agent = agent
            await ws.send_json({"type": "join_queued"})
            await STATE.manager.notice("A human is joining the next hand.")
        else:
            await ws.send_json({"type": "join_rejected", "reason": res.get("reason")})

    elif mtype == "cancel_join":
        orch.cancel_join()
        if ws is STATE.manager.human_client and orch.human_id is None:
            STATE.manager.human_client = None
        await ws.send_json({"type": "join_cancelled"})

    elif mtype == "leave_request":
        if ws is STATE.manager.human_client:
            orch.emergency_leave()
            STATE.manager.human_client = None
            await ws.send_json({"type": "left"})
            await STATE.manager.notice("The human player left the table.")

    elif mtype == "player_action":
        if ws is not STATE.manager.human_client:
            return
        action = _parse_client_action(data)
        if action:
            orch.submit_human_action(action)

    elif mtype == "set_speed":
        speed = data.get("speed", "normal")
        orch.set_speed(speed)
        await STATE.manager.notice(f"Speed set to {speed}.")

    elif mtype == "restart":
        new_raw = data.get("config") or STATE.raw_config
        STATE.manager.human_client = None
        await STATE.start_session(_merge_config(new_raw))
        await STATE.manager.notice("New session started.")


def _parse_client_action(data: dict) -> Optional[Action]:
    a = data.get("action")
    amount = int(data.get("amount", 0) or 0)
    mapping = {
        "fold": ActionType.FOLD,
        "check": ActionType.CHECK,
        "call": ActionType.CALL,
        "raise": ActionType.RAISE,
        "all_in": ActionType.ALL_IN,
    }
    if a not in mapping:
        return None
    return Action(mapping[a], amount=amount)


def _merge_config(new_raw: dict) -> dict:
    """Accept a partial config from the UI restart and merge with base."""
    base = dict(STATE.raw_config)
    if "game" in new_raw:
        base["game"] = {**base.get("game", {}), **new_raw["game"]}
    if "players" in new_raw:
        base["players"] = new_raw["players"]
    return base


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Poker Arena server")
    ap.add_argument("--config", default=None, help="Path to YAML config")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    return ap.parse_args()


if __name__ == "__main__":
    import uvicorn

    args = parse_args()
    if args.config:
        os.environ["POKER_ARENA_CONFIG"] = args.config
    uvicorn.run(app, host=args.host, port=args.port)
