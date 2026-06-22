"""Full join -> play a turn -> leave cycle over a real websocket, in-process."""
import asyncio
import json
import os
import sys

import uvicorn
import websockets

os.environ["POKER_ARENA_CONFIG"] = os.path.join(
    os.path.dirname(__file__), "..", "poker_arena.demo.yaml"
)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from server import app  # noqa: E402

PORT = 8766


async def client():
    await asyncio.sleep(1.5)
    async with websockets.connect(f"ws://127.0.0.1:{PORT}/ws") as ws:
        await ws.send(json.dumps({"type": "set_speed", "speed": "fast"}))
        await ws.send(json.dumps({"type": "join_request"}))
        played_turn = False
        got_player_view = False
        left = False
        for _ in range(200):
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), 10))
            except asyncio.TimeoutError:
                break
            t = msg.get("type")
            if t == "game_state" and msg.get("view") == "player":
                got_player_view = True
                if msg.get("your_turn"):
                    va = msg.get("valid_actions", [])
                    act = "check" if "check" in va else ("call" if "call" in va else "fold")
                    await ws.send(json.dumps({"type": "player_action", "action": act, "amount": 0}))
                    played_turn = True
                    if not left:
                        await ws.send(json.dumps({"type": "leave_request"}))
                        left = True
            elif t == "left":
                print("CONFIRMED LEFT")
                break
        print("got_player_view:", got_player_view)
        print("played_turn:", played_turn)
        assert got_player_view, "never received a player view after joining"
        assert played_turn, "never got a turn"
        print("JOIN CYCLE OK")


async def main():
    cfg = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="error")
    server = uvicorn.Server(cfg)
    task = asyncio.create_task(server.serve())
    try:
        await client()
    finally:
        server.should_exit = True
        await task


if __name__ == "__main__":
    asyncio.run(main())
