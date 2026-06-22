"""In-process smoke test: spin up the server and drive it via websocket.

Run: python tests/manual_ws_smoke.py
"""
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

PORT = 8765


async def run_client():
    await asyncio.sleep(2.0)
    uri = f"ws://127.0.0.1:{PORT}/ws"
    async with websockets.connect(uri) as ws:
        got = 0
        last = None
        stages = set()
        joined = False
        while got < 60:
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), 10))
            except asyncio.TimeoutError:
                break
            if msg["type"] == "game_state":
                got += 1
                last = msg
                stages.add(msg.get("stage"))
                if got == 3 and not joined:
                    await ws.send(json.dumps({"type": "join_request"}))
                    joined = True
            elif msg["type"] == "notice":
                print("NOTICE:", msg["text"])
            elif msg["type"] == "join_queued":
                print("JOIN QUEUED OK")
        print("states:", got, "stages:", stages)
        assert got > 5, "expected multiple game states"
        assert "preflop" in stages
        hist = last.get("action_history", []) if last else []
        wr = [h for h in hist if h.get("reasoning")]
        print("actions:", len(hist), "with reasoning:", len(wr))
        assert wr, "expected reasoning in history"
        print("sample reasoning:", wr[-1]["player"], "->", wr[-1]["reasoning"][:80])
    print("SMOKE OK")


async def main():
    config = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())
    try:
        await run_client()
    finally:
        server.should_exit = True
        await server_task


if __name__ == "__main__":
    asyncio.run(main())
