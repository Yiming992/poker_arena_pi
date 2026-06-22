# Poker Arena 🂡

**Watch AI models battle it out at a No-Limit Texas Hold'em table — and jump in to play against them.**

Poker Arena seats multiple LLMs (Claude, GPT-4o, Gemini, Llama via NVIDIA NIM, and a deterministic rule-based bot) at a poker table, lets them play hands autonomously, and exposes **every model's full decision-making reasoning** in real time. You watch as a spectator with god-view of all cards and all thinking — then click **"Join Next Hand"** to take a seat and try to beat them yourself.

> The core value is watching AI reasoning, not just playing poker. Each model's "personality" — tight-aggressive, calling station, overbluffer — emerges naturally from how it reasons.

## Features

- **Transparent AI reasoning** — every decision comes with the model's inner monologue, shown live in the reasoning panel.
- **4+ models at one table** — OpenAI, Anthropic, Google, NVIDIA NIM, and a rule-based TAG bot.
- **Observer-to-player** — watch with full god-view, then join the next hand as a live player. Leave anytime between hands.
- **Robust engine** — full side-pot logic, dynamic seating (1–9 seats), heads-up rules, validated against a 20,000-hand fuzz test (zero chip leaks).
- **Self-healing agents** — invalid actions get re-prompted once, then default to check/fold; models that misbehave 3 times are benched and replaced by the rule-based fallback. API failures retry with backoff and never block the game.
- **Runs with zero API keys** — a demo config seats four rule-based bots so you can see the whole thing working immediately.

## Quick start (under 5 minutes)

```bash
git clone https://github.com/Yiming992/poker_arena_pi.git
cd poker_arena_pi

# Set up a virtualenv and install deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Option A — zero-key demo (4 deterministic bots, no API keys needed)
python server.py --config poker_arena.demo.yaml

# Option B — real LLMs (see "API keys" below)
python server.py --config poker_arena.yaml
```

Open **http://127.0.0.1:8000** and watch the arena. Click **Join Next Hand** to play.

## API keys

LLM agents bring your own keys. Export them (or put them in a `.env` file in the
project root — it's auto-loaded):

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=...
export NVIDIA_API_KEY=nvapi-...
```

Then edit `poker_arena.yaml` to choose which models sit at the table. The
`${ENV_VAR}` placeholders in the config are expanded automatically.

Install only the SDKs you need:

```bash
pip install openai        # OpenAI + NVIDIA NIM (OpenAI-compatible)
pip install anthropic     # Claude
pip install google-genai  # Gemini
```

## Configuration

```yaml
game:
  starting_stack: 1000
  small_blind: 5
  big_blind: 10
  max_hands: 100
  human_starting_stack: 1000   # or "average" to match seated AI stacks
  human_action_timeout: 60     # seconds before auto-fold
  casual_mode: true            # include hand-strength hints in prompts

players:
  - { name: "Claude",  agent: "anthropic", model: "claude-sonnet-4-20250514" }
  - { name: "GPT-4o",  agent: "openai",    model: "gpt-4o" }
  - { name: "Gemini",  agent: "google",    model: "gemini-2.5-pro" }
  - { name: "Llama",   agent: "nvidia",    model: "meta/llama-3.3-70b-instruct" }
  - { name: "RuleBot", agent: "rule_based" }
```

**Speed control** (in the UI): *slow* adds readability delays, *normal* is the
default, *fast* runs at API speed for batch play.

## How it works

```
Browser (vanilla JS) ── WebSocket ──> FastAPI server
                                          │
                                   Orchestrator  ── join/leave queue, policies
                                          │
                                    Poker Engine  ── pure rules, side pots
                                          │
                                  Agent interface  ── one filtered view per agent
```

- The **engine** owns a single canonical `GameState` and all chip movement.
- The **orchestrator** runs hands turn by turn, builds a filtered `AgentGameView`
  for each agent (which **never** contains opponents' hole cards), applies the
  validation/repair/benching/API-failure policies, and broadcasts state.
- The **projector** turns the canonical state into an *observer view* (god-view)
  or a *player view* (own cards only, opponent cards hidden until showdown,
  reasoning only for completed actions in the current street).

See [`poker-arena-design.md`](poker-arena-design.md) equivalent design notes in
the source for the full architecture.

## Development

```bash
pip install pytest
pytest -q                       # 38 unit + integration tests
python tests/manual_ws_smoke.py # in-process end-to-end websocket smoke test
```

Test coverage includes the hand evaluator, full betting flow, side-pot formation,
chip conservation, the action parser/validator (golden fixtures, no live API
calls), the rule-based agent, and a full autonomous orchestrator session with
human join/leave/timeout.

## Roadmap (not in v1)

ELO ranking with a SQLite leaderboard, custom agents (BYOA via the `PokerAgent`
ABC), additional game types (Omaha, Short Deck), tournament mode with increasing
blinds, and multi-human support — the architecture is built to support all of
these.

## License

MIT
