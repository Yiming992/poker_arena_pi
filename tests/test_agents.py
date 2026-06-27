import asyncio

from agents.base import AgentGameView, OpponentView
from agents.rule_based_agent import RuleBasedAgent, _preflop_tier
from game import validator
from poker.models import Action, ActionType, Card


def C(code):
    return Card.from_code(code)


def make_view(
    hole=("Ah", "Kh"),
    community=(),
    valid=("fold", "call", "raise"),
    call_amount=20,
    min_raise=40,
    max_raise=500,
    current_bet=20,
    stage="preflop",
    pot=60,
):
    return AgentGameView(
        your_name="Bot",
        your_seat=3,
        your_stack=500,
        your_hole_cards=(C(hole[0]), C(hole[1])),
        community_cards=[C(c) for c in community],
        pot=pot,
        current_bet=current_bet,
        your_current_bet=0,
        call_amount=call_amount,
        min_raise_to=min_raise,
        max_raise_to=max_raise,
        valid_actions=list(valid),
        stage=stage,
        hand_number=1,
        dealer_seat=1,
        big_blind=10,
        small_blind=5,
        opponents=[OpponentView("Foe", 5, 480, 20, 20, False, False, False)],
    )


# ---------------------------------------------------------------------- #
# Parser golden tests (recorded response variations — no live API)
# ---------------------------------------------------------------------- #
def test_parse_strict_template():
    text = "I have top pair.\nACTION: raise AMOUNT: 40"
    p = validator.parse_response(text)
    assert p.action.type == ActionType.RAISE
    assert p.action.amount == 40


def test_parse_fold_amount_zero():
    text = "This is trash. ACTION: fold AMOUNT: 0"
    p = validator.parse_response(text)
    assert p.action.type == ActionType.FOLD


def test_parse_with_dollar_and_commas():
    text = "ACTION: raise AMOUNT: $1,250"
    p = validator.parse_response(text)
    assert p.action.amount == 1250


def test_parse_fallback_no_template():
    text = "I think I'll just call here given the pot odds."
    p = validator.parse_response(text)
    assert p.action.type == ActionType.CALL


def test_parse_all_in_variants():
    for text in ["ACTION: all-in AMOUNT: 500", "ACTION: all_in AMOUNT: 500"]:
        p = validator.parse_response(text)
        assert p.action.type == ActionType.ALL_IN


def test_parse_empty_raises():
    try:
        validator.parse_response("   ")
        assert False, "expected ParseError"
    except validator.ParseError:
        pass


# ---------------------------------------------------------------------- #
# Validation golden tests
# ---------------------------------------------------------------------- #
def test_validate_raise_below_min_rejected():
    view = make_view(min_raise=40, max_raise=500)
    try:
        validator.validate(Action(ActionType.RAISE, 30), view)
        assert False
    except validator.ValidationError:
        pass


def test_validate_raise_over_max_becomes_all_in():
    view = make_view(max_raise=500)
    out = validator.validate(Action(ActionType.RAISE, 999), view)
    assert out.type == ActionType.ALL_IN
    assert out.amount == 500


def test_validate_check_when_facing_bet_rejected():
    view = make_view(valid=("fold", "call", "raise"))
    try:
        validator.validate(Action(ActionType.CHECK), view)
        assert False
    except validator.ValidationError:
        pass


def test_validate_call_with_nothing_becomes_check():
    view = make_view(valid=("check", "raise"), call_amount=0, current_bet=0)
    out = validator.validate(Action(ActionType.CALL), view)
    assert out.type == ActionType.CHECK


def test_fallback_action_prefers_check():
    view = make_view(valid=("check", "raise"), call_amount=0)
    assert validator.fallback_action(view).type == ActionType.CHECK
    view2 = make_view(valid=("fold", "call", "raise"))
    assert validator.fallback_action(view2).type == ActionType.FOLD


# ---------------------------------------------------------------------- #
# Rule-based agent
# ---------------------------------------------------------------------- #
def test_preflop_tier_premium():
    assert _preflop_tier((C("Ah"), C("As"))) == 0
    assert _preflop_tier((C("Ah"), C("Ks"))) <= 1
    assert _preflop_tier((C("2h"), C("7d"))) == 3


def test_rule_based_folds_trash_to_bet():
    agent = RuleBasedAgent("RuleBot")
    view = make_view(hole=("2h", "7d"), valid=("fold", "call", "raise"))
    action, reasoning = asyncio.run(agent.decide(view))
    assert action.type == ActionType.FOLD
    assert "Hand strength" in reasoning


def test_rule_based_raises_premium():
    agent = RuleBasedAgent("RuleBot")
    view = make_view(hole=("Ah", "As"), valid=("fold", "call", "raise"))
    action, reasoning = asyncio.run(agent.decide(view))
    assert action.type in (ActionType.RAISE, ActionType.ALL_IN)


def test_nvidia_agent_routing_chat_vs_messages():
    from agents.factory import build_agent
    from agents.nvidia_agent import _NvidiaChatAgent, _NvidiaMessagesAgent

    providers = {
        "nvidia": {
            "api_key": "nvapi-test",
            "base_url": "https://inference-api.nvidia.com/v1",
        }
    }
    chat = build_agent(
        "GPT", "nvidia", model="openai/openai/gpt-5.5", providers=providers, api="chat"
    )
    msgs = build_agent(
        "Opus",
        "nvidia",
        model="aws/anthropic/bedrock-claude-opus-4-8",
        providers=providers,
        api="messages",
    )
    assert isinstance(chat, _NvidiaChatAgent)
    assert isinstance(msgs, _NvidiaMessagesAgent)
    assert chat.agent_type == msgs.agent_type == "nvidia"
    # Messages agent must strip the trailing /v1 (SDK re-appends /v1/messages).
    assert str(msgs._client.base_url).rstrip("/").endswith("inference-api.nvidia.com")
    # Chat agent keeps the /v1 base.
    assert "/v1" in str(chat._client.base_url)


def test_nvidia_default_api_is_chat():
    from agents.factory import build_agent
    from agents.nvidia_agent import _NvidiaChatAgent

    agent = build_agent(
        "Llama", "nvidia", model="meta/llama-3.3-70b-instruct",
        providers={"nvidia": {"api_key": "nvapi-test"}},
    )
    assert isinstance(agent, _NvidiaChatAgent)


def test_rule_based_action_always_legal():
    agent = RuleBasedAgent("RuleBot")
    for hole in [("Ah", "As"), ("Kh", "Qh"), ("9c", "9d"), ("2h", "7d")]:
        for valid in [("fold", "call", "raise"), ("fold", "check", "raise"), ("fold", "check")]:
            view = make_view(
                hole=hole, valid=valid,
                call_amount=0 if "check" in valid else 20,
                current_bet=0 if "check" in valid else 20,
            )
            action, _ = asyncio.run(agent.decide(view))
            # The returned action must be among valid types.
            legal = validator.validate(action, view)
            assert legal is not None
