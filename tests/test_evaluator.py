from poker.evaluator import evaluate, hand_class, rank_percentage
from poker.models import Card


def C(code):
    return Card.from_code(code)


def test_royal_flush_beats_pair():
    royal = [C("Ah"), C("Kh")]
    pair = [C("2c"), C("2d")]
    board = [C("Qh"), C("Jh"), C("Th"), C("3s"), C("4s")]
    assert evaluate(royal, board) < evaluate(pair, board)


def test_hand_class_strings():
    royal = [C("Ah"), C("Kh")]
    board = [C("Qh"), C("Jh"), C("Th"), C("3s"), C("4s")]
    assert "Flush" in hand_class(royal, board) or "Straight" in hand_class(royal, board)


def test_rank_percentage_monotonic():
    strong = rank_percentage([C("Ah"), C("As")], [C("Ad"), C("Kc"), C("Qh")])
    weak = rank_percentage([C("2h"), C("7s")], [C("Ad"), C("Kc"), C("Qh")])
    assert strong > weak
    assert 0.0 <= weak <= 1.0 and 0.0 <= strong <= 1.0
