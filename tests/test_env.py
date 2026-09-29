"""Quick correctness checks for the Othello rules engine."""

from __future__ import annotations

from othello.env import BLACK, PASS_ACTION, WHITE, OthelloEnv, rc_to_idx


def test_initial_position():
    env = OthelloEnv()
    assert env.board[3, 3] == WHITE and env.board[4, 4] == WHITE
    assert env.board[3, 4] == BLACK and env.board[4, 3] == BLACK
    assert env.to_play == BLACK
    legal = set(env.legal_actions())
    # Classic opening: d3, c4, f5, e6 in 0-index: (2,3),(3,2),(4,5),(5,4)
    expected = {rc_to_idx(2, 3), rc_to_idx(3, 2), rc_to_idx(4, 5), rc_to_idx(5, 4)}
    assert legal == expected, legal


def test_flip_and_pass():
    env = OthelloEnv()
    env.step(rc_to_idx(2, 3))  # Black d3-style
    assert env.board[3, 3] == BLACK  # flipped center
    assert env.to_play == WHITE


def test_full_random_game_terminates():
    import random

    env = OthelloEnv()
    steps = 0
    while not env.done:
        a = random.choice(env.legal_actions())
        env.step(a)
        steps += 1
        assert steps < 200
    assert env.winner in (BLACK, WHITE, 0)
    b, w = env.score()
    assert b + w <= 64


def test_illegal_raises():
    env = OthelloEnv()
    try:
        env.step(0)  # corner usually illegal at start
        assert False, "should have raised"
    except ValueError:
        pass


if __name__ == "__main__":
    test_initial_position()
    test_flip_and_pass()
    test_full_random_game_terminates()
    test_illegal_raises()
    print("All env tests passed.")
