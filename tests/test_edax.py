"""Edax wrapper helpers and a live first-move check."""

from __future__ import annotations

from othello.edax_agent import board_to_setboard, edax_to_idx
from othello.env import BLACK, OthelloEnv, rc_to_idx


def test_setboard_start():
    env = OthelloEnv()
    s = board_to_setboard(env)
    body, side = s.rsplit(" ", 1)
    assert len(body) == 64
    assert side == "X"
    assert body[27] == "O" and body[28] == "X"
    assert body[35] == "X" and body[36] == "O"


def test_edax_coords():
    assert edax_to_idx("d3") == rc_to_idx(2, 3)
    assert edax_to_idx("F5") == rc_to_idx(4, 5)
    assert edax_to_idx("pass") == 64


def test_edax_first_move_legal():
    from othello.edax_agent import EdaxAgent

    env = OthelloEnv()
    bot = EdaxAgent(level=8, n_tasks=1)
    try:
        action = bot.select_action(env)
        assert action in env.legal_actions()
        assert env.to_play == BLACK
    finally:
        bot.close()


if __name__ == "__main__":
    test_setboard_start()
    test_edax_coords()
    test_edax_first_move_legal()
    print("edax tests passed")
