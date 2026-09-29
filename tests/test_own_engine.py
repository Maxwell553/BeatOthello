"""The packaged bot is our engine, not an Edax wrapper."""

from __future__ import annotations

import random

from othello.agent import EngineAgent, unbeatable_agent
from othello.bitboard import board_to_bits, legal_moves, popcount
from othello.engine import OthelloEngine
from othello.env import BLACK, PASS_ACTION, OthelloEnv
from othello.match import play_match
from othello.opponents import GreedyAgent, RandomAgent


OPENING = {19, 26, 37, 44}  # d3, c4, f5, e6


def test_packaged_is_original_engine():
    bot = unbeatable_agent(time_limit=0.2)
    assert isinstance(bot, EngineAgent)
    assert isinstance(bot.engine, OthelloEngine)
    assert not hasattr(bot, "proc")


def test_first_move_legal_opening():
    env = OthelloEnv()
    bot = EngineAgent(midgame_depth=6, exact_empties=8, time_limit=0.4)
    move = bot.select_action(env)
    assert move in env.legal_actions()
    assert move in OPENING


def test_plays_legal_full_game():
    rng = random.Random(3)
    bot = EngineAgent(midgame_depth=5, exact_empties=10, time_limit=0.25)
    env = OthelloEnv()
    while not env.done:
        if env.to_play == BLACK:
            action = bot.select_action(env)
        else:
            action = rng.choice(env.legal_actions())
        assert action in env.legal_actions()
        env.step(action)
    assert env.winner in (BLACK, -1, 0)


def test_no_edax_import_on_move():
    import sys

    # Drop any previous wrapper import so we can see if a move pulls it in.
    sys.modules.pop("othello.edax_agent", None)
    bot = EngineAgent(midgame_depth=4, exact_empties=8, time_limit=0.15)
    env = OthelloEnv()
    bot.select_action(env)
    assert "othello.edax_agent" not in sys.modules


def test_engine_beats_random_and_greedy():
    bot = EngineAgent(midgame_depth=5, exact_empties=10, time_limit=0.25)
    rnd = play_match(bot, RandomAgent(), n_games=4)
    assert rnd["losses"] == 0, rnd
    assert rnd["wins"] >= 3, rnd
    greedy = play_match(bot, GreedyAgent(), n_games=2)
    assert greedy["losses"] == 0, greedy
    assert greedy["wins"] == 2, greedy


def test_eval_is_finite():
    from othello.eval import evaluate

    env = OthelloEnv()
    black, white = board_to_bits(env.board)
    val = evaluate(black, white)
    assert val == val  # not NaN
    assert abs(val) < 1e6
    assert popcount(legal_moves(black, white)) == 4


if __name__ == "__main__":
    test_packaged_is_original_engine()
    print("packaged ok", flush=True)
    test_first_move_legal_opening()
    print("opening ok", flush=True)
    test_plays_legal_full_game()
    print("legal game ok", flush=True)
    test_no_edax_import_on_move()
    print("no edax ok", flush=True)
    test_eval_is_finite()
    print("eval ok", flush=True)
    test_engine_beats_random_and_greedy()
    print("ladder ok", flush=True)
    print("All own-engine tests passed.")
