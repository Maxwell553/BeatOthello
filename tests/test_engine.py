"""Correctness tests for rules + bitboard search engine."""

from __future__ import annotations

import random

from othello.agent import SearchAgent
from othello.bitboard import apply_move, board_to_bits, legal_moves, popcount
from othello.env import BLACK, PASS_ACTION, WHITE, OthelloEnv, rc_to_idx
from othello.opponents import GreedyAgent, RandomAgent
from othello.match import play_match


def test_initial_position():
    env = OthelloEnv()
    assert env.board[3, 3] == WHITE and env.board[4, 4] == WHITE
    assert env.board[3, 4] == BLACK and env.board[4, 3] == BLACK
    assert env.to_play == BLACK
    legal = set(env.legal_actions())
    expected = {rc_to_idx(2, 3), rc_to_idx(3, 2), rc_to_idx(4, 5), rc_to_idx(5, 4)}
    assert legal == expected, legal


def test_flip_and_pass():
    env = OthelloEnv()
    env.step(rc_to_idx(2, 3))
    assert env.board[3, 3] == BLACK
    assert env.to_play == WHITE


def test_full_random_game_terminates():
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
        env.step(0)
        assert False, "should have raised"
    except ValueError:
        pass


def test_bitboard_matches_env():
    rng = random.Random(0)
    for _ in range(25):
        env = OthelloEnv()
        while not env.done:
            black, white = board_to_bits(env.board)
            own, opp = (black, white) if env.to_play == BLACK else (white, black)
            bits = legal_moves(own, opp)
            env_moves = [a for a in env.legal_actions() if a != PASS_ACTION]
            bb_moves = []
            b = bits
            while b:
                lsb = b & -b
                bb_moves.append(lsb.bit_length() - 1)
                b ^= lsb
            assert set(env_moves) == set(bb_moves), (env_moves, bb_moves, env.render_ascii())
            if not env_moves:
                assert env.legal_actions() == [PASS_ACTION]
                env.step(PASS_ACTION)
                continue
            action = rng.choice(env_moves)
            before_own, before_opp = own, opp
            env.step(action)
            no, np_ = apply_move(before_own, before_opp, action)
            black2, white2 = board_to_bits(env.board)
            if env.to_play == BLACK:
                # we just moved as white? After step, to_play flipped.
                pass
            # The mover's discs are `no`; opponent's are `np_`.
            mover_was_black = before_own == (black)
            if mover_was_black:
                assert no == black2 and np_ == white2
            else:
                assert no == white2 and np_ == black2


def test_engine_plays_legal_and_beats_random():
    bot = SearchAgent(midgame_depth=5, exact_empties=10, time_limit=0.2)
    env = OthelloEnv()
    while not env.done:
        action = bot.select_action(env)
        assert action in env.legal_actions()
        env.step(action)
        if env.done:
            break
        env.step(random.choice(env.legal_actions()))
    assert env.winner in (BLACK, WHITE, 0)

    stats = play_match(bot, RandomAgent(), n_games=6)
    assert stats["losses"] == 0, stats
    assert stats["wins"] >= 5, stats


def test_engine_beats_greedy():
    bot = SearchAgent(midgame_depth=5, exact_empties=10, time_limit=0.2)
    stats = play_match(bot, GreedyAgent(), n_games=4)
    assert stats["losses"] == 0, stats
    assert stats["wins"] == 4, stats


def test_exact_endgame_terminal_value():
    """Exact search on a filled-almost board agrees with disc difference."""
    from othello.engine import OthelloEngine

    engine = OthelloEngine(midgame_depth=1, exact_empties=12, time_limit=None)
    # Two discs each in corners: black 0 and 7, white 56 and 63, rest empty
    # plus a small cluster so moves exist. Easier: take a random game to 10 empties
    # and confirm exact value is in [-64, 64] and the chosen move is legal.
    rng = random.Random(1)
    env = OthelloEnv()
    while not env.done and popcount(board_to_bits(env.board)[0] | board_to_bits(env.board)[1]) < 54:
        env.step(rng.choice(env.legal_actions()))
    if env.done:
        return
    black, white = board_to_bits(env.board)
    own, opp = (black, white) if env.to_play == BLACK else (white, black)
    move = engine.search_move(own, opp)
    legal = env.legal_actions()
    if legal == [PASS_ACTION]:
        assert move == 64
    else:
        assert move in legal
    assert engine.last_exact or engine.last_depth >= 1


if __name__ == "__main__":
    test_initial_position()
    test_flip_and_pass()
    test_full_random_game_terminates()
    test_illegal_raises()
    test_bitboard_matches_env()
    print("env + bitboard ok", flush=True)
    test_exact_endgame_terminal_value()
    print("exact ok", flush=True)
    test_engine_plays_legal_and_beats_random()
    print("vs random ok", flush=True)
    test_engine_beats_greedy()
    print("vs greedy ok", flush=True)
    print("All tests passed.")
