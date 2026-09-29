"""Evaluation opponents. None of these are the packaged bot."""

from __future__ import annotations

import random

from othello.bitboard import PASS, apply_move, board_to_bits, legal_moves, popcount
from othello.env import BLACK, PASS_ACTION, OthelloEnv
from othello.fast_search import best_move_root, set_strong_eval


class RandomAgent:
    def select_action(self, env: OthelloEnv) -> int:
        return random.choice(env.legal_actions())


class GreedyAgent:
    """Myopic: maximize discs flipped this turn."""

    def select_action(self, env: OthelloEnv) -> int:
        legal = env.legal_actions()
        if legal == [PASS_ACTION]:
            return PASS_ACTION
        best, best_flip = legal[0], -1
        for a in legal:
            if a == PASS_ACTION:
                continue
            clone = env.clone()
            _, _, _, info = clone.step(a)
            if info["flipped"] > best_flip:
                best_flip = info["flipped"]
                best = a
        return best


class MobilityAgent:
    """1-ply: minimize opponent legal replies (maximize our mobility gap)."""

    def select_action(self, env: OthelloEnv) -> int:
        black, white = board_to_bits(env.board)
        own, opp = (black, white) if env.to_play == BLACK else (white, black)
        moves = legal_moves(own, opp)
        if moves == 0:
            return PASS_ACTION
        best_sq, best_key = None, -10**9
        b = moves
        while b:
            lsb = b & -b
            sq = lsb.bit_length() - 1
            no, np_ = apply_move(own, opp, sq)
            key = popcount(legal_moves(no, np_)) - popcount(legal_moves(np_, no))
            if key > best_key:
                best_key = key
                best_sq = sq
            b ^= lsb
        return PASS_ACTION if best_sq is None else best_sq


class DepthSearchAgent:
    """Shallower positional alpha-beta (old baseline search)."""

    def __init__(self, depth: int = 4, adaptive: bool = False) -> None:
        self.depth = depth
        self.adaptive = adaptive

    def select_action(self, env: OthelloEnv) -> int:
        black, white = board_to_bits(env.board)
        own, opp = (black, white) if env.to_play == BLACK else (white, black)
        set_strong_eval(self.adaptive)
        try:
            move = best_move_root(own, opp, self.depth, adaptive=self.adaptive)
        finally:
            set_strong_eval(False)
        return PASS_ACTION if move == PASS else move
