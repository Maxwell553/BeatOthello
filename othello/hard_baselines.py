"""Harder heuristic opponents for evaluation (not the packaged bot)."""

from __future__ import annotations

from othello.env import BLACK, PASS_ACTION, OthelloEnv, WHITE
from othello.fast_search import best_move_root, board_to_bits, set_strong_eval


class Depth2PositionalAgent:
    """Alpha-beta over positional static eval.

    ``adaptive=False`` keeps the nominal depth (curriculum ladder).
    ``adaptive=True`` deepens in the endgame (hard pro-proxy).
    """

    def __init__(self, depth: int = 2, adaptive: bool = False) -> None:
        self.depth = depth
        self.adaptive = adaptive

    def select_action(self, env: OthelloEnv) -> int:
        black, white = board_to_bits(env.board)
        if env.to_play == BLACK:
            own, opp = black, white
        else:
            own, opp = white, black
        # Strong eval only for adaptive hard proxies (d5+); classic for ladder
        set_strong_eval(self.adaptive)
        try:
            move = best_move_root(own, opp, self.depth, adaptive=self.adaptive)
        finally:
            set_strong_eval(False)
        return PASS_ACTION if move == 64 else move


class Depth3PositionalAgent(Depth2PositionalAgent):
    def __init__(self, adaptive: bool = False) -> None:
        super().__init__(depth=3, adaptive=adaptive)


class Depth4PositionalAgent(Depth2PositionalAgent):
    def __init__(self, adaptive: bool = False) -> None:
        super().__init__(depth=4, adaptive=adaptive)


class Depth5PositionalAgent(Depth2PositionalAgent):
    """Stronger search proxy; endgame-adaptive by default."""

    def __init__(self, adaptive: bool = True) -> None:
        super().__init__(depth=5, adaptive=adaptive)


class Depth6PositionalAgent(Depth2PositionalAgent):
    def __init__(self, adaptive: bool = True) -> None:
        super().__init__(depth=6, adaptive=adaptive)


class ExactEndgameAgent:
    """Near-exact endgame labels for BC (search for training only).

    When empties <= ``exact_at``, searches full remaining depth with strong eval
    so leaves are disk-accurate. Outside that window, falls back to depth-5 adaptive.
    """

    def __init__(self, exact_at: int = 10, fallback_depth: int = 5) -> None:
        self.exact_at = exact_at
        self.fallback_depth = fallback_depth

    def select_action(self, env: OthelloEnv) -> int:
        black, white = board_to_bits(env.board)
        if env.to_play == BLACK:
            own, opp = black, white
        else:
            own, opp = white, black
        from othello.fast_search import popcount

        empties = 64 - popcount(own | opp)
        depth = empties if empties <= self.exact_at else self.fallback_depth
        set_strong_eval(True)
        try:
            move = best_move_root(own, opp, depth, adaptive=True)
        finally:
            set_strong_eval(False)
        return PASS_ACTION if move == 64 else move


class VariableDepthPositionalAgent:
    """Curriculum bridge: shallow midgame, deeper as empties fall.

    Closes the cliff between crushing fixed depth-4 and losing to full depth-5.
    """

    def __init__(
        self,
        min_depth: int = 3,
        max_depth: int = 5,
        adaptive: bool = False,
    ) -> None:
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.adaptive = adaptive

    def _depth_for(self, empties: int) -> int:
        # Gentler ramp: only touch depth-5 in true endgame
        if empties > 40:
            return self.min_depth
        if empties > 28:
            return min(self.max_depth, max(self.min_depth, 3))
        if empties > 16:
            return min(self.max_depth, 4)
        if empties > 10:
            return min(self.max_depth, 4)
        return self.max_depth

    def select_action(self, env: OthelloEnv) -> int:
        black, white = board_to_bits(env.board)
        if env.to_play == BLACK:
            own, opp = black, white
        else:
            own, opp = white, black
        empties = 64 - bin(own | opp).count("1")
        try:
            from othello.fast_search import popcount

            empties = 64 - popcount(own | opp)
        except Exception:
            pass
        depth = self._depth_for(empties)
        set_strong_eval(self.adaptive)
        try:
            move = best_move_root(own, opp, depth, adaptive=self.adaptive)
        finally:
            set_strong_eval(False)
        return PASS_ACTION if move == 64 else move

