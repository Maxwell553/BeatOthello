"""Search-based Othello agents.

The packaged bot is ``EngineAgent``: our PVS searcher + original eval.
``SearchAgent`` remains as a weaker positional baseline. Edax is optional.
"""

from __future__ import annotations

import time

from othello.bitboard import PASS, board_to_bits, popcount
from othello.env import BLACK, PASS_ACTION, OthelloEnv
from othello.fast_search import best_move_root, set_strong_eval


class SearchAgent:
    """Shallower positional alpha-beta (evaluation opponent / fallback)."""

    def __init__(
        self,
        midgame_depth: int = 8,
        exact_empties: int = 10,
        time_limit: float | None = None,
        engine=None,
        parity_adjust: bool = True,
    ) -> None:
        self.midgame_depth = midgame_depth
        self.exact_empties = exact_empties
        self.time_limit = time_limit
        self.parity_adjust = parity_adjust
        self.last_depth = 0
        self.last_exact = False
        self.last_nodes = 0
        self.last_ms = 0.0
        self.last_value = 0.0
        self.engine = self

    def new_game(self) -> None:
        return

    def _depth_for(self, env: OthelloEnv, empties: int) -> int:
        if empties <= self.exact_empties:
            return empties
        depth = self.midgame_depth
        if self.parity_adjust:
            if env.to_play == BLACK:
                if depth % 2 == 0:
                    depth += 1
            else:
                if depth % 2 == 1:
                    depth -= 1
        return depth

    def select_action(self, env: OthelloEnv) -> int:
        t0 = time.perf_counter()
        black, white = board_to_bits(env.board)
        own, opp = (black, white) if env.to_play == BLACK else (white, black)
        empties = 64 - popcount(own | opp)
        exact = empties <= self.exact_empties
        depth = self._depth_for(env, empties)
        set_strong_eval(True)
        try:
            move = best_move_root(own, opp, depth, adaptive=False)
        finally:
            set_strong_eval(False)
        self.last_depth = depth
        self.last_exact = exact
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return PASS_ACTION if move == PASS else move


class EngineAgent:
    """Packaged bot: original engine (pattern eval + PVS + exact endgame)."""

    def __init__(
        self,
        midgame_depth: int = 9,
        exact_empties: int = 12,
        time_limit: float | None = 1.4,
    ) -> None:
        from othello.engine import OthelloEngine

        self.engine = OthelloEngine(
            midgame_depth=midgame_depth,
            exact_empties=exact_empties,
            time_limit=time_limit,
        )
        self.midgame_depth = midgame_depth
        self.exact_empties = exact_empties
        self.time_limit = time_limit

    def new_game(self) -> None:
        self.engine.reset_game()

    def select_action(self, env: OthelloEnv) -> int:
        legal = env.legal_actions()
        if legal == [PASS_ACTION]:
            self.engine.last_depth = 0
            self.engine.last_exact = False
            self.engine.last_nodes = 0
            self.engine.last_ms = 0.0
            return PASS_ACTION
        black, white = board_to_bits(env.board)
        own, opp = (black, white) if env.to_play == BLACK else (white, black)
        move = self.engine.search_move(own, opp)
        if move == PASS:
            return PASS_ACTION
        if move not in legal:
            return legal[0]
        return move


def unbeatable_agent(time_limit: float | None = 1.4) -> EngineAgent:
    """Packaged bot: our engine, not Edax."""
    return EngineAgent(time_limit=time_limit)


def packaged_agent(
    midgame_depth: int = 9,
    exact_empties: int = 12,
    time_limit: float | None = 1.4,
) -> EngineAgent:
    return EngineAgent(
        midgame_depth=midgame_depth,
        exact_empties=exact_empties,
        time_limit=time_limit,
    )
