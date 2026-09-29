"""Original Othello engine: iterative-deepening PVS + trained eval + exact endgame.

Search, evaluation, and weights all live in this repo. Midgame uses the
pattern / linear / tiny-net eval in ``othello.eval``; when few empties remain
it switches to exact disc-difference search.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

from othello.bitboard import (
    CORNER_X,
    CORNERS,
    C_SQUARES,
    PASS,
    X_SQUARES,
    apply_move,
    bits_to_squares,
    legal_moves,
    popcount,
)
from othello.eval import evaluate as _pattern_evaluate

# TT flag: 0 exact, -1 upper (fail low), 1 lower (fail high)
_EXACT, _UPPER, _LOWER = 0, -1, 1
_TT_MAX = 2_000_000

# Opening replies keyed by (own, opp) after converting to side-to-move.
# Filled lazily from first-move theory: Black d3 (sq 19).
OPENING_D3 = 19  # (2, 3)


def _evaluate(own: int, opp: int) -> float:
    return _pattern_evaluate(own, opp)


class SearchTimeout(Exception):
    pass


class OthelloEngine:
    """Stateful searcher. Keep one instance per game so the TT persists."""

    def __init__(
        self,
        midgame_depth: int = 9,
        exact_empties: int = 12,
        time_limit: Optional[float] = 1.4,
    ) -> None:
        self.midgame_depth = midgame_depth
        self.exact_empties = exact_empties
        self.time_limit = time_limit
        self.tt: dict = {}
        self.killers = [[-1, -1] for _ in range(80)]
        self.history = [0] * 64
        self.nodes = 0
        self.deadline = 0.0
        self.check_timeout = False
        self.last_depth = 0
        self.last_value = 0.0
        self.last_nodes = 0
        self.last_ms = 0.0
        self.last_exact = False

    def reset_game(self) -> None:
        """Clear killers/history between games; keep TT (positions still valid)."""
        self.killers = [[-1, -1] for _ in range(80)]
        self.history = [0] * 64

    def _maybe_timeout(self) -> None:
        if self.check_timeout and (self.nodes & 4095) == 0:
            if time.perf_counter() >= self.deadline:
                raise SearchTimeout()

    def search_move(self, own: int, opp: int) -> int:
        t0 = time.perf_counter()
        self.nodes = 0
        self.last_exact = False
        moves = legal_moves(own, opp)
        if moves == 0:
            self.last_depth = 0
            self.last_value = 0.0
            self.last_nodes = 0
            self.last_ms = 0.0
            return PASS
        squares = bits_to_squares(moves)
        if len(squares) == 1:
            self.last_depth = 0
            self.last_value = 0.0
            self.last_nodes = 1
            self.last_ms = (time.perf_counter() - t0) * 1000
            return squares[0]

        empties = 64 - popcount(own | opp)
        if empties <= self.exact_empties:
            move, val = self._exact_root(own, opp, squares)
            self.last_exact = True
            self.last_depth = empties
            self.last_value = float(val)
            self.last_nodes = self.nodes
            self.last_ms = (time.perf_counter() - t0) * 1000
            return move

        depth_cap = self.midgame_depth
        if empties <= 20:
            depth_cap = max(depth_cap, min(empties, self.midgame_depth + 2))
        if empties >= 56:
            depth_cap = max(depth_cap, self.midgame_depth + 2)

        self.check_timeout = False
        self.deadline = (
            time.perf_counter() + self.time_limit if self.time_limit else 0.0
        )

        best_move = squares[0]
        best_val = float("-inf")
        completed_depth = 0
        min_depth = min(depth_cap, 6)
        window = 48.0
        try:
            for depth in range(1, depth_cap + 1):
                # Always finish a usable minimum depth; only then honor the clock.
                self.check_timeout = bool(self.time_limit) and depth > min_depth
                if depth <= 2 or best_val == float("-inf"):
                    move, val = self._pvs_root(
                        own, opp, squares, depth, best_move, float("-inf"), float("inf")
                    )
                else:
                    alpha, beta = best_val - window, best_val + window
                    move, val = self._pvs_root(
                        own, opp, squares, depth, best_move, alpha, beta
                    )
                    if val <= alpha or val >= beta:
                        move, val = self._pvs_root(
                            own, opp, squares, depth, best_move, float("-inf"), float("inf")
                        )
                best_move, best_val = move, val
                completed_depth = depth
        except SearchTimeout:
            pass

        self.last_depth = completed_depth
        self.last_value = best_val
        self.last_nodes = self.nodes
        self.last_ms = (time.perf_counter() - t0) * 1000
        self.check_timeout = False
        return best_move

    # ----- evaluation -------------------------------------------------
    def evaluate(self, own: int, opp: int) -> float:
        return _evaluate(own, opp)

    # ----- midgame PVS ------------------------------------------------
    def _pvs_root(
        self,
        own: int,
        opp: int,
        squares: List[int],
        depth: int,
        hint: int,
        alpha: float = float("-inf"),
        beta: float = float("inf"),
    ) -> Tuple[int, float]:
        ordered = self._order_moves(own, opp, squares, ply=0, tt_move=hint)
        best_move = ordered[0]
        best_val = float("-inf")
        first = True
        for sq in ordered:
            no, np_ = apply_move(own, opp, sq)
            if first:
                val = -self._pvs(np_, no, depth - 1, -beta, -alpha, 1, False)
                first = False
            else:
                val = -self._pvs(np_, no, depth - 1, -alpha - 1e-6, -alpha, 1, False)
                if val > alpha:
                    val = -self._pvs(np_, no, depth - 1, -beta, -alpha, 1, False)
            if val > best_val:
                best_val = val
                best_move = sq
            if val > alpha:
                alpha = val
            if alpha >= beta:
                break
        self._store_tt(own, opp, depth, False, best_val, _EXACT, best_move)
        return best_move, best_val

    def _pvs(
        self,
        own: int,
        opp: int,
        depth: int,
        alpha: float,
        beta: float,
        ply: int,
        passed: bool,
    ) -> float:
        self.nodes += 1
        self._maybe_timeout()

        occupied = own | opp
        empties = 64 - popcount(occupied)
        if empties == 0:
            return float(1000 * (popcount(own) - popcount(opp)))

        # Drop into exact endgame from inside the tree when the window is small.
        if empties <= min(8, self.exact_empties) and depth >= empties:
            return float(self._exact(own, opp, -64, 64, passed))

        tt_move = -1
        hit = self._probe_tt(own, opp, depth, False)
        if hit is not None:
            e_depth, flag, e_val, e_move = hit
            tt_move = e_move if e_move is not None else -1
            if e_depth >= depth:
                if flag == _EXACT:
                    return e_val
                if flag == _UPPER and e_val <= alpha:
                    return e_val
                if flag == _LOWER and e_val >= beta:
                    return e_val

        moves_bits = legal_moves(own, opp)
        if moves_bits == 0:
            if passed:
                return _terminal(own, opp)
            return -self._pvs(opp, own, depth, -beta, -alpha, ply + 1, True)

        if depth <= 0:
            return self.evaluate(own, opp)

        squares = bits_to_squares(moves_bits)
        ordered = self._order_moves(own, opp, squares, ply, tt_move if tt_move >= 0 else None)

        best = float("-inf")
        best_move = ordered[0]
        a0 = alpha
        first = True
        for sq in ordered:
            no, np_ = apply_move(own, opp, sq)
            if first:
                val = -self._pvs(np_, no, depth - 1, -beta, -alpha, ply + 1, False)
                first = False
            else:
                val = -self._pvs(np_, no, depth - 1, -alpha - 1e-6, -alpha, ply + 1, False)
                if val > alpha and val < beta:
                    val = -self._pvs(np_, no, depth - 1, -beta, -alpha, ply + 1, False)
            if val > best:
                best = val
                best_move = sq
            if best > alpha:
                alpha = best
            if alpha >= beta:
                self._record_cut(sq, ply)
                break

        flag = _EXACT
        if best <= a0:
            flag = _UPPER
        elif best >= beta:
            flag = _LOWER
        self._store_tt(own, opp, depth, False, best, flag, best_move)
        return best

    def _order_moves(
        self,
        own: int,
        opp: int,
        squares: List[int],
        ply: int,
        tt_move: Optional[int],
    ) -> List[int]:
        k1, k2 = self.killers[ply]
        scored = []
        occupied = own | opp
        for sq in squares:
            key = 0.0
            if tt_move is not None and sq == tt_move:
                key = 1e9
            elif sq in CORNERS:
                key = 1e8
            elif sq == k1:
                key = 1e7
            elif sq == k2:
                key = 1e6
            else:
                key = float(self.history[sq])
                if sq in X_SQUARES:
                    for c, x in CORNER_X.items():
                        if sq == x and not (occupied & (1 << c)):
                            key -= 800.0
                if sq in C_SQUARES:
                    key -= 40.0
            scored.append((key, sq))
        scored.sort(reverse=True)
        return [sq for _, sq in scored]

    def _record_cut(self, sq: int, ply: int) -> None:
        k = self.killers[ply]
        if k[0] != sq:
            k[1] = k[0]
            k[0] = sq
        self.history[sq] += 2

    # ----- exact endgame (disc difference) ----------------------------
    def _exact_root(self, own: int, opp: int, squares: List[int]) -> Tuple[int, int]:
        ordered = self._exact_order(own, opp, squares)
        best_move = ordered[0]
        best_val = -64
        alpha, beta = -64, 64
        for sq in ordered:
            no, np_ = apply_move(own, opp, sq)
            val = -self._exact(np_, no, -beta, -alpha, False)
            if val > best_val:
                best_val = val
                best_move = sq
            if val > alpha:
                alpha = val
        self._store_tt(own, opp, 64 - popcount(own | opp), True, float(best_val), _EXACT, best_move)
        return best_move, best_val

    def _exact(self, own: int, opp: int, alpha: int, beta: int, passed: bool) -> int:
        self.nodes += 1
        self._maybe_timeout()
        occupied = own | opp
        empties = 64 - popcount(occupied)
        if empties == 0:
            return popcount(own) - popcount(opp)

        hit = self._probe_tt(own, opp, empties, True)
        tt_move = -1
        if hit is not None:
            _d, flag, e_val, e_move = hit
            tt_move = e_move if e_move is not None else -1
            iv = int(e_val)
            if flag == _EXACT:
                return iv
            if flag == _UPPER and iv <= alpha:
                return iv
            if flag == _LOWER and iv >= beta:
                return iv

        moves_bits = legal_moves(own, opp)
        if moves_bits == 0:
            if passed:
                return popcount(own) - popcount(opp)
            return -self._exact(opp, own, -beta, -alpha, True)

        squares = bits_to_squares(moves_bits)
        if tt_move >= 0 and tt_move in squares:
            squares.remove(tt_move)
            squares.insert(0, tt_move)
        else:
            squares = self._exact_order(own, opp, squares)

        best = -64
        best_move = squares[0]
        a0 = alpha
        for sq in squares:
            no, np_ = apply_move(own, opp, sq)
            val = -self._exact(np_, no, -beta, -alpha, False)
            if val > best:
                best = val
                best_move = sq
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break

        flag = _EXACT
        if best <= a0:
            flag = _UPPER
        elif best >= beta:
            flag = _LOWER
        self._store_tt(own, opp, empties, True, float(best), flag, best_move)
        return best

    def _exact_order(self, own: int, opp: int, squares: List[int]) -> List[int]:
        scored = []
        for sq in squares:
            no, np_ = apply_move(own, opp, sq)
            opp_m = popcount(legal_moves(np_, no))
            key = -10 * opp_m
            if sq in CORNERS:
                key += 50
            scored.append((key, sq))
        scored.sort(reverse=True)
        return [sq for _, sq in scored]

    # ----- transposition table ----------------------------------------
    def _probe_tt(self, own: int, opp: int, depth: int, exact: bool):
        return self.tt.get((own, opp, exact))

    def _store_tt(
        self,
        own: int,
        opp: int,
        depth: int,
        exact: bool,
        val: float,
        flag: int,
        move: int,
    ) -> None:
        key = (own, opp, exact)
        prev = self.tt.get(key)
        if prev is not None and prev[0] > depth:
            return
        if len(self.tt) > _TT_MAX:
            self.tt.clear()
        self.tt[key] = (depth, flag, val, move)


def _terminal(own: int, opp: int) -> float:
    oc, oo = popcount(own), popcount(opp)
    if oc > oo:
        return 1000.0 + (oc - oo)
    if oc < oo:
        return -1000.0 - (oo - oc)
    return 0.0
