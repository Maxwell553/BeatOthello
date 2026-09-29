"""Shallower positional alpha-beta used as evaluation *opponents*.

The packaged unbeatable bot lives in ``othello.engine`` and uses a stronger
eval, iterative deepening, and a larger exact-endgame window.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

_W = (
    100, -20, 10, 5, 5, 10, -20, 100,
    -20, -50, -2, -2, -2, -2, -50, -20,
    10, -2, -1, 0, 0, -1, -2, 10,
    5, -2, 0, -1, -1, 0, -2, 5,
    5, -2, 0, -1, -1, 0, -2, 5,
    10, -2, -1, 0, 0, -1, -2, 10,
    -20, -50, -2, -2, -2, -2, -50, -20,
    100, -20, 10, 5, 5, 10, -20, 100,
)

# Stronger midgame weights used only by hard (adaptive) proxies
_W_STRONG = (
    120, -20, 20, 5, 5, 20, -20, 120,
    -20, -40, -5, -5, -5, -5, -40, -20,
    20, -5, 15, 3, 3, 15, -5, 20,
    5, -5, 3, 3, 3, 3, -5, 5,
    5, -5, 3, 3, 3, 3, -5, 5,
    20, -5, 15, 3, 3, 15, -5, 20,
    -20, -40, -5, -5, -5, -5, -40, -20,
    120, -20, 20, 5, 5, 20, -20, 120,
)

_USE_STRONG_EVAL = False


CORNERS = (0, 7, 56, 63)
X_SQUARES = (9, 14, 49, 54)
C_SQUARES = (1, 6, 8, 15, 48, 55, 57, 62)

NOT_A = 0xFEFEFEFEFEFEFEFE
NOT_H = 0x7F7F7F7F7F7F7F7F
PASS = 64
DIRS = (
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1),
)

# key (own, opp, depth_bucket) -> (depth, flag, value, best_move)
# flag: 0 exact, -1 upperbound, 1 lowerbound
_TT: Dict[Tuple[int, int, int], Tuple[int, int, float, Optional[int]]] = {}
_TT_MAX = 1_500_000


def _shift(bits: int, dr: int, dc: int) -> int:
    if dr == -1 and dc == 0:
        return bits >> 8
    if dr == 1 and dc == 0:
        return (bits << 8) & 0xFFFFFFFFFFFFFFFF
    if dr == 0 and dc == -1:
        return (bits & NOT_A) >> 1
    if dr == 0 and dc == 1:
        return (bits & NOT_H) << 1
    if dr == -1 and dc == -1:
        return (bits & NOT_A) >> 9
    if dr == -1 and dc == 1:
        return (bits & NOT_H) >> 7
    if dr == 1 and dc == -1:
        return ((bits & NOT_A) << 7) & 0xFFFFFFFFFFFFFFFF
    if dr == 1 and dc == 1:
        return ((bits & NOT_H) << 9) & 0xFFFFFFFFFFFFFFFF
    raise ValueError((dr, dc))


def board_to_bits(board) -> Tuple[int, int]:
    black = white = 0
    flat = board.reshape(-1)
    for i, v in enumerate(flat):
        if v == 1:
            black |= 1 << i
        elif v == -1:
            white |= 1 << i
    return black, white


def legal_moves(own: int, opp: int) -> int:
    empty = ~(own | opp) & 0xFFFFFFFFFFFFFFFF
    moves = 0
    for dr, dc in DIRS:
        mask = _shift(own, dr, dc) & opp
        for _ in range(5):
            mask |= _shift(mask, dr, dc) & opp
        moves |= _shift(mask, dr, dc) & empty
    return moves


def apply_move(own: int, opp: int, sq: int) -> Tuple[int, int]:
    bit = 1 << sq
    own |= bit
    flips = 0
    for dr, dc in DIRS:
        mask = 0
        x = _shift(bit, dr, dc)
        while x and (x & opp):
            mask |= x
            x = _shift(x, dr, dc)
        if x & own:
            flips |= mask
    own ^= flips
    opp ^= flips
    return own, opp


def popcount(x: int) -> int:
    try:
        return x.bit_count()
    except AttributeError:
        return bin(x).count("1")


def clear_tt() -> None:
    _TT.clear()


def set_strong_eval(enabled: bool) -> None:
    global _USE_STRONG_EVAL
    _USE_STRONG_EVAL = enabled


def static_eval(own: int, opp: int) -> float:
    occupied = own | opp
    empties = 64 - popcount(occupied)
    if empties == 0:
        return float(1000 * (popcount(own) - popcount(opp)))
    # Near-exact: disk differential dominates deep endgame leaves
    if empties <= 8:
        return float(100 * (popcount(own) - popcount(opp))) + 2.0 * (
            popcount(legal_moves(own, opp)) - popcount(legal_moves(opp, own))
        )

    weights = _W_STRONG if _USE_STRONG_EVAL else _W
    score = 0.0
    bits_own, bits_opp = own, opp
    while bits_own:
        lsb = bits_own & -bits_own
        idx = lsb.bit_length() - 1
        score += weights[idx]
        bits_own ^= lsb
    while bits_opp:
        lsb = bits_opp & -bits_opp
        idx = lsb.bit_length() - 1
        score -= weights[idx]
        bits_opp ^= lsb

    my_moves = legal_moves(own, opp)
    opp_moves = legal_moves(opp, own)
    mob = popcount(my_moves) - popcount(opp_moves)
    if _USE_STRONG_EVAL:
        if empties > 20:
            score += 12.0 * mob
        elif empties > 10:
            score += 8.0 * mob + 2.0 * (popcount(own) - popcount(opp))
        else:
            score += 4.0 * mob + 6.0 * (popcount(own) - popcount(opp))
        for c in CORNERS:
            bit = 1 << c
            if own & bit:
                score += 25.0
            elif opp & bit:
                score -= 25.0
    else:
        score += 8.0 * mob
    return score


def _order_key(own: int, opp: int, sq: int) -> float:
    if sq in CORNERS:
        return 1e6
    no, np_ = apply_move(own, opp, sq)
    val = static_eval(no, np_)
    if sq in X_SQUARES:
        for c, x in ((0, 9), (7, 14), (56, 49), (63, 54)):
            if sq == x and not ((own | opp) & (1 << c)):
                val -= 80.0
    if sq in C_SQUARES:
        val -= 15.0
    return val


def move_list_ordered(
    own: int, opp: int, bits: int, tt_move: Optional[int] = None
) -> List[int]:
    scored = []
    b = bits
    while b:
        lsb = b & -b
        sq = lsb.bit_length() - 1
        scored.append((_order_key(own, opp, sq), sq))
        b ^= lsb
    scored.sort(reverse=True)
    moves = [sq for _, sq in scored]
    if tt_move is not None and tt_move in moves:
        moves.remove(tt_move)
        moves.insert(0, tt_move)
    return moves


def _terminal_score(own: int, opp: int) -> float:
    oc, oo = popcount(own), popcount(opp)
    if oc > oo:
        return 1e6 + (oc - oo)
    if oc < oo:
        return -1e6 - (oo - oc)
    return 0.0


def _search(own: int, opp: int, depth: int, alpha: float, beta: float) -> float:
    key = (own, opp, depth)
    tt_move = None
    entry = _TT.get(key)
    if entry is not None:
        e_depth, flag, e_val, e_move = entry
        if e_depth >= depth:
            if flag == 0:
                return e_val
            if flag < 0 and e_val <= alpha:
                return e_val
            if flag > 0 and e_val >= beta:
                return e_val
        tt_move = e_move

    moves = legal_moves(own, opp)
    if moves == 0:
        if legal_moves(opp, own) == 0:
            return _terminal_score(own, opp)
        # Pass
        if depth <= 0:
            return -static_eval(opp, own)
        return -_search(opp, own, depth - 1, -beta, -alpha)

    if depth <= 0:
        return static_eval(own, opp)

    best = float("-inf")
    best_move = None
    a0 = alpha
    for sq in move_list_ordered(own, opp, moves, tt_move):
        no, np_ = apply_move(own, opp, sq)
        val = -_search(np_, no, depth - 1, -beta, -alpha)
        if val > best:
            best = val
            best_move = sq
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break

    flag = 0
    if best <= a0:
        flag = -1
    elif best >= beta:
        flag = 1
    if len(_TT) > _TT_MAX:
        _TT.clear()
    _TT[key] = (depth, flag, best, best_move)
    return best


def best_move_root(own: int, opp: int, depth: int, adaptive: bool = True) -> int:
    moves = legal_moves(own, opp)
    if moves == 0:
        return PASS
    empties = 64 - popcount(own | opp)
    # Optional late-game deepen (use for hard proxies only — not shallow ladder)
    if adaptive:
        if empties <= 10:
            depth = max(depth, empties)
        elif empties <= 14:
            depth = max(depth, depth + 1)

    best_sq = None
    best_val = float("-inf")
    alpha, beta = float("-inf"), float("inf")
    ordered = move_list_ordered(own, opp, moves)
    for i, sq in enumerate(ordered):
        no, np_ = apply_move(own, opp, sq)
        if depth <= 1 and empties > 14:
            val = static_eval(no, np_)
        elif i == 0:
            val = -_search(np_, no, depth - 1, -beta, -alpha)
        else:
            val = -_search(np_, no, depth - 1, -alpha - 1, -alpha)
            if val > alpha:
                val = -_search(np_, no, depth - 1, -beta, -alpha)
        if val > best_val:
            best_val = val
            best_sq = sq
        if val > alpha:
            alpha = val
    assert best_sq is not None
    return best_sq
