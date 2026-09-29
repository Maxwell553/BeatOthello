"""Bitboard Othello primitives shared by the search engine and baselines."""

from __future__ import annotations

from typing import List, Tuple

MASK = 0xFFFFFFFFFFFFFFFF
NOT_A = 0xFEFEFEFEFEFEFEFE
NOT_H = 0x7F7F7F7F7F7F7F7F
PASS = 64

DIRS = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)

CORNERS = (0, 7, 56, 63)
CORNER_MASK = (1 << 0) | (1 << 7) | (1 << 56) | (1 << 63)
X_SQUARES = (9, 14, 49, 54)
C_SQUARES = (1, 6, 8, 15, 48, 55, 57, 62)

# Adjacent X/C for each corner.
CORNER_X = {0: 9, 7: 14, 56: 49, 63: 54}
CORNER_CS = {0: (1, 8), 7: (6, 15), 56: (48, 57), 63: (55, 62)}

ROWS = tuple(0xFF << (8 * r) for r in range(8))
COLS = tuple(0x0101010101010101 << c for c in range(8))


def popcount(x: int) -> int:
    x &= MASK
    x = x - ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
    return ((x * 0x0101010101010101) & MASK) >> 56


def _shift(bits: int, dr: int, dc: int) -> int:
    if dr == -1 and dc == 0:
        return bits >> 8
    if dr == 1 and dc == 0:
        return (bits << 8) & MASK
    if dr == 0 and dc == -1:
        return (bits & NOT_A) >> 1
    if dr == 0 and dc == 1:
        return (bits & NOT_H) << 1
    if dr == -1 and dc == -1:
        return (bits & NOT_A) >> 9
    if dr == -1 and dc == 1:
        return (bits & NOT_H) >> 7
    if dr == 1 and dc == -1:
        return ((bits & NOT_A) << 7) & MASK
    if dr == 1 and dc == 1:
        return ((bits & NOT_H) << 9) & MASK
    raise ValueError((dr, dc))


NEIGHBOR_SHIFTS = DIRS


def neighbor_mask(bits: int) -> int:
    acc = 0
    for dr, dc in NEIGHBOR_SHIFTS:
        acc |= _shift(bits, dr, dc)
    return acc & MASK


def board_to_bits(board) -> Tuple[int, int]:
    black = white = 0
    flat = board.reshape(-1)
    for i, v in enumerate(flat):
        if v == 1:
            black |= 1 << i
        elif v == -1:
            white |= 1 << i
    return black, white


def bits_to_squares(bits: int) -> List[int]:
    out: List[int] = []
    b = bits
    while b:
        lsb = b & -b
        out.append(lsb.bit_length() - 1)
        b ^= lsb
    return out


def legal_moves(own: int, opp: int) -> int:
    empty = ~(own | opp) & MASK
    moves = 0
    for dr, dc in DIRS:
        mask = _shift(own, dr, dc) & opp
        mask |= _shift(mask, dr, dc) & opp
        mask |= _shift(mask, dr, dc) & opp
        mask |= _shift(mask, dr, dc) & opp
        mask |= _shift(mask, dr, dc) & opp
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


def potential_mobility(player: int, empty: int) -> int:
    """Empty squares adjacent to `player` (frontier)."""
    return popcount(neighbor_mask(player) & empty)
