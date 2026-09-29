"""Original Othello evaluation: patterns + linear features + a tiny net.

Weights are ours (heuristic prior, then self-play SGD). Nothing is loaded
from Edax ``eval.dat`` or any third-party engine.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from othello.bitboard import (
    COLS,
    CORNER_MASK,
    MASK,
    ROWS,
    legal_moves,
    popcount,
    potential_mobility,
)

EDGE_MASK = ROWS[0] | ROWS[7] | COLS[0] | COLS[7]

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = ROOT / "data" / "eval_weights.npz"

N_PHASES = 4
EDGE_N = 3**8  # 6561
CORNER_N = 3**6  # 729
N_LIN = 9
NN_IN = 16
NN_H = 16

POW3 = (1, 3, 9, 27, 81, 243, 729, 2187)

# Edges oriented so index 0 and 7 are corners.
EDGES = (
    (0, 1, 2, 3, 4, 5, 6, 7),
    (7, 15, 23, 31, 39, 47, 55, 63),
    (63, 62, 61, 60, 59, 58, 57, 56),
    (56, 48, 40, 32, 24, 16, 8, 0),
)
# Corner, two C, X, next-C on each wing.
CORNERS6 = (
    (0, 1, 2, 8, 9, 16),
    (7, 6, 5, 15, 14, 23),
    (56, 57, 58, 48, 49, 40),
    (63, 62, 61, 55, 54, 47),
)

# Public-domain-style square priors (not Edax tables).
_POS_OPEN = np.array(
    [
        120, -25, 22, 6, 6, 22, -25, 120,
        -25, -45, -6, -2, -2, -6, -45, -25,
        22, -6, 16, 4, 4, 16, -6, 22,
        6, -2, 4, 2, 2, 4, -2, 6,
        6, -2, 4, 2, 2, 4, -2, 6,
        22, -6, 16, 4, 4, 16, -6, 22,
        -25, -45, -6, -2, -2, -6, -45, -25,
        120, -25, 22, 6, 6, 22, -25, 120,
    ],
    dtype=np.float64,
)
_POS_MID = np.array(
    [
        130, -22, 24, 8, 8, 24, -22, 130,
        -22, -42, -8, -4, -4, -8, -42, -22,
        24, -8, 18, 5, 5, 18, -8, 24,
        8, -4, 5, 3, 3, 5, -4, 8,
        8, -4, 5, 3, 3, 5, -4, 8,
        24, -8, 18, 5, 5, 18, -8, 24,
        -22, -42, -8, -4, -4, -8, -42, -22,
        130, -22, 24, 8, 8, 24, -22, 130,
    ],
    dtype=np.float64,
)
_POS_LATE = np.array(
    [
        140, -10, 18, 8, 8, 18, -10, 140,
        -10, -28, -4, 0, 0, -4, -28, -10,
        18, -4, 12, 4, 4, 12, -4, 18,
        8, 0, 4, 2, 2, 4, 0, 8,
        8, 0, 4, 2, 2, 4, 0, 8,
        18, -4, 12, 4, 4, 12, -4, 18,
        -10, -28, -4, 0, 0, -4, -28, -10,
        140, -10, 18, 8, 8, 18, -10, 140,
    ],
    dtype=np.float64,
)

# mob, pot, disc, parity, corner, xpen, cpen, stable, frontier
_LIN_PRIOR = np.array(
    [
        [14.0, 9.0, -1.5, 3.0, 28.0, 70.0, 18.0, 10.0, 7.0],
        [12.0, 7.0, 0.5, 4.0, 26.0, 65.0, 16.0, 12.0, 6.0],
        [8.0, 4.0, 3.0, 6.0, 22.0, 50.0, 12.0, 16.0, 4.0],
        [5.0, 2.0, 9.0, 8.0, 18.0, 30.0, 8.0, 20.0, 2.0],
    ],
    dtype=np.float64,
)


def phase_of(empties: int) -> int:
    if empties >= 46:
        return 0
    if empties >= 30:
        return 1
    if empties >= 18:
        return 2
    return 3


def _encode(own: int, opp: int, squares: Tuple[int, ...]) -> int:
    idx = 0
    for i, sq in enumerate(squares):
        bit = 1 << sq
        if own & bit:
            idx += POW3[i]
        elif opp & bit:
            idx += POW3[i] << 1
    return idx


def _semi_stable(own: int) -> int:
    """Discs of `own` walking inward from owned corners along each edge."""
    stable = own & CORNER_MASK
    for edge in EDGES:
        if own & (1 << edge[0]):
            for sq in edge:
                bit = 1 << sq
                if own & bit:
                    stable |= bit
                else:
                    break
        if own & (1 << edge[-1]):
            for sq in reversed(edge):
                bit = 1 << sq
                if own & bit:
                    stable |= bit
                else:
                    break
    return stable


def _pos_prior(phase: int) -> np.ndarray:
    if phase == 0:
        return _POS_OPEN
    if phase == 3:
        return _POS_LATE
    return _POS_MID


def _pattern_from_squares(squares: Tuple[int, ...], pos: np.ndarray, n: int) -> np.ndarray:
    table = np.zeros(n, dtype=np.float64)
    width = len(squares)
    for idx in range(n):
        score = 0.0
        rest = idx
        own_c0 = own_c1 = False
        empty_c0 = empty_c1 = False
        own_x = False
        for i, sq in enumerate(squares):
            d = rest % 3
            rest //= 3
            if d == 1:
                score += pos[sq]
            elif d == 2:
                score -= pos[sq]
            if i == 0:
                own_c0 = d == 1
                empty_c0 = d == 0
            if i == width - 1 and width == 8:
                own_c1 = d == 1
                empty_c1 = d == 0
            # X-square is index 4 in the 6-square corner pattern
            if width == 6 and i == 4:
                own_x = d == 1
                if d == 1 and empty_c0:
                    score -= 55.0
                elif d == 2 and empty_c0:
                    score += 55.0
        if width == 8 and own_c0 and own_c1:
            score += 18.0
        if width == 8 and empty_c0 and empty_c1:
            # unbalanced-ish: both corners empty, leave residual 0
            pass
        # Residuals only — the frozen core already has square priors.
        table[idx] = 0.0
    return table


class EvalModel:
    """Trainable evaluation. Safe to share across searchers."""

    def __init__(self) -> None:
        self.edge = np.zeros((N_PHASES, EDGE_N), dtype=np.float64)
        self.corner = np.zeros((N_PHASES, CORNER_N), dtype=np.float64)
        self.pos = np.zeros((N_PHASES, 64), dtype=np.float64)
        self.lin = np.zeros((N_PHASES, N_LIN), dtype=np.float64)
        self.nn_w1 = np.zeros((NN_IN, NN_H), dtype=np.float64)
        self.nn_b1 = np.zeros(NN_H, dtype=np.float64)
        self.nn_w2 = np.zeros(NN_H, dtype=np.float64)
        self.nn_b2 = 0.0
        self.nn_scale = 8.0
        self.residual_scale = 1.0
        self.init_heuristic()

    def init_heuristic(self) -> None:
        rng = np.random.default_rng(4)
        for p in range(N_PHASES):
            pos = _pos_prior(p)
            self.pos[p] = pos
            self.edge[p] = _pattern_from_squares(EDGES[0], pos, EDGE_N)
            self.corner[p] = _pattern_from_squares(CORNERS6[0], pos, CORNER_N)
            self.lin[p] = _LIN_PRIOR[p]
        self.nn_w1 = rng.normal(0.0, 0.04, size=(NN_IN, NN_H))
        self.nn_b1[:] = 0.0
        self.nn_w2 = rng.normal(0.0, 0.04, size=NN_H)
        self.nn_b2 = 0.0

    def load(self, path: Path | str) -> None:
        data = np.load(path, allow_pickle=False)
        self.edge = data["edge"].astype(np.float64, copy=True)
        self.corner = data["corner"].astype(np.float64, copy=True)
        self.pos = data["pos"].astype(np.float64, copy=True)
        self.lin = data["lin"].astype(np.float64, copy=True)
        self.nn_w1 = data["nn_w1"].astype(np.float64, copy=True)
        self.nn_b1 = data["nn_b1"].astype(np.float64, copy=True)
        self.nn_w2 = data["nn_w2"].astype(np.float64, copy=True)
        self.nn_b2 = float(data["nn_b2"])
        if "nn_scale" in data:
            self.nn_scale = float(data["nn_scale"])

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            edge=self.edge,
            corner=self.corner,
            pos=self.pos,
            lin=self.lin,
            nn_w1=self.nn_w1,
            nn_b1=self.nn_b1,
            nn_w2=self.nn_w2,
            nn_b2=np.array(self.nn_b2),
            nn_scale=np.array(self.nn_scale),
        )

    def _features(self, own: int, opp: int):
        occupied = own | opp
        empties = 64 - popcount(occupied)
        empty = ~occupied & MASK
        ph = phase_of(empties)
        my_m = legal_moves(own, opp)
        opp_m = legal_moves(opp, own)
        my_mob = popcount(my_m)
        opp_mob = popcount(opp_m)
        mob = my_mob - opp_mob
        pot = potential_mobility(opp, empty) - potential_mobility(own, empty)
        disc = popcount(own) - popcount(opp)
        corners = popcount(own & CORNER_MASK) - popcount(opp & CORNER_MASK)
        xpen = 0.0
        cpen = 0.0
        for c, x, cs in (
            (0, 9, (1, 8)),
            (7, 14, (6, 15)),
            (56, 49, (48, 57)),
            (63, 54, (55, 62)),
        ):
            if occupied & (1 << c):
                continue
            if own & (1 << x):
                xpen -= 1.0
            elif opp & (1 << x):
                xpen += 1.0
            for s in cs:
                if own & (1 << s):
                    cpen -= 1.0
                elif opp & (1 << s):
                    cpen += 1.0
        stable = popcount(_semi_stable(own)) - popcount(_semi_stable(opp))
        edge_d = popcount(own & EDGE_MASK) - popcount(opp & EDGE_MASK)
        feats = np.array(
            [mob, pot, disc, float(empties & 1), corners, xpen, cpen, stable, edge_d],
            dtype=np.float64,
        )
        edges = [_encode(own, opp, e) for e in EDGES]
        corners6 = [_encode(own, opp, c) for c in CORNERS6]
        return ph, feats, edges, corners6, my_mob, opp_mob

    def _nn_in(self, feats: np.ndarray, empties: int, my_mob: float, opp_mob: float) -> np.ndarray:
        return np.array(
            [
                feats[0] / 10.0,
                feats[1] / 10.0,
                feats[2] / 20.0,
                feats[3],
                feats[4] / 4.0,
                feats[5] / 4.0,
                feats[6] / 8.0,
                feats[7] / 10.0,
                feats[8] / 10.0,
                empties / 60.0,
                my_mob / 16.0,
                opp_mob / 16.0,
                max(feats[4], 0.0) / 4.0,
                max(-feats[4], 0.0) / 4.0,
                (32.0 + feats[2]) / 64.0,
                (empties & 1),
            ],
            dtype=np.float64,
        )

    def _nn_forward(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        h = np.tanh(x @ self.nn_w1 + self.nn_b1)
        y = float(h @ self.nn_w2 + self.nn_b2)
        return y, h

    def evaluate(self, own: int, opp: int) -> float:
        occupied = own | opp
        empties = 64 - popcount(occupied)
        if empties == 0:
            return float(1000 * (popcount(own) - popcount(opp)))
        my_m = legal_moves(own, opp)
        opp_m = legal_moves(opp, own)
        mob = popcount(my_m) - popcount(opp_m)
        if empties <= 8:
            return float(100 * (popcount(own) - popcount(opp))) + 2.5 * mob

        ph = phase_of(empties)
        pos = _pos_prior(ph)
        score = 0.0
        bits = own
        while bits:
            lsb = bits & -bits
            score += float(pos[lsb.bit_length() - 1])
            bits ^= lsb
        bits = opp
        while bits:
            lsb = bits & -bits
            score -= float(pos[lsb.bit_length() - 1])
            bits ^= lsb

        empty = ~occupied & MASK
        pot = potential_mobility(opp, empty) - potential_mobility(own, empty)
        disc = popcount(own) - popcount(opp)
        corners = popcount(own & CORNER_MASK) - popcount(opp & CORNER_MASK)
        xpen = 0.0
        cpen = 0.0
        for c, x, cs in (
            (0, 9, (1, 8)),
            (7, 14, (6, 15)),
            (56, 49, (48, 57)),
            (63, 54, (55, 62)),
        ):
            if occupied & (1 << c):
                continue
            if own & (1 << x):
                xpen -= 1.0
            elif opp & (1 << x):
                xpen += 1.0
            for s in cs:
                if own & (1 << s):
                    cpen -= 1.0
                elif opp & (1 << s):
                    cpen += 1.0
        stable = popcount(_semi_stable(own)) - popcount(_semi_stable(opp))
        edge_d = popcount(own & EDGE_MASK) - popcount(opp & EDGE_MASK)
        lin = _LIN_PRIOR[ph]
        score += (
            lin[0] * mob
            + lin[1] * pot
            + lin[2] * disc
            + lin[3] * float(empties & 1)
            + lin[4] * corners
            + lin[5] * xpen
            + lin[6] * cpen
            + lin[7] * stable
            + lin[8] * edge_d
        )
        # Midgame pattern residual only (skip opening; late game is disc-heavy).
        if 16 < empties < 48:
            ew = self.edge[ph]
            cw = self.corner[ph]
            for e in EDGES:
                score += self.residual_scale * float(ew[_encode(own, opp, e)])
            for c6 in CORNERS6:
                score += self.residual_scale * float(cw[_encode(own, opp, c6)])
        if empties & 1:
            score += 4.0
        return float(score)

    def train_step(self, own: int, opp: int, disc_diff: float, lr: float) -> float:
        """Tanh-normalized SGD on residuals only. `disc_diff` is final own-opp discs."""
        occupied = own | opp
        empties = 64 - popcount(occupied)
        if empties <= 8:
            return 0.0
        ph, feats, edges, corners6, my_mob, opp_mob = self._features(own, opp)
        pred = self.evaluate(own, opp)
        pred_n = math.tanh(pred / 220.0)
        targ_n = math.tanh(disc_diff / 16.0)
        err_n = pred_n - targ_n
        # d pred_n / d pred = (1 - pred_n^2) / 220
        g = lr * err_n * (1.0 - pred_n * pred_n)
        for idx in edges:
            self.edge[ph, idx] -= g
        for idx in corners6:
            self.corner[ph, idx] -= g
        x = self._nn_in(feats, empties, my_mob, opp_mob)
        nn, h = self._nn_forward(x)
        d_nn = g * self.nn_scale
        self.nn_w2 -= d_nn * h
        self.nn_b2 -= d_nn
        d_h = d_nn * self.nn_w2 * (1.0 - h * h)
        self.nn_w1 -= np.outer(x, d_h)
        self.nn_b1 -= d_h
        np.clip(self.edge[ph], -40.0, 40.0, out=self.edge[ph])
        np.clip(self.corner[ph], -40.0, 40.0, out=self.corner[ph])
        np.clip(self.nn_w1, -1.5, 1.5, out=self.nn_w1)
        np.clip(self.nn_w2, -1.5, 1.5, out=self.nn_w2)
        return float(err_n)


_MODEL: Optional[EvalModel] = None


def default_eval() -> EvalModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = EvalModel()
        if WEIGHTS_PATH.exists():
            _MODEL.load(WEIGHTS_PATH)
    return _MODEL


def reload_eval(path: Optional[Path] = None) -> EvalModel:
    global _MODEL
    _MODEL = EvalModel()
    p = path or WEIGHTS_PATH
    if p.exists():
        _MODEL.load(p)
    return _MODEL


def evaluate(own: int, opp: int) -> float:
    return default_eval().evaluate(own, opp)


def outcome_target(own_final: int, opp_final: int) -> float:
    return float(own_final - opp_final)
