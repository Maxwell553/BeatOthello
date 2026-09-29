#!/usr/bin/env python3
"""Train original eval weights from self-play and mixed-policy games.

Does not call Edax. Writes data/eval_weights.npz.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from othello.bitboard import (
    PASS,
    apply_move,
    bits_to_squares,
    legal_moves,
    popcount,
)
from othello.engine import OthelloEngine
from othello.eval import WEIGHTS_PATH, EvalModel, reload_eval
from othello.fast_search import best_move_root, set_strong_eval

START_BLACK = (1 << 28) | (1 << 35)
START_WHITE = (1 << 27) | (1 << 36)


def _legal_or_pass(own: int, opp: int) -> list[int]:
    bits = legal_moves(own, opp)
    if bits == 0:
        return [PASS]
    return bits_to_squares(bits)


def play_game(policy, rng: random.Random, noise: float, random_plies: int) -> list[tuple[int, int, int, int]]:
    """Return (own, opp, black, white) snapshots for every side-to-move, plus finals.

    Last tuple uses own=opp=0 as a sentinel? Better return positions + final counts.
    """
    black, white = START_BLACK, START_WHITE
    black_to_move = True
    passed = False
    ply = 0
    positions: list[tuple[int, int, bool]] = []
    while True:
        own, opp = (black, white) if black_to_move else (white, black)
        moves = _legal_or_pass(own, opp)
        if moves == [PASS]:
            if passed:
                break
            passed = True
            black_to_move = not black_to_move
            ply += 1
            continue
        passed = False
        empties = 64 - popcount(black | white)
        positions.append((own, opp, black_to_move))
        if ply < random_plies or (empties > 12 and rng.random() < noise):
            move = rng.choice(moves)
        else:
            move = policy(own, opp, empties)
            if move not in moves:
                move = moves[0]
        no, np_ = apply_move(own, opp, move)
        if black_to_move:
            black, white = no, np_
        else:
            white, black = no, np_
        black_to_move = not black_to_move
        ply += 1
        if ply > 80:
            break
    return positions, popcount(black), popcount(white)


def shallow_policy(own: int, opp: int, empties: int) -> int:
    depth = 2
    if empties <= 10:
        depth = empties
    elif empties <= 16:
        depth = 4
    set_strong_eval(True)
    try:
        return best_move_root(own, opp, depth, adaptive=False)
    finally:
        set_strong_eval(False)


def engine_policy(engine: OthelloEngine):
    def policy(own: int, opp: int, empties: int) -> int:
        return engine.search_move(own, opp)

    return policy


def collect(
    n_games: int,
    policy,
    rng: random.Random,
    noise: float,
    random_plies: int,
) -> list[tuple[int, int, float]]:
    samples: list[tuple[int, int, float]] = []
    for i in range(n_games):
        positions, b, w = play_game(policy, rng, noise, random_plies)
        for own, opp, black_to_move in positions:
            empties = 64 - popcount(own | opp)
            if empties <= 8:
                continue
            target = float(b - w) if black_to_move else float(w - b)
            samples.append((own, opp, target))
        if (i + 1) % 25 == 0:
            print(f"  games {i + 1}/{n_games}  samples={len(samples)}", flush=True)
    return samples


def sgd(model: EvalModel, samples: list[tuple[int, int, float]], epochs: int, lr: float, rng: random.Random) -> float:
    if not samples:
        return 0.0
    last = 0.0
    for ep in range(epochs):
        rng.shuffle(samples)
        total = 0.0
        n = 0
        for own, opp, target in samples:
            err = model.train_step(own, opp, target, lr)
            total += abs(err)
            n += 1
        last = total / max(n, 1)
        print(f"  epoch {ep + 1}/{epochs}  mae={last:.3f}  lr={lr}", flush=True)
        lr *= 0.72
    return last


def main() -> None:
    parser = argparse.ArgumentParser(description="Train original Othello eval")
    parser.add_argument("--fast-games", type=int, default=500)
    parser.add_argument("--engine-games", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    model = EvalModel()
    if WEIGHTS_PATH.exists():
        print(f"warming from {WEIGHTS_PATH}", flush=True)
        model.init_heuristic()
        # keep heuristic prior; do not load stale Edax-era files
    t0 = time.perf_counter()

    print(f"mixed shallow games: {args.fast_games}", flush=True)
    samples = collect(args.fast_games, shallow_policy, rng, noise=0.18, random_plies=8)
    print(f"SGD on {len(samples)} shallow samples", flush=True)
    sgd(model, samples, epochs=3, lr=0.28, rng=rng)

    print(f"engine self-play: {args.engine_games}", flush=True)
    engine = OthelloEngine(midgame_depth=4, exact_empties=10, time_limit=None)
    more = collect(args.engine_games, engine_policy(engine), rng, noise=0.12, random_plies=6)
    print(f"SGD on {len(more)} engine samples", flush=True)
    sgd(model, more, epochs=4, lr=0.18, rng=rng)

    blend = samples[::3] + more
    rng.shuffle(blend)
    sgd(model, blend[:8000], epochs=2, lr=0.08, rng=rng)

    model.save(WEIGHTS_PATH)
    reload_eval(WEIGHTS_PATH)
    print(f"wrote {WEIGHTS_PATH}  in {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
