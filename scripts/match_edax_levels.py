#!/usr/bin/env python3
"""Our EngineAgent vs every standard Edax playing level (0-21), 100 games each.

Edax accepts levels 0-60; 22-60 are solve-class and not runnable at 100 games.
Each game uses a seeded random opening so the 100 games are not 50 copies of
two deterministic lines. Edax's book is off. Results stream to JSON.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from othello.agent import EngineAgent
from othello.edax_agent import EdaxAgent
from othello.env import BLACK, PASS_ACTION, WHITE, OthelloEnv

DEFAULT_OUT = ROOT / "graphs" / "vs_edax_levels.json"
STANDARD_LEVELS = list(range(0, 22))  # Edax playing levels 0-21


def apply_opening(env: OthelloEnv, rng: random.Random, plies: int) -> list[int]:
    moves: list[int] = []
    for _ in range(plies):
        if env.done:
            break
        legal = env.legal_actions()
        action = rng.choice(legal)
        env.step(action)
        moves.append(int(action))
    return moves


def play_from(env: OthelloEnv, black, white) -> dict:
    for agent in (black, white):
        if hasattr(agent, "new_game"):
            agent.new_game()
    while not env.done:
        agent = black if env.to_play == BLACK else white
        env.step(agent.select_action(env))
    b, w = env.score()
    return {"winner": env.winner, "black": b, "white": w}


def empty_side() -> dict:
    return {"wins": 0, "losses": 0, "draws": 0}


def run_level(level: int, n_games: int, opening_plies: int, seed: int) -> dict:
    bot = EngineAgent(midgame_depth=9, exact_empties=12, time_limit=1.4)
    edax = EdaxAgent(level=level, n_tasks=1, book=False)
    wins = losses = draws = 0
    margin = 0
    as_black, as_white = empty_side(), empty_side()
    t0 = time.perf_counter()
    try:
        for i in range(n_games):
            rng = random.Random(seed + 10007 * level + i)
            env = OthelloEnv()
            apply_opening(env, rng, opening_plies)
            a_is_black = i % 2 == 0
            black, white = (bot, edax) if a_is_black else (edax, bot)
            result = play_from(env, black, white)
            a_player = BLACK if a_is_black else WHITE
            side = as_black if a_is_black else as_white
            if result["winner"] == 0:
                draws += 1
                side["draws"] += 1
            elif result["winner"] == a_player:
                wins += 1
                side["wins"] += 1
            else:
                losses += 1
                side["losses"] += 1
            a_discs = result["black"] if a_is_black else result["white"]
            b_discs = result["white"] if a_is_black else result["black"]
            margin += a_discs - b_discs
            if (i + 1) % 10 == 0 or i == 0:
                print(
                    f"  edax-l{level}  game {i + 1}/{n_games}  "
                    f"W{wins} D{draws} L{losses}  margin={margin / (i + 1):+.1f}",
                    flush=True,
                )
    finally:
        edax.close()
    n = max(n_games, 1)
    return {
        "edax_level": level,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / n,
        "score": (wins + 0.5 * draws) / n,
        "avg_disc_margin": margin / n,
        "games": n_games,
        "as_black": as_black,
        "as_white": as_white,
        "seconds": round(time.perf_counter() - t0, 1),
    }


def _worker(payload: tuple) -> dict:
    level, n_games, opening_plies, seed = payload
    return run_level(level, n_games, opening_plies, seed)


def load_report(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {
        "bot": {
            "engine": "EngineAgent",
            "midgame_depth": 9,
            "exact_empties": 12,
            "time_limit": 1.4,
            "edax": False,
        },
        "edax": {"book": False, "n_tasks": 1},
        "opening_plies": None,
        "games_per_level": None,
        "levels": {},
        "note": (
            "Our packaged engine vs Edax playing levels 0-21. "
            "100 games each, alternating colors, seeded random openings. "
            "Edax book off. Levels 22-60 are solve-class and omitted."
        ),
    }


def save_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2))
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="EngineAgent vs every Edax playing level")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--opening-plies", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--levels", type=str, default="0-21", help="e.g. 0-21 or 0,1,2")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = parser.parse_args()

    if "-" in args.levels and "," not in args.levels:
        a, b = args.levels.split("-", 1)
        levels = list(range(int(a), int(b) + 1))
    else:
        levels = [int(x) for x in args.levels.split(",") if x.strip() != ""]

    out = Path(args.out)
    report = load_report(out)
    report["opening_plies"] = args.opening_plies
    report["games_per_level"] = args.games
    report["seed"] = args.seed
    save_report(out, report)

    todo = [lv for lv in levels if str(lv) not in report["levels"]]
    print(
        f"levels {levels}  games={args.games}  workers={args.workers}  "
        f"todo={todo}  out={out}",
        flush=True,
    )
    if not todo:
        print("nothing to do", flush=True)
        return

    jobs = [(lv, args.games, args.opening_plies, args.seed) for lv in todo]
    if args.workers <= 1 or len(todo) == 1:
        for job in jobs:
            stats = _worker(job)
            report["levels"][str(job[0])] = stats
            save_report(out, report)
            print(json.dumps(stats), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_worker, job): job[0] for job in jobs}
            for fut in as_completed(futs):
                stats = fut.result()
                report["levels"][str(stats["edax_level"])] = stats
                save_report(out, report)
                print(json.dumps(stats), flush=True)

    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
