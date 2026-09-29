"""Evaluate the search bot against a ladder of opponents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from othello.agent import EngineAgent, SearchAgent
from othello.baselines import PositionalAgent
from othello.match import play_match
from othello.opponents import DepthSearchAgent, GreedyAgent, MobilityAgent, RandomAgent


def default_ladder(bot: SearchAgent) -> dict:
    return {
        "random": (RandomAgent(), 40),
        "greedy": (GreedyAgent(), 20),
        "mobility": (MobilityAgent(), 16),
        "positional": (PositionalAgent(), 12),
        "depth2": (DepthSearchAgent(2, adaptive=False), 8),
        "depth3": (DepthSearchAgent(3, adaptive=False), 8),
        "depth4": (DepthSearchAgent(4, adaptive=False), 6),
        "depth5": (DepthSearchAgent(5, adaptive=True), 4),
        "depth6": (DepthSearchAgent(6, adaptive=True), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the unbeatable Othello bot")
    parser.add_argument("--games", type=int, default=0, help="Override games per opponent (0=ladder defaults)")
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--exact", type=int, default=12)
    parser.add_argument("--time", type=float, default=0.85)
    parser.add_argument("--legacy-search", action="store_true")
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    if args.legacy_search:
        bot = SearchAgent(midgame_depth=args.depth, exact_empties=args.exact, time_limit=args.time or None)
    else:
        bot = EngineAgent(midgame_depth=args.depth, exact_empties=args.exact, time_limit=args.time)
    ladder = default_ladder(bot)
    report = {
        "bot": {"depth": args.depth, "exact": args.exact, "time": args.time},
        "opponents": {},
        "total_losses": 0,
        "unbeaten": True,
    }
    for name, (opp, n) in ladder.items():
        games = args.games if args.games > 0 else n
        print(f"vs {name}  ({games} games)...", flush=True)
        stats = play_match(bot, opp, n_games=games)
        report["opponents"][name] = stats
        report["total_losses"] += stats["losses"]
        if stats["losses"]:
            report["unbeaten"] = False
        print(json.dumps({name: stats}, indent=2), flush=True)

    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


if __name__ == "__main__":
    main()
