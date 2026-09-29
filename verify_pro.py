"""Verify the packaged original engine against weaker bots."""

from __future__ import annotations

import json
from pathlib import Path

from othello.agent import EngineAgent, SearchAgent
from othello.match import play_match
from othello.opponents import GreedyAgent, RandomAgent

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "graphs" / "pro_strength.json"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # No clock: finish the advertised midgame depth so we out-search legacy d6.
    bot = EngineAgent(midgame_depth=8, exact_empties=12, time_limit=None)
    ladder = [
        ("random", RandomAgent(), 4),
        ("greedy", GreedyAgent(), 2),
        ("legacy_search_d6", SearchAgent(midgame_depth=6, exact_empties=8, parity_adjust=True), 2),
    ]
    report = {
        "bot": {
            "engine": "original",
            "name": "EngineAgent",
            "midgame_depth": 8,
            "exact_empties": 12,
            "time_limit": None,
            "eval": "positional/mobility/stability core + self-play pattern residual",
            "edax": False,
        },
        "note": (
            "Packaged bot is our searcher: bitboard PVS, original eval, exact endgame. "
            "Edax is not used at play time. Winning this ladder is a check that the "
            "original engine plays legal, crushing Othello."
        ),
        "opponents": {},
        "total_losses": 0,
        "unbeaten": True,
    }
    for name, opp, n in ladder:
        print(f"original vs {name} ({n} games)", flush=True)
        stats = play_match(bot, opp, n_games=n)
        report["opponents"][name] = stats
        report["total_losses"] += stats["losses"]
        if stats["losses"]:
            report["unbeaten"] = False
        print(
            f"  W{stats['wins']} D{stats['draws']} L{stats['losses']} "
            f"score={stats['score']:.3f} margin={stats['avg_disc_margin']:.1f}",
            flush=True,
        )
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if report["total_losses"]:
        raise SystemExit("Original engine lost a game to a far weaker opponent")


if __name__ == "__main__":
    main()
