"""Packaged original Othello bot — bitboard search + trained eval + exact endgame."""

from __future__ import annotations

import argparse

from othello.agent import EngineAgent
from othello.env import BLACK, PASS_ACTION, WHITE, OthelloEnv, idx_to_rc, rc_to_idx


def parse_human_move(text: str) -> int:
    text = text.strip().lower()
    if text in {"p", "pass"}:
        return PASS_ACTION
    if text in {"q", "quit"}:
        raise SystemExit(0)
    text = text.replace(",", " ")
    parts = text.split()
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        return rc_to_idx(int(parts[0]), int(parts[1]))
    if len(text) == 2 and text[0].isdigit() and text[1].isdigit():
        return rc_to_idx(int(text[0]), int(text[1]))
    if len(text) == 2 and text[0].isalpha() and text[1].isdigit():
        return rc_to_idx(int(text[1]) - 1, ord(text[0]) - ord("a"))
    raise ValueError(f"Could not parse move '{text}'")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="bot",
        description="Play Othello against the packaged original engine.",
    )
    parser.add_argument("--side", choices=["black", "white"], default="black")
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--exact", type=int, default=12)
    parser.add_argument("--time", type=float, default=1.4)
    parser.add_argument(
        "--edax",
        action="store_true",
        help="Optional: play the old Edax wrapper instead of our engine",
    )
    parser.add_argument("--level", type=int, default=15, help="Edax level if --edax")
    args = parser.parse_args(argv)

    if args.edax:
        from othello.edax_agent import EdaxAgent

        bot = EdaxAgent(level=args.level)
        bot.new_game()
        label = f"Edax wrapper (level {args.level})"
    else:
        bot = EngineAgent(
            midgame_depth=args.depth,
            exact_empties=args.exact,
            time_limit=args.time,
        )
        bot.new_game()
        label = f"original engine  depth {args.depth}, exact {args.exact}"

    env = OthelloEnv()
    human_black = args.side == "black"

    print(f"{label}")
    print(env.render_ascii())
    print("Moves: 'row col', 'rc', algebraic (d3), or 'pass'. 'quit' exits.\n")

    while not env.done:
        human_turn = (env.to_play == BLACK) == human_black
        if human_turn:
            while True:
                try:
                    action = parse_human_move(input("You> "))
                    if action not in env.legal_actions():
                        print("Illegal move.")
                        continue
                    break
                except ValueError as exc:
                    print(exc)
            env.step(action)
        else:
            action = bot.select_action(env)
            if action == PASS_ACTION:
                print("Bot> pass")
            else:
                r, c = idx_to_rc(action)
                print(f"Bot> {r} {c}  ({chr(ord('a') + c)}{r + 1})")
            env.step(action)
        print(env.render_ascii())
        print()

    b, w = env.score()
    label_w = {BLACK: "Black", WHITE: "White", 0: "Draw"}[
        env.winner if env.winner is not None else 0
    ]
    print(f"Final: Black {b} – {w} White  →  {label_w}")
    if hasattr(bot, "close"):
        bot.close()


if __name__ == "__main__":
    main()
