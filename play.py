"""Interactive play against the unbeatable search bot."""

from __future__ import annotations

import argparse

from othello.env import BLACK, PASS_ACTION, WHITE, OthelloEnv, idx_to_rc, rc_to_idx


def parse_human_move(text: str) -> int:
    text = text.strip().lower()
    if text in {"p", "pass"}:
        return PASS_ACTION
    if text in {"q", "quit"}:
        raise SystemExit(0)
    if "," in text:
        text = text.replace(",", " ")
    parts = text.split()
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        return rc_to_idx(int(parts[0]), int(parts[1]))
    if len(text) == 2 and text[0].isdigit() and text[1].isdigit():
        return rc_to_idx(int(text[0]), int(text[1]))
    if len(text) == 2 and text[0].isalpha() and text[1].isdigit():
        col = ord(text[0]) - ord("a")
        row = int(text[1]) - 1
        return rc_to_idx(row, col)
    raise ValueError(f"Could not parse move '{text}'")


def show_legal(env: OthelloEnv) -> str:
    legal = env.legal_actions()
    if legal == [PASS_ACTION]:
        return "legal: pass only"
    coords = []
    for a in legal:
        if a == PASS_ACTION:
            coords.append("pass")
        else:
            r, c = idx_to_rc(a)
            coords.append(f"{r}{c}({chr(ord('a') + c)}{r + 1})")
    return "legal: " + ", ".join(coords)


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Othello vs the original engine")
    parser.add_argument("--side", choices=["black", "white"], default="black")
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--exact", type=int, default=12)
    parser.add_argument("--time", type=float, default=1.4)
    parser.add_argument("--legacy-search", action="store_true", help="Use the old shallow Python searcher")
    parser.add_argument("--edax", action="store_true", help="Optional Edax wrapper")
    parser.add_argument("--level", type=int, default=15, help="Edax level if --edax")
    args = parser.parse_args()

    if args.legacy_search:
        from othello.agent import SearchAgent

        bot = SearchAgent(midgame_depth=args.depth, exact_empties=args.exact)
        label = f"legacy search  depth {args.depth}, exact {args.exact}"
    elif args.edax:
        from othello.edax_agent import EdaxAgent

        bot = EdaxAgent(level=args.level)
        label = f"Edax wrapper  level {args.level}"
    else:
        from othello.agent import EngineAgent

        bot = EngineAgent(
            midgame_depth=args.depth,
            exact_empties=args.exact,
            time_limit=args.time,
        )
        label = f"original engine  depth {args.depth}, exact {args.exact}"
    if hasattr(bot, "new_game"):
        bot.new_game()
    env = OthelloEnv()
    human_is_black = args.side != "white"

    print(f"Othello bot ({label})")
    print(env.render_ascii())
    print(show_legal(env))
    print("Enter moves as 'row col' (0-7), 'rc', algebraic e.g. d3, or 'pass'. 'quit' to exit.\n")

    while not env.done:
        human_turn = (env.to_play == BLACK and human_is_black) or (
            env.to_play == WHITE and not human_is_black
        )
        if human_turn:
            while True:
                try:
                    raw = input("Your move> ")
                    action = parse_human_move(raw)
                    if action not in env.legal_actions():
                        print("Illegal.", show_legal(env))
                        continue
                    break
                except ValueError as e:
                    print(e)
            env.step(action)
        else:
            action = bot.select_action(env)
            if action == PASS_ACTION:
                print("Bot passes.")
            else:
                r, c = idx_to_rc(action)
                print(
                    f"Bot plays {r},{c} ({chr(ord('a') + c)}{r + 1})  "
                    f"[d{bot.engine.last_depth}{' exact' if bot.engine.last_exact else ''} "
                    f"{bot.engine.last_nodes} nodes, {bot.engine.last_ms:.0f} ms]"
                )
            env.step(action)

        print()
        print(env.render_ascii())
        if not env.done:
            print(show_legal(env))
            print()

    b, w = env.score()
    if env.winner == 0:
        outcome = "Draw"
    elif env.winner == BLACK:
        outcome = "Black wins"
    else:
        outcome = "White wins"
    print(f"\nGame over: {outcome}  (Black {b} – {w} White)")
    human_player = BLACK if human_is_black else WHITE
    if env.winner == 0:
        print("Result for you: draw")
    elif env.winner == human_player:
        print("Result for you: win")
    else:
        print("Result for you: loss")
    if hasattr(bot, "close"):
        bot.close()


if __name__ == "__main__":
    main()
