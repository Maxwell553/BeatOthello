"""Generate graphs/ and animations/ showing the search bot's performance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from othello.agent import SearchAgent
from othello.baselines import PositionalAgent
from othello.env import BLACK, EMPTY, WHITE
from othello.match import play_game, play_match
from othello.opponents import DepthSearchAgent, GreedyAgent, MobilityAgent, RandomAgent

ROOT = Path(__file__).resolve().parent
GRAPHS = ROOT / "graphs"
ANIMS = ROOT / "animations"

GREEN = (15, 107, 67)
DARK = (11, 79, 50)
LINE = (8, 48, 32)
CREAM = (236, 227, 200)


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("Arial.ttf", "DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_board(board, title: str, subtitle: str = "") -> Image.Image:
    cell = 56
    pad = 36
    header = 70
    size = pad * 2 + cell * 8
    img = Image.new("RGB", (size, size + header), (18, 22, 20))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, size, header], fill=(18, 22, 20))
    draw.text((16, 10), title, fill=(240, 240, 235), font=_font(22))
    if subtitle:
        draw.text((16, 40), subtitle, fill=(180, 190, 180), font=_font(16))
    origin = (pad, header + 8)
    board_px = cell * 8
    draw.rectangle(
        [origin[0] - 6, origin[1] - 6, origin[0] + board_px + 6, origin[1] + board_px + 6],
        fill=(90, 62, 32),
    )
    draw.rectangle(
        [origin[0], origin[1], origin[0] + board_px, origin[1] + board_px],
        fill=GREEN,
    )
    for i in range(9):
        x = origin[0] + i * cell
        y = origin[1] + i * cell
        draw.line([origin[0], y, origin[0] + board_px, y], fill=LINE, width=1)
        draw.line([x, origin[1], x, origin[1] + board_px], fill=LINE, width=1)
    letters = "abcdefgh"
    for i in range(8):
        draw.text((origin[0] + i * cell + 20, origin[1] + board_px + 8), letters[i], fill=CREAM, font=_font(14))
        draw.text((origin[0] - 18, origin[1] + i * cell + 18), str(i + 1), fill=CREAM, font=_font(14))
    for r in range(8):
        for c in range(8):
            v = int(board[r, c])
            if v == EMPTY:
                continue
            cx = origin[0] + c * cell + cell // 2
            cy = origin[1] + r * cell + cell // 2
            rad = cell // 2 - 6
            fill = (12, 12, 12) if v == BLACK else (245, 245, 240)
            outline = (40, 40, 40) if v == WHITE else (200, 200, 200)
            draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=fill, outline=outline, width=2)
    return img


def save_gif(frames: List[Image.Image], path: Path, duration: int = 220) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        return
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=False,
    )


def animate_game(result: Dict[str, Any], path: Path, label: str) -> None:
    frames = []
    n = max(len(result["boards"]) - 1, 1)
    for i, board in enumerate(result["boards"]):
        b = int(np.sum(board == BLACK))
        w = int(np.sum(board == WHITE))
        if i == 0:
            sub = "Start  Black 2 – 2 White"
        elif i == len(result["boards"]) - 1:
            winner = result["winner"]
            outcome = {BLACK: "Black wins", WHITE: "White wins", 0: "Draw"}[winner]
            sub = f"Final  Black {b} – {w} White   {outcome}"
        else:
            sub = f"Move {i}/{n}   Black {b} – {w} White"
        frames.append(render_board(board, label, sub))
    # Hold the final position a bit longer.
    if frames:
        frames.extend([frames[-1]] * 6)
    save_gif(frames, path)


def animate_disc_race(result: Dict[str, Any], path: Path, label: str) -> None:
    blacks = [int(np.sum(b == BLACK)) for b in result["boards"]]
    whites = [int(np.sum(b == WHITE)) for b in result["boards"]]
    frames = []
    w, h = 640, 360
    font = _font(18)
    title_font = _font(22)
    for i in range(len(blacks)):
        img = Image.new("RGB", (w, h), (18, 22, 20))
        draw = ImageDraw.Draw(img)
        draw.text((20, 14), label, fill=(240, 240, 235), font=title_font)
        draw.text((20, 44), "Disc count over the game", fill=(170, 180, 170), font=font)
        plot = [48, 80, w - 24, h - 36]
        draw.rectangle(plot, outline=(70, 80, 70), width=1)
        max_y = 64
        def pt(idx, val):
            x = plot[0] + (plot[2] - plot[0]) * idx / max(len(blacks) - 1, 1)
            y = plot[3] - (plot[3] - plot[1]) * val / max_y
            return x, y
        for series, color in ((blacks[: i + 1], (20, 20, 20)), (whites[: i + 1], (230, 230, 225))):
            pts = [pt(j, series[j]) for j in range(len(series))]
            if len(pts) >= 2:
                draw.line(pts, fill=color, width=3)
            if pts:
                x, y = pts[-1]
                draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=color)
        draw.text((20, h - 28), f"Black {blacks[i]}   White {whites[i]}", fill=CREAM, font=font)
        frames.append(img)
    frames.extend([frames[-1]] * 8)
    save_gif(frames, path, duration=120)


def style_axes(ax) -> None:
    ax.set_facecolor("#121614")
    ax.tick_params(colors="#d6dcd6")
    ax.xaxis.label.set_color("#d6dcd6")
    ax.yaxis.label.set_color("#d6dcd6")
    ax.title.set_color("#f2f4f0")
    for spine in ax.spines.values():
        spine.set_color("#3a453a")
    ax.grid(color="#2a332a", linestyle="--", linewidth=0.6, alpha=0.8)


def plot_results(results: Dict[str, Any], graphs_dir: Path) -> None:
    graphs_dir.mkdir(parents=True, exist_ok=True)
    names = list(results["opponents"].keys())
    scores = [results["opponents"][n]["score"] for n in names]
    wins = [results["opponents"][n]["wins"] for n in names]
    draws = [results["opponents"][n]["draws"] for n in names]
    losses = [results["opponents"][n]["losses"] for n in names]
    margins = [results["opponents"][n]["avg_disc_margin"] for n in names]
    labels = [n.replace("depth", "depth-") for n in names]

    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor="#0e120f")
    style_axes(ax)
    colors = ["#3dd68c" if s >= 1.0 else "#f0c14b" if s >= 0.75 else "#e05d5d" for s in scores]
    ax.bar(labels, scores, color=colors, edgecolor="#0e120f")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score  (W + 0.5 D) / N")
    ax.set_title("Unbeatable bot score vs opponents")
    ax.axhline(1.0, color="#3dd68c", linewidth=0.8, linestyle=":")
    fig.tight_layout()
    fig.savefig(graphs_dir / "score_by_opponent.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor="#0e120f")
    style_axes(ax)
    x = np.arange(len(labels))
    ax.bar(x - 0.25, wins, 0.25, label="Wins", color="#3dd68c")
    ax.bar(x, draws, 0.25, label="Draws", color="#f0c14b")
    ax.bar(x + 0.25, losses, 0.25, label="Losses", color="#e05d5d")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Games")
    ax.set_title("Win / draw / loss record (losses should be zero)")
    ax.legend(facecolor="#1a201c", edgecolor="#3a453a", labelcolor="#d6dcd6")
    fig.tight_layout()
    fig.savefig(graphs_dir / "win_draw_loss.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor="#0e120f")
    style_axes(ax)
    ax.bar(labels, margins, color="#6cb6ff", edgecolor="#0e120f")
    ax.axhline(0, color="#d6dcd6", linewidth=0.8)
    ax.set_ylabel("Average disc margin (bot − opponent)")
    ax.set_title("How badly the bot outscores each opponent")
    fig.tight_layout()
    fig.savefig(graphs_dir / "disc_margin.png", dpi=140)
    plt.close(fig)

    # Color split
    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor="#0e120f")
    style_axes(ax)
    black_scores = []
    white_scores = []
    for n in names:
        ab = results["opponents"][n]["as_black"]
        aw = results["opponents"][n]["as_white"]
        nb = max(ab["wins"] + ab["losses"] + ab["draws"], 1)
        nw = max(aw["wins"] + aw["losses"] + aw["draws"], 1)
        black_scores.append((ab["wins"] + 0.5 * ab["draws"]) / nb)
        white_scores.append((aw["wins"] + 0.5 * aw["draws"]) / nw)
    x = np.arange(len(labels))
    ax.bar(x - 0.18, black_scores, 0.36, label="Bot as Black", color="#222")
    ax.bar(x + 0.18, white_scores, 0.36, label="Bot as White", color="#eee", edgecolor="#222")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Color-split strength (both seats)")
    ax.legend(facecolor="#1a201c", edgecolor="#3a453a", labelcolor="#d6dcd6")
    fig.tight_layout()
    fig.savefig(graphs_dir / "color_split.png", dpi=140)
    plt.close(fig)

    # Unbeaten cumulative losses
    fig, ax = plt.subplots(figsize=(11, 4.8), facecolor="#0e120f")
    style_axes(ax)
    cum = np.cumsum(losses)
    ax.plot(labels, cum, marker="o", color="#3dd68c", linewidth=2)
    ax.set_ylabel("Cumulative losses")
    ax.set_title("Unbeaten check: cumulative losses across the ladder")
    ax.set_ylim(-0.05, max(1.0, float(max(cum) + 1)))
    fig.tight_layout()
    fig.savefig(graphs_dir / "unbeaten_cumulative_losses.png", dpi=140)
    plt.close(fig)


def animate_ladder(results: Dict[str, Any], path: Path) -> None:
    names = list(results["opponents"].keys())
    frames = []
    w, h = 720, 420
    for k in range(1, len(names) + 1):
        img = Image.new("RGB", (w, h), (14, 18, 16))
        draw = ImageDraw.Draw(img)
        draw.text((24, 16), "Ladder score vs opponents", fill=(240, 240, 235), font=_font(22))
        draw.text((24, 48), "Bot score after each matchup (W + 0.5 D) / N", fill=(170, 180, 170), font=_font(16))
        plot = [70, 90, w - 30, h - 50]
        draw.rectangle(plot, outline=(70, 80, 70), width=1)
        subset = names[:k]
        bar_w = (plot[2] - plot[0]) / max(len(names), 1) * 0.7
        gap = (plot[2] - plot[0]) / max(len(names), 1)
        for i, name in enumerate(subset):
            score = results["opponents"][name]["score"]
            x0 = plot[0] + i * gap + (gap - bar_w) / 2
            y0 = plot[3] - score * (plot[3] - plot[1])
            color = (61, 214, 140) if score >= 0.999 else (240, 193, 75)
            draw.rectangle([x0, y0, x0 + bar_w, plot[3]], fill=color)
            draw.text((x0, plot[3] + 6), name[:8], fill=(210, 216, 210), font=_font(12))
        frames.append(img)
    frames.extend([frames[-1]] * 8)
    save_gif(frames, path, duration=700)


def run_suite() -> Dict[str, Any]:
    GRAPHS.mkdir(parents=True, exist_ok=True)
    ANIMS.mkdir(parents=True, exist_ok=True)

    # Fast enough to finish, strong enough to never lose to these opponents.
    fast = SearchAgent(midgame_depth=6, exact_empties=8)
    champion = SearchAgent(midgame_depth=8, exact_empties=10)
    ladder = [
        ("random", RandomAgent(), 30, fast),
        ("greedy", GreedyAgent(), 16, fast),
        ("mobility", MobilityAgent(), 12, fast),
        ("positional", PositionalAgent(), 10, fast),
        ("depth2", DepthSearchAgent(2, adaptive=False), 12, champion),
        ("depth3", DepthSearchAgent(3, adaptive=False), 8, champion),
        ("depth4", DepthSearchAgent(4, adaptive=False), 6, champion),
        ("depth5", DepthSearchAgent(5, adaptive=True), 8, champion),
        ("depth6", DepthSearchAgent(6, adaptive=True), 2, champion),
    ]
    results: Dict[str, Any] = {
        "bot": {
            "fast": {"midgame_depth": 6, "exact_empties": 8},
            "champion": {
                "midgame_depth": 8,
                "exact_empties": 10,
                "parity_adjust": True,
            },
        },
        "opponents": {},
        "total_losses": 0,
        "unbeaten": True,
    }
    for name, opp, n, bot in ladder:
        print(f"eval vs {name} ({n} games)", flush=True)
        stats = play_match(bot, opp, n_games=n)
        results["opponents"][name] = stats
        results["total_losses"] += stats["losses"]
        if stats["losses"]:
            results["unbeaten"] = False
        print(
            f"  W{stats['wins']} D{stats['draws']} L{stats['losses']}  "
            f"score={stats['score']:.3f}  margin={stats['avg_disc_margin']:.1f}",
            flush=True,
        )

    (GRAPHS / "results.json").write_text(json.dumps(results, indent=2))
    plot_results(results, GRAPHS)

    # Representative games for animation (fresh engine, slightly more time).
    show_bot = SearchAgent(midgame_depth=8, exact_empties=10)
    anim_specs = [
        ("vs_random", RandomAgent(), "Bot (Black) vs Random"),
        ("vs_greedy", GreedyAgent(), "Bot (Black) vs Greedy"),
        ("vs_positional", PositionalAgent(), "Bot (Black) vs Positional"),
        ("vs_depth4", DepthSearchAgent(4, adaptive=False), "Bot (Black) vs Depth-4 search"),
        ("vs_depth5", DepthSearchAgent(5, adaptive=True), "Bot (Black) vs Depth-5 search"),
        ("vs_depth6", DepthSearchAgent(6, adaptive=True), "Bot (Black) vs Depth-6 search"),
    ]
    for slug, opp, title in anim_specs:
        print(f"animate {slug}", flush=True)
        game = play_game(show_bot, opp, record=True)
        animate_game(game, ANIMS / f"{slug}.gif", title)
        animate_disc_race(game, ANIMS / f"{slug}_discs.gif", title)

    print("animate self-play", flush=True)
    twin = SearchAgent(midgame_depth=8, exact_empties=10)
    self_game = play_game(show_bot, twin, record=True)
    animate_game(self_game, ANIMS / "self_play.gif", "Search bot vs search bot")
    animate_disc_race(self_game, ANIMS / "self_play_discs.gif", "Search bot vs search bot")
    animate_ladder(results, ANIMS / "ladder_scores.gif")

    results["artifacts"] = {
        "graphs": sorted(p.name for p in GRAPHS.glob("*")),
        "animations": sorted(p.name for p in ANIMS.glob("*")),
    }
    (GRAPHS / "results.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    out = run_suite()
    print(json.dumps({k: out[k] for k in ("bot", "unbeaten", "total_losses", "opponents", "artifacts")}, indent=2))
