#!/usr/bin/env python3
"""Viral-paced Instagram Reel: Day 8, 52-0 hook, fast cuts."""

from __future__ import annotations

import json
import subprocess
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
REELS = ROOT / "reels"
BUILD = REELS / "_build"
ASSETS = REELS / "assets"
CURSOR_ASSETS = Path(
    "/Users/maxingargiola/.cursor/projects/Users-maxingargiola-Desktop-LockIn-BeatGames-Othello/assets"
)

W, H = 1080, 1920
FPS = 30
SIDE = 64

INK = (246, 241, 230)
MUTED = (198, 192, 176)
GOLD = (232, 196, 122)
GREEN = (61, 214, 140)
DARK = (6, 8, 7)
RED = (255, 78, 66)

# Punchy. Short. Built for a 30-second completion rate.
VO_SEGMENTS = [
    ("Fifty two to zero.", 0.05),
    ("Day eight.", 0.06),
    ("This engine class has been beating world champions since nineteen ninety seven.", 0.08),
    ("We built our own first.", 0.04),
    ("Python.", 0.03),
    ("Search.", 0.03),
    ("Not enough.", 0.06),
    ("The gap was evaluation.", 0.05),
    ("Trained patterns. Not toy weights.", 0.06),
    ("Eedax four point six.", 0.05),
    ("Fifteen ply.", 0.03),
    ("Exact endgame.", 0.05),
    ("Zero losses.", 0.04),
    ("Fifty disc margins.", 0.05),
    ("Level fifteen.", 0.04),
    ("Superhuman.", 0.08),
    ("Day eight. See you tomorrow.", 0.40),
]

# Extra visual cuts inside / across VO lines. (label, start_frac_of_total)
# Actual times are assigned after VO is measured.


def font(path: str, size: int, index: int = 0) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size=size, index=index)


def f_display(size: int, weight: str = "bold") -> ImageFont.FreeTypeFont:
    idx = {"bold": 0, "demi": 2, "medium": 5, "regular": 7, "heavy": 8}[weight]
    return font("/System/Library/Fonts/Avenir Next.ttc", size, idx)


def f_cond(size: int, weight: str = "bold") -> ImageFont.FreeTypeFont:
    idx = {"bold": 0, "demi": 2, "medium": 5, "regular": 7}[weight]
    return font("/System/Library/Fonts/Avenir Next Condensed.ttc", size, idx)


def f_num(size: int) -> ImageFont.FreeTypeFont:
    return font("/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf", size)


def f_label(size: int) -> ImageFont.FreeTypeFont:
    return font("/System/Library/Fonts/HelveticaNeue.ttc", size, 10)


def ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def run(cmd: list[str]) -> None:
    subprocess.check_call(cmd)


def load_bg(name: str) -> Image.Image:
    for folder in (ASSETS, CURSOR_ASSETS):
        path = folder / name
        if path.exists():
            img = Image.open(path).convert("RGB")
            break
    else:
        img = Image.new("RGB", (W, H), DARK)
    img = img.resize((W, H), Image.Resampling.LANCZOS)
    arr = np.asarray(img).astype(np.float32)
    yy = np.linspace(0, 1, H)[:, None]
    fade = np.clip((yy - 0.06) / 0.10, 0, 1)
    dark = np.array(DARK, dtype=np.float32)
    arr = arr * fade[..., None] + dark * (1 - fade[..., None])
    fade_b = np.clip((0.93 - yy) / 0.10, 0, 1)
    arr = arr * fade_b[..., None] + dark * (1 - fade_b[..., None])
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return ImageEnhance.Brightness(img).enhance(0.55)


def vignette_mask() -> Image.Image:
    y, x = np.ogrid[:H, :W]
    cx, cy = W / 2, H * 0.48
    r = np.sqrt(((x - cx) / (W * 0.62)) ** 2 + ((y - cy) / (H * 0.56)) ** 2)
    v = np.clip(1.25 - r, 0.12, 1.0)
    layer = np.zeros((H, W, 4), dtype=np.uint8)
    layer[..., 3] = ((1.0 - v) * 220).astype(np.uint8)
    return Image.fromarray(layer, "RGBA")


def grain_tile() -> Image.Image:
    rng = np.random.default_rng(3)
    noise = rng.integers(0, 255, (H, W), dtype=np.uint8)
    g = Image.fromarray(noise, "L").convert("RGBA")
    g.putalpha(28)
    return g


def disc(draw: ImageDraw.ImageDraw, cx: int, cy: int, rad: int, black: bool) -> None:
    if black:
        draw.ellipse([cx - rad, cy - rad + 3, cx + rad, cy + rad + 5], fill=(0, 0, 0, 90))
        draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=(16, 16, 16, 255))
        draw.ellipse([cx - rad + 3, cy - rad + 3, cx + rad - 3, cy + rad - 3], fill=(28, 28, 28, 255))
        draw.ellipse(
            [cx - rad // 2, cy - rad + 4, cx + rad // 3, cy - rad // 3],
            fill=(70, 70, 68, 70),
        )
    else:
        draw.ellipse([cx - rad, cy - rad + 3, cx + rad, cy + rad + 5], fill=(0, 0, 0, 70))
        draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=(236, 232, 220, 255))
        draw.ellipse([cx - rad + 4, cy - rad + 4, cx + rad - 6, cy + rad - 6], fill=(248, 245, 236, 255))
        draw.ellipse(
            [cx - rad // 2, cy - rad + 5, cx + rad // 4, cy - rad // 4],
            fill=(255, 255, 255, 90),
        )


def render_board(board: np.ndarray, last_move: int | None, cell: int = 96) -> Image.Image:
    pad, frame = 18, 16
    size = pad * 2 + frame * 2 + cell * 8
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=26, fill=(62, 42, 24, 255))
    d.rounded_rectangle([7, 7, size - 8, size - 8], radius=20, fill=(92, 64, 36, 255))
    ox = oy = pad + frame
    board_px = cell * 8
    d.rounded_rectangle(
        [ox - 6, oy - 6, ox + board_px + 6, oy + board_px + 6],
        radius=10,
        fill=(12, 78, 48, 255),
    )
    for r in range(8):
        for c in range(8):
            x0, y0 = ox + c * cell, oy + r * cell
            tone = (15, 110, 68, 255) if (r + c) % 2 == 0 else (12, 94, 58, 255)
            d.rectangle([x0, y0, x0 + cell, y0 + cell], fill=tone)
    if last_move is not None and 0 <= last_move < 64:
        lr, lc = divmod(last_move, 8)
        x0 = ox + lc * cell + 3
        y0 = oy + lr * cell + 3
        d.rounded_rectangle(
            [x0, y0, x0 + cell - 6, y0 + cell - 6],
            radius=8,
            outline=(232, 196, 122, 240),
            width=4,
        )
    rad = cell // 2 - 7
    for r in range(8):
        for c in range(8):
            v = int(board[r, c])
            if v == 0:
                continue
            disc(d, ox + c * cell + cell // 2, oy + r * cell + cell // 2, rad, black=(v == 1))
    return img


def shadow_text(draw, xy, text, font_obj, fill=INK, anchor="mm") -> None:
    x, y = xy
    draw.text((x + 2, y + 3), text, font=font_obj, fill=(0, 0, 0, 190), anchor=anchor)
    draw.text(xy, text, font=font_obj, fill=fill, anchor=anchor)


def wrapped_block(draw, text, y, font_obj, fill=INK, gap=4) -> None:
    lines = text.split("\n")
    heights = []
    for line in lines:
        b = draw.textbbox((0, 0), line, font=font_obj)
        heights.append(b[3] - b[1])
    total = sum(heights) + gap * (len(lines) - 1)
    cy = y - total // 2
    for line, h in zip(lines, heights):
        shadow_text(draw, (W // 2, cy + h // 2), line, font_obj, fill=fill)
        cy += h + gap


def day_badge(draw, y: int = 272) -> None:
    label = "DAY  8"
    fnt = f_cond(44, "bold")
    bb = draw.textbbox((0, 0), label, font=fnt)
    tw = bb[2] - bb[0]
    box = [W // 2 - tw // 2 - 36, y - 34, W // 2 + tw // 2 + 36, y + 34]
    draw.rounded_rectangle(box, radius=12, fill=(232, 196, 122, 250))
    draw.text((W // 2, y), label, font=fnt, fill=(18, 14, 8, 255), anchor="mm")


def draw_score(draw, y: int, black: int, white: int) -> None:
    chip = f"{black}   –   {white}"
    shadow_text(draw, (W // 2, y), chip, f_num(52), fill=INK)
    r = 17
    for cx, fill, outline in (
        (W // 2 - 158, (12, 12, 12, 255), (236, 230, 214, 255)),
        (W // 2 + 158, (242, 238, 228, 255), (48, 44, 38, 255)),
    ):
        draw.ellipse([cx - r, y - r, cx + r, y + r], fill=fill, outline=outline, width=3)


def probe_dur(path: Path) -> float:
    return float(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nw=1:nk=1",
                str(path),
            ],
            text=True,
        ).strip()
    )


def generate_vo(times: list[tuple[float, float]]) -> Path:
    segs_dir = BUILD / "vo_segs_v2"
    segs_dir.mkdir(parents=True, exist_ok=True)
    wavs = []
    cursor = 0.10
    times.clear()
    for i, (text, pause) in enumerate(VO_SEGMENTS):
        aiff = segs_dir / f"{i:02d}.aiff"
        wav = segs_dir / f"{i:02d}.wav"
        run(["say", "-v", "Reed", "-r", "208", "-o", str(aiff), text])
        run(["ffmpeg", "-y", "-i", str(aiff), "-ar", "48000", "-ac", "1", str(wav)])
        dur = probe_dur(wav)
        times.append((cursor, cursor + dur))
        wavs.append((wav, dur, pause))
        cursor += dur + pause
    parts = BUILD / "vo_parts_v2.txt"
    lines = []
    preroll = BUILD / "silence_010.wav"
    run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=mono",
            "-t",
            "0.10",
            str(preroll),
        ]
    )
    lines.append(f"file '{preroll.resolve()}'")
    silent_cache = {}
    for wav, _dur, pause in wavs:
        lines.append(f"file '{wav.resolve()}'")
        key = f"{pause:.2f}"
        if key not in silent_cache:
            sp = BUILD / f"sil_v2_{key.replace('.', '')}.wav"
            run(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "anullsrc=r=48000:cl=mono",
                    "-t",
                    f"{pause:.2f}",
                    str(sp),
                ]
            )
            silent_cache[key] = sp
        lines.append(f"file '{silent_cache[key].resolve()}'")
    parts.write_text("\n".join(lines) + "\n")
    raw = BUILD / "vo_concat_v2.wav"
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(parts), "-c", "copy", str(raw)])
    vo = BUILD / "vo_v2.wav"
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(raw),
            "-af",
            "highpass=f=80,lowpass=f=11000,acompressor=threshold=-14dB:ratio=3.2:attack=6:release=90,"
            "aecho=0.7:0.6:18:0.08,loudnorm=I=-15:TP=-1.2:LRA=7",
            "-ar",
            "48000",
            "-ac",
            "1",
            str(vo),
        ]
    )
    return vo


def write_wav(path: Path, stereo: np.ndarray, sr: int = 48000) -> None:
    pcm = (np.clip(stereo, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def synth_music(duration: float, move_times: list[float], hits: list[float]) -> Path:
    sr = 48000
    n = int(duration * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(21)
    out = np.zeros((n, 2), dtype=np.float64)
    bpm = 128.0
    beat = 60.0 / bpm

    # Dark stab chords, 2 beats each
    chords = [(73.42, 110.00, 146.83), (65.41, 98.00, 130.81), (87.31, 130.81, 174.61), (82.41, 123.47, 164.81)]
    for i, (f1, f2, f3) in enumerate(chords * 16):
        start = int(i * 2 * beat * sr)
        if start >= n:
            break
        length = int(1.6 * beat * sr)
        end = min(n, start + length)
        tt = np.arange(end - start) / sr
        env = np.exp(-tt * 3.2) * (tt < 1.6 * beat)
        tone = 0.16 * np.sin(2 * np.pi * f1 * tt) + 0.10 * np.sin(2 * np.pi * f2 * tt) + 0.06 * np.sin(2 * np.pi * f3 * tt)
        out[start:end] += np.stack([tone * env, tone * env * 1.05], axis=1)

    for i, time in enumerate(np.arange(0, duration, beat)):
        s = int(time * sr)
        # four-on-the-floor kick
        length = int(0.16 * sr)
        tt = np.arange(min(length, n - s)) / sr
        kick = np.sin(2 * np.pi * (86 * np.exp(-tt * 22)) * tt) * np.exp(-tt * 12)
        out[s : s + len(tt), :] += kick[:, None] * 0.34
        # snare on 2 and 4
        if i % 2 == 1:
            noise = rng.normal(0, 1, len(tt)) * np.exp(-tt * 28)
            tone = 0.25 * np.sin(2 * np.pi * 180 * tt) * np.exp(-tt * 20)
            sn = (noise * 0.55 + tone) * 0.22
            out[s : s + len(tt), :] += sn[:, None]
        # hats
        hlen = int(0.025 * sr)
        hat = rng.normal(0, 1, hlen) * np.exp(-np.arange(hlen) / sr * 90)
        for off in (0.0, 0.5):
            hs = int((time + off * beat) * sr)
            if 0 <= hs < n:
                sl = slice(hs, min(n, hs + hlen))
                out[sl, 0] += hat[: sl.stop - sl.start] * 0.055
                out[sl, 1] += hat[: sl.stop - sl.start] * 0.04

    # Sub
    out[:, 0] += 0.09 * np.sin(2 * np.pi * 38 * t)
    out[:, 1] += 0.09 * np.sin(2 * np.pi * 38 * t + 0.15)

    # Move clicks
    for mt in move_times:
        s = int(mt * sr)
        if s >= n:
            continue
        tt = np.arange(min(int(0.045 * sr), n - s)) / sr
        click = rng.normal(0, 1, len(tt)) * np.exp(-tt * 100) + 0.5 * np.sin(2 * np.pi * 1100 * tt) * np.exp(-tt * 50)
        out[s : s + len(tt), 0] += click * 0.10
        out[s : s + len(tt), 1] += click * 0.08

    # Hard hits / whooshes on scene slams
    for ht in hits:
        s = int(ht * sr)
        if s >= n:
            continue
        tt = np.arange(min(int(0.22 * sr), n - s)) / sr
        boom = np.sin(2 * np.pi * (55 * np.exp(-tt * 14)) * tt) * np.exp(-tt * 8)
        whoosh = rng.normal(0, 1, len(tt)) * np.exp(-tt * 12) * (1 - tt / 0.22)
        hit = boom * 0.38 + whoosh * 0.10
        out[s : s + len(tt), :] += hit[:, None]

    fade = int(0.08 * sr)
    out[:fade] *= np.linspace(0, 1, fade)[:, None]
    tail = int(0.7 * sr)
    out[-tail:] *= np.linspace(1, 0, tail)[:, None]
    peak = np.max(np.abs(out)) or 1.0
    out = out / peak * 0.72
    path = BUILD / "music_v2.wav"
    write_wav(path, out, sr)
    return path


def mix_audio(vo: Path, music: Path, duration: float) -> Path:
    mixed = BUILD / "mix_v2.wav"
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(vo),
            "-i",
            str(music),
            "-filter_complex",
            "[1:a]volume=0.34[m];[0:a]aformat=channel_layouts=stereo,volume=1.18[v];"
            "[v][m]amix=inputs=2:duration=longest:normalize=0,loudnorm=I=-13:TP=-1.0:LRA=7",
            "-t",
            f"{duration:.3f}",
            "-ar",
            "48000",
            "-ac",
            "2",
            str(mixed),
        ]
    )
    return mixed


def scene_for(t: float, spans: list[tuple[float, float]]) -> int:
    idx = 0
    for i, (a, _b) in enumerate(spans):
        if t >= a:
            idx = i
    return idx


def shot_kind(t: float, si: int, local: float) -> str:
    """Hard cuts. Hook is time-based so the first second is not one hold."""
    if t < 0.26:
        return "num52"
    if t < 0.52:
        return "score_only"
    if t < 0.84:
        return "smash"
    if t < 1.12:
        return "notypo"
    if si == 0:
        return "smash"
    if si == 1:
        return "day_line"
    if si == 2:
        if local < 1.05:
            return "packaged"
        if local < 2.15:
            return "engine_line"
        if local < 3.20:
            return "champs"
        return "year"
    if si == 4:
        return "word_python"
    if si == 5:
        return "word_search"
    if si == 6:
        return "word_notenough"
    if si == 10:
        return "ply"
    if si == 11:
        return "exact"
    if si == 12:
        return "zero"
    if si == 13:
        return "fifty"
    if si == 14:
        return "level"
    if si == 15:
        return "super"
    if si == 16:
        return "outro"
    return f"line_{si}"


def draw_frame(
    plate: Image.Image,
    si: int,
    kind: str,
    local: float,
    board_img: Image.Image,
    score: tuple[int, int],
    show_board: bool,
    zoom: float,
    flash: float,
    shake: tuple[int, int],
) -> Image.Image:
    frame = plate.convert("RGBA")
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    board_y = 620
    if show_board:
        bw, bh = board_img.size
        scale = 0.90 * zoom
        b = board_img.resize((max(8, int(bw * scale)), max(8, int(bh * scale))), Image.Resampling.LANCZOS)
        bx = (W - b.size[0]) // 2 + shake[0]
        by = board_y + shake[1]
        if kind in {"smash", "notypo", "zero", "fifty", "level", "super"}:
            by = 700 + shake[1]
        frame.alpha_composite(b, (bx, by))
        if kind not in {"smash", "smash2", "zero", "fifty"}:
            draw_score(d, by + b.size[1] + 44, score[0], score[1])

    day_badge(d)

    if kind == "num52":
        shadow_text(d, (W // 2, 900), "52", f_num(420), fill=GOLD)
    elif kind == "score_only":
        shadow_text(d, (W // 2, 860), "52–0", f_num(280), fill=GOLD)
        shadow_text(d, (W // 2, 1080), "FINAL SCORE", f_cond(44, "bold"), fill=INK)
    elif kind in {"smash", "notypo"}:
        wrapped_block(d, "52–0", 420, f_num(200), fill=GOLD, gap=0)
        if kind == "notypo":
            shadow_text(d, (W // 2, 555), "NOT A TYPO", f_cond(40, "bold"), fill=RED)
    elif kind == "day_line":
        wrapped_block(d, "1 VIDEO\nA DAY", 400, f_display(70, "heavy"), fill=INK)
    elif kind == "packaged":
        wrapped_block(d, "I PACKAGED\nTHE ENGINE", 400, f_display(62, "bold"))
    elif kind == "engine_line":
        wrapped_block(d, "AN OTHELLO\nENGINE", 400, f_display(66, "bold"), fill=GOLD)
    elif kind == "champs":
        wrapped_block(d, "WORLD\nCHAMPIONS", 400, f_display(70, "heavy"))
    elif kind == "year":
        shadow_text(d, (W // 2, 400), "1997", f_num(180), fill=GOLD)
        shadow_text(d, (W // 2, 540), "this class of engine", f_label(28), fill=MUTED)
    elif si == 3:
        wrapped_block(d, "We built our\nown first", 400, f_display(62, "bold"))
    elif kind == "word_python":
        shadow_text(d, (W // 2, 420), "PYTHON", f_num(130), fill=INK)
    elif kind == "word_search":
        shadow_text(d, (W // 2, 420), "SEARCH", f_num(130), fill=GOLD)
    elif kind == "word_notenough":
        shadow_text(d, (W // 2, 420), "NOT ENOUGH", f_cond(92, "bold"), fill=RED)
    elif si == 7:
        wrapped_block(d, "The real gap", 360, f_display(38, "demi"), fill=MUTED)
        shadow_text(d, (W // 2, 450), "EVALUATION", f_cond(76, "bold"), fill=GOLD)
    elif si == 8:
        wrapped_block(d, "Trained patterns", 390, f_display(54, "bold"), fill=GOLD)
        shadow_text(d, (W // 2, 500), "not 64 toy weights", f_display(34, "demi"), fill=MUTED)
    elif si == 9:
        shadow_text(d, (W // 2, 400), "EDAX 4.6", f_num(120), fill=INK)
    elif kind == "ply":
        shadow_text(d, (W // 2, 410), "15-PLY", f_num(140), fill=GOLD)
        shadow_text(d, (W // 2, 530), "MIDGAME", f_cond(48, "bold"), fill=INK)
    elif kind == "exact":
        wrapped_block(d, "EXACT\nENDGAME", 420, f_num(110), fill=GREEN)
    elif kind == "zero":
        shadow_text(d, (W // 2, 780), "0", f_num(360), fill=GREEN)
        shadow_text(d, (W // 2, 1040), "LOSSES", f_cond(72, "bold"), fill=INK)
    elif kind == "fifty":
        shadow_text(d, (W // 2, 780), "+50", f_num(260), fill=GOLD)
        shadow_text(d, (W // 2, 1000), "DISC MARGIN", f_cond(52, "bold"), fill=INK)
    elif kind == "level":
        wrapped_block(d, "LEVEL 15", 400, f_num(120), fill=INK)
    elif kind == "super":
        shadow_text(d, (W // 2, 410), "SUPERHUMAN", f_cond(84, "bold"), fill=GOLD)
    elif kind == "outro":
        wrapped_block(d, "DAY 8", 760, f_num(160), fill=GOLD)
        shadow_text(d, (W // 2, 920), "See you tomorrow", f_display(42, "demi"), fill=INK)
        shadow_text(d, (W // 2, 1020), "Edax 4.6  ·  pro-strength Othello", f_label(26), fill=MUTED)

    if flash > 0:
        flash_layer = Image.new("RGBA", (W, H), (255, 248, 230, int(70 * flash)))
        overlay = Image.alpha_composite(overlay, flash_layer)

    frame.alpha_composite(overlay)
    return frame


def main() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    game = json.loads((BUILD / "edax_vs_greedy.json").read_text())
    boards = [np.array(b, dtype=np.int8) for b in game["boards"]]
    moves = game["moves"]
    n_moves = max(len(moves), 1)

    spans_path = BUILD / "spans_v2.json"
    vo = BUILD / "vo_v2.wav"
    mix = BUILD / "mix_v2.wav"
    if vo.exists() and mix.exists() and spans_path.exists():
        print("Reusing v2 audio…", flush=True)
        meta = json.loads(spans_path.read_text())
        spans = [tuple(x) for x in meta["spans"]]
        duration = float(meta["duration"])
    else:
        print("Generating punch voiceover…", flush=True)
        spans = []
        vo = generate_vo(spans)
        duration = spans[-1][1] + VO_SEGMENTS[-1][1] + 0.15
    n_frames = int(round(duration * FPS))
    print(f"spans={len(spans)} duration={duration:.2f}s frames={n_frames}", flush=True)

    # Board: hold the crush for the hook, then rip through the whole game.
    hook_end = spans[2][0]
    play_end = spans[14][1]
    move_times = []
    for i in range(n_moves):
        move_times.append(lerp(hook_end, play_end, (i + 1) / n_moves))
    hits = [0.0, 0.26, 0.52, 0.84, spans[1][0], spans[6][0], spans[9][0], spans[12][0], spans[16][0]]

    print("Synthesizing 128-BPM bed…", flush=True)
    music = synth_music(duration, move_times, hits)
    print("Mixing…", flush=True)
    mix = mix_audio(vo, music, duration)

    print("Caching boards…", flush=True)
    board_cache = []
    for i, board in enumerate(boards):
        last = moves[i - 1] if i > 0 and i - 1 < len(moves) else None
        board_cache.append(render_board(board, last))

    bgs = {
        "champ": load_bg("reel_bg_championship.png"),
        "crush": load_bg("reel_bg_crush.png"),
        "end": load_bg("reel_bg_end.png"),
        "engine": load_bg("reel_bg_engine.png"),
        "discs": load_bg("reel_bg_discs.png"),
    }
    bg_for = [
        "crush",
        "crush",
        "champ",
        "engine",
        "engine",
        "engine",
        "engine",
        "discs",
        "engine",
        "crush",
        "engine",
        "engine",
        "end",
        "end",
        "champ",
        "champ",
        "end",
    ]
    vig = vignette_mask()
    grain = grain_tile()

    raw_path = BUILD / "video_v2.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{W}x{H}",
        "-r",
        str(FPS),
        "-i",
        "pipe:0",
        "-i",
        str(mix),
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-c:v",
        "libx264",
        "-profile:v",
        "high",
        "-level",
        "4.1",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "slow",
        "-crf",
        "16",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-movflags",
        "+faststart",
        "-shortest",
        str(raw_path),
    ]
    print("Rendering…", flush=True)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    cover_saved = False
    last_kind = None
    try:
        for f in range(n_frames):
            t = f / FPS
            si = scene_for(t, spans)
            a, b = spans[si]
            local = t - a
            kind = shot_kind(t, si, local)
            if t < hook_end:
                bi = len(boards) - 1
            elif t >= play_end:
                bi = len(boards) - 1
            else:
                rel = (t - hook_end) / max(play_end - hook_end, 1e-6)
                bi = min(int(rel * n_moves) + 1, len(boards) - 1)
            score = (int(np.sum(boards[bi] == 1)), int(np.sum(boards[bi] == -1)))
            show_board = kind not in {"num52", "score_only", "zero", "fifty", "outro"}

            # Punch zoom + flash on cut
            cut = kind != last_kind
            last_kind = kind
            frames_since_cut = 0 if cut else frames_since_cut + 1
            flash = max(0.0, 1.0 - frames_since_cut / 4.0) if frames_since_cut < 4 else 0.0
            if t < 0.08:
                flash = 0.0
            zoom = 1.0 + 0.06 * max(0.0, 1.0 - frames_since_cut / 8.0)
            shake = (0, 0)
            if frames_since_cut < 3:
                shake = (int((1 - frames_since_cut / 3) * (4 if f % 2 == 0 else -4)), 0)

            key = "end" if kind in {"num52", "score_only"} else bg_for[min(si, len(bg_for) - 1)]
            plate = bgs[key].copy()
            # micro-push in
            z = 1.0 + 0.04 * (t / max(duration, 1e-6)) + (0.03 if kind in {"smash", "zero", "fifty"} else 0)
            nw, nh = int(W * z), int(H * z)
            big = plate.resize((nw, nh), Image.Resampling.BILINEAR)
            plate = big.crop(((nw - W) // 2, (nh - H) // 2, (nw - W) // 2 + W, (nh - H) // 2 + H))
            plate = ImageEnhance.Brightness(plate).enhance(0.36 if si in (1, 9, 11) else 0.62)

            frame = draw_frame(
                plate,
                si,
                kind,
                local,
                board_cache[bi],
                score,
                show_board,
                zoom,
                flash,
                shake,
            )
            frame.alpha_composite(vig)
            frame.alpha_composite(grain)
            rgb = frame.convert("RGB")
            if (not cover_saved) and 0.52 <= t < 0.84:
                rgb.save(REELS / "cover.jpg", quality=94)
                cover_saved = True
            proc.stdin.write(np.asarray(rgb, dtype=np.uint8).tobytes())
            if f % 90 == 0:
                print(f"  f={f}/{n_frames} t={t:.2f} si={si} {kind}", flush=True)
    finally:
        proc.stdin.close()
        rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"ffmpeg failed {rc}")

    final = REELS / "Othello_Edax_Reel.mp4"
    if final.exists():
        final.unlink()
    raw_path.replace(final)
    (BUILD / "spans_v2.json").write_text(json.dumps({"spans": spans, "duration": duration}, indent=2))
    (REELS / "INSTAGRAM_CAPTION.txt").write_text(
        "Day 8 of 1 video a day.\n\n"
        "52–0. That’s Edax 4.6 — the class of Othello engine that has been above "
        "world-championship human play since 1997.\n\n"
        "We built a Python searcher first. Bitboards. Alpha-beta. Exact endgame. "
        "Strong for a hobby bot. Not professional. The gap was evaluation: trained "
        "pattern tables, not 64 square weights. So we packaged Edax. Default level 15.\n\n"
        "Wrapper check vs random, greedy, and our old depth-6 searcher: 0 losses. "
        "Typical margin ~50 discs.\n\n"
        "We cannot seat a living world champion in CI. The claim is Edax-class "
        "strength, wrapped and verified.\n\n"
        "Day 8. See you tomorrow.\n\n"
        "#day8 #1videoaday #othello #reversi #edax #ai #coding #gamedev"
        "\n"
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            "0.70",
            "-i",
            str(final),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(REELS / "cover.jpg"),
        ]
    )
    print("Wrote", final, flush=True)


if __name__ == "__main__":
    main()
