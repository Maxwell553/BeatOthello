"""Edax native-UI wrapper — packaged pro-strength Othello engine."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from othello.env import BLACK, PASS_ACTION, WHITE, OthelloEnv, idx_to_rc, rc_to_idx

ROOT = Path(__file__).resolve().parents[1]
EDAX_ROOT = ROOT / "third_party" / "edax-reversi"
EDAX_BIN = EDAX_ROOT / "bin" / "mEdax-native"


def board_to_setboard(env: OthelloEnv) -> str:
    chars = []
    for r in range(8):
        for c in range(8):
            v = int(env.board[r, c])
            if v == BLACK:
                chars.append("X")
            elif v == WHITE:
                chars.append("O")
            else:
                chars.append("-")
    side = "X" if env.to_play == BLACK else "O"
    return "".join(chars) + " " + side


def edax_to_idx(text: str) -> int:
    t = text.strip().lower().replace(".", "")
    if t in {"pass", "pa", "ps"}:
        return PASS_ACTION
    if len(t) >= 2 and t[0].isalpha() and t[1].isdigit():
        return rc_to_idx(int(t[1]) - 1, ord(t[0]) - ord("a"))
    raise ValueError(f"Cannot parse Edax move '{text}'")


class EdaxAgent:
    """Long-lived Edax process using setboard + go.

    Level N ≈ N-ply midgame with Edax's pattern eval, then exact endgame.
    Level 12+ is well above professional human play.
    """

    def __init__(self, level: int = 15, n_tasks: int = 2, book: bool = True) -> None:
        if not EDAX_BIN.exists():
            raise FileNotFoundError(
                f"Edax binary missing: {EDAX_BIN}\nBuild with: bash scripts/build_edax.sh"
            )
        if not (EDAX_ROOT / "data" / "eval.dat").exists():
            raise FileNotFoundError(f"Missing {EDAX_ROOT / 'data' / 'eval.dat'}")
        self.level = level
        self.n_tasks = n_tasks
        self.book = book
        self.proc: Optional[subprocess.Popen] = None
        self.last_depth = level
        self.last_exact = False
        self.last_nodes = 0
        self.last_ms = 0.0
        self.last_value = 0.0
        self.engine = self
        self._lock = threading.Lock()
        self._start()

    def _start(self) -> None:
        self.close()
        cmd = [
            str(EDAX_BIN),
            "-l",
            str(self.level),
            "-n",
            str(self.n_tasks),
            "-book-usage",
            "on" if self.book else "off",
            "-ponder",
            "off",
        ]
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(EDAX_ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._send("mode 3")
        self._send("verbose 0")
        time.sleep(0.15)
        self._drain(0.15)

    def _send(self, cmd: str) -> None:
        assert self.proc is not None and self.proc.stdin
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _drain(self, seconds: float) -> None:
        assert self.proc is not None and self.proc.stdout
        end = time.time() + seconds
        self.proc.stdout.flush()
        while time.time() < end:
            # nonblocking-ish: readline can block; keep drain short at startup only
            break

    def _read_until_plays(self, timeout: float = 120.0) -> str:
        assert self.proc is not None and self.proc.stdout
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if line == "":
                raise RuntimeError("Edax exited")
            if "Edax plays" in line or "edax plays" in line.lower():
                return line.split()[-1].strip()
            if "*** Game Over ***" in line:
                return "pass"
        raise TimeoutError("Timed out waiting for Edax move")

    def new_game(self) -> None:
        with self._lock:
            if self.proc is None or self.proc.poll() is not None:
                self._start()
            else:
                self._send("init")

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.poll() is None and self.proc.stdin:
                self.proc.stdin.write("quit\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=2)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        self.proc = None

    def select_action(self, env: OthelloEnv) -> int:
        with self._lock:
            if self.proc is None or self.proc.poll() is not None:
                self._start()
            legal = env.legal_actions()
            if legal == [PASS_ACTION]:
                return PASS_ACTION
            self._send("setboard " + board_to_setboard(env))
            self._send("go")
            token = self._read_until_plays()
            action = edax_to_idx(token)
            if action not in legal:
                raise RuntimeError(f"Edax played {token} ({action}); legal={legal}")
            return action

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def pro_agent(level: int = 15) -> EdaxAgent:
    n = os.cpu_count() or 2
    return EdaxAgent(level=level, n_tasks=max(1, min(4, n)))
