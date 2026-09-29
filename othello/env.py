"""Classic 8x8 Othello rules engine."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

BOARD_SIZE = 8
N_SQUARES = BOARD_SIZE * BOARD_SIZE
PASS_ACTION = N_SQUARES
ACTION_SIZE = N_SQUARES + 1  # 64 placements + pass

EMPTY = 0
BLACK = 1
WHITE = -1

DIRECTIONS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def idx_to_rc(action: int) -> Tuple[int, int]:
    return divmod(action, BOARD_SIZE)


def rc_to_idx(row: int, col: int) -> int:
    return row * BOARD_SIZE + col


def opponent(player: int) -> int:
    return -player


class OthelloEnv:
    """Single-game Othello state machine."""

    def __init__(self) -> None:
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.to_play = BLACK
        self.done = False
        self.winner: Optional[int] = None  # BLACK, WHITE, or 0 for draw
        self.pass_streak = 0
        self.move_count = 0
        self.reset()

    def reset(self) -> np.ndarray:
        self.board.fill(EMPTY)
        mid = BOARD_SIZE // 2
        self.board[mid - 1, mid - 1] = WHITE
        self.board[mid - 1, mid] = BLACK
        self.board[mid, mid - 1] = BLACK
        self.board[mid, mid] = WHITE
        self.to_play = BLACK
        self.done = False
        self.winner = None
        self.pass_streak = 0
        self.move_count = 0
        return self.observe()

    def clone(self) -> "OthelloEnv":
        env = OthelloEnv.__new__(OthelloEnv)
        env.board = self.board.copy()
        env.to_play = self.to_play
        env.done = self.done
        env.winner = self.winner
        env.pass_streak = self.pass_streak
        env.move_count = self.move_count
        return env

    def observe(self) -> np.ndarray:
        """Return a 4x8x8 float32 tensor from the current player's view."""
        own = (self.board == self.to_play).astype(np.float32)
        opp = (self.board == opponent(self.to_play)).astype(np.float32)
        legal = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        for a in self.legal_actions():
            if a != PASS_ACTION:
                r, c = idx_to_rc(a)
                legal[r, c] = 1.0
        empty_frac = float(np.mean(self.board == EMPTY))
        phase = np.full((BOARD_SIZE, BOARD_SIZE), empty_frac, dtype=np.float32)
        return np.stack([own, opp, legal, phase], axis=0)

    def legal_actions(self) -> List[int]:
        if self.done:
            return []
        moves: List[int] = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r, c] == EMPTY and self._would_flip(r, c, self.to_play):
                    moves.append(rc_to_idx(r, c))
        if not moves:
            moves.append(PASS_ACTION)
        return moves

    def legal_mask(self) -> np.ndarray:
        mask = np.zeros(ACTION_SIZE, dtype=np.bool_)
        for a in self.legal_actions():
            mask[a] = True
        return mask

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        if self.done:
            raise RuntimeError("Game already finished")

        legal = self.legal_actions()
        if action not in legal:
            raise ValueError(f"Illegal action {action}; legal={legal}")

        reward = 0.0
        info: dict = {"passed": False, "flipped": 0}

        if action == PASS_ACTION:
            self.pass_streak += 1
            info["passed"] = True
        else:
            self.pass_streak = 0
            row, col = idx_to_rc(action)
            flipped = self._apply_move(row, col, self.to_play)
            info["flipped"] = flipped
            self.move_count += 1

        self.to_play = opponent(self.to_play)

        if self.pass_streak >= 2 or np.all(self.board != EMPTY):
            self.done = True
            black_count = int(np.sum(self.board == BLACK))
            white_count = int(np.sum(self.board == WHITE))
            if black_count > white_count:
                self.winner = BLACK
            elif white_count > black_count:
                self.winner = WHITE
            else:
                self.winner = 0
            actor = opponent(self.to_play)
            if self.winner == 0:
                reward = 0.0
            elif self.winner == actor:
                reward = 1.0
            else:
                reward = -1.0
            info["black"] = black_count
            info["white"] = white_count
            info["winner"] = self.winner

        return self.observe(), reward, self.done, info

    def score(self) -> Tuple[int, int]:
        return int(np.sum(self.board == BLACK)), int(np.sum(self.board == WHITE))

    def render_ascii(self) -> str:
        cols = "  " + " ".join(str(c) for c in range(BOARD_SIZE))
        lines = [cols]
        for r in range(BOARD_SIZE):
            cells = []
            for c in range(BOARD_SIZE):
                v = self.board[r, c]
                if v == BLACK:
                    cells.append("●")
                elif v == WHITE:
                    cells.append("○")
                else:
                    cells.append(".")
            lines.append(f"{r} " + " ".join(cells))
        side = "Black (●)" if self.to_play == BLACK else "White (○)"
        b, w = self.score()
        lines.append(f"Score  Black={b}  White={w}  |  to play: {side}")
        return "\n".join(lines)

    def _would_flip(self, row: int, col: int, player: int) -> bool:
        opp = opponent(player)
        for dr, dc in DIRECTIONS:
            r, c = row + dr, col + dc
            seen_opp = False
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                if self.board[r, c] == opp:
                    seen_opp = True
                elif self.board[r, c] == player:
                    if seen_opp:
                        return True
                    break
                else:
                    break
                r += dr
                c += dc
        return False

    def _apply_move(self, row: int, col: int, player: int) -> int:
        opp = opponent(player)
        self.board[row, col] = player
        flipped = 0
        for dr, dc in DIRECTIONS:
            path: List[Tuple[int, int]] = []
            r, c = row + dr, col + dc
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                if self.board[r, c] == opp:
                    path.append((r, c))
                elif self.board[r, c] == player:
                    for pr, pc in path:
                        self.board[pr, pc] = player
                        flipped += 1
                    break
                else:
                    break
                r += dr
                c += dc
        return flipped


def result_for_player(winner: Optional[int], player: int) -> float:
    """Map terminal winner to reward for `player`: +1 / -1 / 0."""
    if winner is None or winner == 0:
        return 0.0
    return 1.0 if winner == player else -1.0
