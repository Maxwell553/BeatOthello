"""1-ply positional heuristic used as an evaluation opponent."""

from __future__ import annotations

from othello.env import PASS_ACTION, OthelloEnv, opponent

# Classic square weights (corners high, X-squares heavily penalized).
WEIGHTS = (
    (100, -20, 10, 5, 5, 10, -20, 100),
    (-20, -50, -2, -2, -2, -2, -50, -20),
    (10, -2, -1, 0, 0, -1, -2, 10),
    (5, -2, 0, -1, -1, 0, -2, 5),
    (5, -2, 0, -1, -1, 0, -2, 5),
    (10, -2, -1, 0, 0, -1, -2, 10),
    (-20, -50, -2, -2, -2, -2, -50, -20),
    (100, -20, 10, 5, 5, 10, -20, 100),
)


def positional_score(env: OthelloEnv, player: int) -> float:
    board = env.board
    score = 0.0
    for r in range(8):
        for c in range(8):
            v = int(board[r, c])
            if v == player:
                score += WEIGHTS[r][c]
            elif v == opponent(player):
                score -= WEIGHTS[r][c]
    # Mobility (legal moves for player minus opponent), computed on a clone
    # of the position with the appropriate side to move.
    own_mob = _mobility_for(env, player)
    opp_mob = _mobility_for(env, opponent(player))
    score += 8.0 * (own_mob - opp_mob)
    return score


def _mobility_for(env: OthelloEnv, player: int) -> int:
    clone = env.clone()
    clone.to_play = player
    acts = clone.legal_actions()
    if acts == [PASS_ACTION]:
        return 0
    return len(acts)


class PositionalAgent:
    """Pick the legal move that maximizes positional score after the move.

    This is a 1-ply heuristic policy used as a *strong baseline opponent*,
    not as part of the neural bot.
    """

    def select_action(self, env: OthelloEnv) -> int:
        legal = env.legal_actions()
        if legal == [PASS_ACTION]:
            return PASS_ACTION
        player = env.to_play
        best_a = legal[0]
        best_s = float("-inf")
        for a in legal:
            if a == PASS_ACTION:
                continue
            clone = env.clone()
            clone.step(a)
            # After step, to_play flipped; score from original player's view
            s = positional_score(clone, player)
            if s > best_s:
                best_s = s
                best_a = a
        return best_a
