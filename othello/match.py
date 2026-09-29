"""Play matches between two agents that implement select_action(env)."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from othello.env import BLACK, WHITE, OthelloEnv


def play_game(
    black_agent: Any,
    white_agent: Any,
    record: bool = False,
) -> Dict[str, Any]:
    env = OthelloEnv()
    boards: List = []
    moves: List[int] = []
    if record:
        boards.append(env.board.copy())
    for agent in (black_agent, white_agent):
        if hasattr(agent, "new_game"):
            agent.new_game()
    while not env.done:
        agent = black_agent if env.to_play == BLACK else white_agent
        action = agent.select_action(env)
        env.step(action)
        if record:
            boards.append(env.board.copy())
            moves.append(int(action))
    b, w = env.score()
    return {
        "winner": env.winner,
        "black": b,
        "white": w,
        "boards": boards,
        "moves": moves,
    }


def play_match(
    agent_a: Any,
    agent_b: Any,
    n_games: int = 50,
    on_game: Optional[Callable[[int, Dict[str, Any]], None]] = None,
) -> dict:
    """agent_a is Black on even games, White on odd. Returns A's W/L/D stats."""
    wins = draws = losses = 0
    disc_margin = 0  # A discs minus B discs, summed
    as_black = {"wins": 0, "losses": 0, "draws": 0}
    as_white = {"wins": 0, "losses": 0, "draws": 0}
    for i in range(n_games):
        a_is_black = i % 2 == 0
        black, white = (agent_a, agent_b) if a_is_black else (agent_b, agent_a)
        result = play_game(black, white)
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
        disc_margin += a_discs - b_discs
        if on_game is not None:
            on_game(i, result)
    n = max(n_games, 1)
    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / n,
        "score": (wins + 0.5 * draws) / n,
        "avg_disc_margin": disc_margin / n,
        "games": n_games,
        "as_black": as_black,
        "as_white": as_white,
    }
