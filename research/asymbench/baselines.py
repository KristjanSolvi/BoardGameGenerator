from __future__ import annotations

import random
from collections import Counter
from typing import Callable


class RandomAgent:
    name = "random"

    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def choose(self, game, state, player: int) -> int:
        return self.rng.choice(game.legal_actions(state))


def play_game(game, agents: dict[int, object], seed: int, seat_roles=(0, 1)) -> dict:
    del seed
    state = game.initial_state(seat_roles=seat_roles)
    while not game.is_terminal(state):
        player = game.current_player(state)
        action = agents[player].choose(game, state, player)
        state = game.apply_action(state, action)
    result = game.result(state)
    winner_role = (
        None if result.winner is None else game.player_role(state, result.winner)
    )
    return {
        "winner": result.winner,
        "winner_role": winner_role,
        "reason": result.reason,
        "plies": result.plies,
        "seat_roles": list(seat_roles),
    }


def evaluate_matchup(
    game,
    agent_factories: dict[int, Callable[[int], object]],
    games: int,
    seed: int,
    seat_roles_list=((0, 1), (1, 0)),
) -> dict:
    if games <= 0:
        raise ValueError("games must be positive")

    outcomes = []
    for i in range(games):
        seat_roles = seat_roles_list[i % len(seat_roles_list)]
        agents = {
            0: agent_factories[0](seed ^ (i * 7919) ^ 0xA0),
            1: agent_factories[1](seed ^ (i * 7919) ^ 0xB1),
        }
        outcomes.append(play_game(game, agents, seed=seed + i, seat_roles=seat_roles))

    winner_counts = Counter(g["winner"] for g in outcomes)
    role_counts = Counter(g["winner_role"] for g in outcomes)
    reason_counts = Counter(g["reason"] for g in outcomes)
    return {
        "games": games,
        "p0_win_rate": round(winner_counts[0] / games, 3),
        "p1_win_rate": round(winner_counts[1] / games, 3),
        "draw_rate": round(winner_counts[None] / games, 3),
        "first_player_win_rate": round(winner_counts[0] / games, 3),
        "role_win_rates": {
            str(role): round(role_counts[role] / games, 3)
            for role in range(len(game.roles))
        },
        "termination_reasons": dict(reason_counts),
        "avg_plies": round(sum(g["plies"] for g in outcomes) / games, 2),
        "outcomes": outcomes,
    }
