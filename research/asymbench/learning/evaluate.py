from __future__ import annotations

from collections import Counter
from typing import Any

import torch

from research.asymbench.baselines import RandomAgent, play_game
from research.asymbench.learning.selfplay import NeuralEvaluator
from research.asymbench.search.mcts import MCTSAgent


class ModelAgent:
    name = "model"

    def __init__(
        self,
        model: torch.nn.Module,
        device: str | torch.device,
        simulations: int,
        seed: int | None = None,
    ) -> None:
        evaluator = NeuralEvaluator(model, device=device)
        self.agent = MCTSAgent(evaluator, simulations=simulations, seed=seed)

    def choose(self, game: Any, state: Any, player: int) -> int:
        return self.agent.choose(game, state, player)


def evaluate_model_vs_random(
    *,
    game: Any,
    model: torch.nn.Module,
    device: str | torch.device,
    games: int,
    simulations: int,
    seed: int,
) -> dict[str, Any]:
    if games <= 0:
        raise ValueError("games must be positive")
    if simulations <= 0:
        raise ValueError("simulations must be positive")

    was_training = model.training
    outcomes = []
    schedule = (
        (0, (0, 1)),
        (1, (1, 0)),
        (0, (1, 0)),
        (1, (0, 1)),
    )
    try:
        for index in range(games):
            model_player, seat_roles = schedule[index % len(schedule)]
            random_player = 1 - model_player
            game_seed = seed ^ (index * 7919)

            agents = {
                model_player: ModelAgent(
                    model=model,
                    device=device,
                    simulations=simulations,
                    seed=game_seed ^ 0xA0,
                ),
                random_player: RandomAgent(seed=game_seed ^ 0xB1),
            }
            outcome = play_game(
                game,
                agents,
                seed=seed + index,
                seat_roles=seat_roles,
            )
            model_won = outcome["winner"] == model_player
            outcome["model_player"] = model_player
            outcome["model_role"] = seat_roles[model_player]
            outcome["model_won"] = bool(model_won)
            outcomes.append(outcome)
    finally:
        model.train(was_training)

    winner_counts = Counter(outcome["winner"] for outcome in outcomes)
    role_counts = Counter(outcome["winner_role"] for outcome in outcomes)
    reason_counts = Counter(outcome["reason"] for outcome in outcomes)
    model_wins = sum(1 for outcome in outcomes if outcome["model_won"])
    model_role_games = Counter(outcome["model_role"] for outcome in outcomes)
    model_role_wins = Counter(
        outcome["model_role"] for outcome in outcomes if outcome["model_won"]
    )

    return {
        "games": games,
        "model_win_rate": round(model_wins / games, 3),
        "random_win_rate": round((games - model_wins - winner_counts[None]) / games, 3),
        "draw_rate": round(winner_counts[None] / games, 3),
        "role_win_rates": {
            str(role): round(role_counts[role] / games, 3)
            for role in range(len(game.roles))
        },
        "model_role_win_rates": {
            str(role): round(model_role_wins[role] / model_role_games[role], 3)
            for role in range(len(game.roles))
            if model_role_games[role] > 0
        },
        "termination_reasons": dict(reason_counts),
        "avg_plies": round(sum(outcome["plies"] for outcome in outcomes) / games, 2),
        "outcomes": outcomes,
    }


__all__ = ["ModelAgent", "evaluate_model_vs_random"]
