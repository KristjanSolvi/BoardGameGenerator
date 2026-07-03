from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from research.asymbench.learning.replay import TrainingExample
from research.asymbench.search.mcts import MCTSAgent


class NeuralEvaluator:
    """MCTS evaluator backed by a policy/value network."""

    def __init__(self, model: torch.nn.Module, device: str | torch.device) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

    def evaluate(self, game: Any, state: Any, player: int) -> tuple[np.ndarray, float]:
        if game.is_terminal(state):
            raise ValueError("cannot evaluate terminal state")

        legal_actions = game.legal_actions(state)
        if not legal_actions:
            raise ValueError("cannot evaluate state with no legal actions")

        observation = game.observation_tensor(state, player)
        role = game.player_role(state, player)
        action_mask = np.asarray(game.action_mask(state), dtype=np.bool_)
        if action_mask.shape != (game.action_size,):
            raise ValueError("game action_mask shape must match action_size")
        if not bool(action_mask.any()):
            raise ValueError("cannot evaluate state with no legal actions")

        dtype = self._model_dtype()
        observations = torch.as_tensor(
            observation, dtype=dtype, device=self.device
        ).unsqueeze(0)
        roles = torch.tensor([role], dtype=torch.long, device=self.device)
        masks = torch.as_tensor(
            action_mask, dtype=torch.bool, device=self.device
        ).unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            logits, values = self.model(observations, roles, masks)
            mask_value = torch.finfo(logits.dtype).min
            masked_logits = logits.masked_fill(~masks, mask_value)
            probabilities = torch.softmax(masked_logits, dim=1)
            total = probabilities.sum(dim=1, keepdim=True)
            probabilities = probabilities / total

        prior = probabilities.squeeze(0).detach().cpu().numpy().astype(np.float64)
        return prior, float(values.squeeze(0).detach().cpu().item())

    def _model_dtype(self) -> torch.dtype:
        try:
            parameter = next(self.model.parameters())
        except StopIteration:
            return torch.float32
        return parameter.dtype if parameter.is_floating_point() else torch.float32


@dataclass(frozen=True)
class _PendingExample:
    observation: np.ndarray
    role: int
    action_mask: np.ndarray
    policy: np.ndarray
    player: int


def generate_selfplay_game(
    *,
    game: Any,
    model: torch.nn.Module,
    device: str | torch.device = "cpu",
    simulations: int = 64,
    seed: int | None = None,
    seat_roles: tuple[int, int] = (0, 1),
    max_steps: int | None = None,
) -> tuple[list[TrainingExample], dict[str, Any]]:
    """Generate one deterministic MCTS self-play game and replay examples."""

    if max_steps is None:
        max_steps = getattr(game, "max_plies", 1000)

    evaluator = NeuralEvaluator(model, device=device)
    agent = MCTSAgent(evaluator, simulations=simulations, seed=seed)
    state = game.initial_state(seat_roles=seat_roles)
    pending: list[_PendingExample] = []
    steps = 0

    while not game.is_terminal(state):
        if steps >= max_steps:
            raise RuntimeError(f"generate_selfplay_game exceeded max_steps={max_steps}")

        player = game.current_player(state)
        legal_actions = game.legal_actions(state)
        if not legal_actions:
            raise ValueError("non-terminal state has no legal actions")

        observation = np.asarray(game.observation_tensor(state, player)).copy()
        role = int(game.player_role(state, player))
        action_mask = np.asarray(game.action_mask(state), dtype=np.bool_).copy()
        policy = np.asarray(agent.policy(game, state, player), dtype=np.float64).copy()

        action = int(np.argmax(policy))
        if action not in legal_actions:
            raise ValueError("MCTS policy selected an illegal action")

        pending.append(
            _PendingExample(
                observation=observation,
                role=role,
                action_mask=action_mask,
                policy=policy,
                player=player,
            )
        )
        state = game.apply_action(state, action)
        steps += 1

    result = game.result(state)
    examples = [
        TrainingExample(
            observation=record.observation,
            role=record.role,
            action_mask=record.action_mask,
            policy=record.policy,
            value=result.value_for_player(record.player),
        )
        for record in pending
    ]
    return examples, {
        "winner": result.winner,
        "reason": result.reason,
        "plies": result.plies,
    }


__all__ = ["NeuralEvaluator", "generate_selfplay_game"]
