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

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                logits, values = self.model(observations, roles, masks)
                mask_value = torch.finfo(logits.dtype).min
                masked_logits = logits.masked_fill(~masks, mask_value)
                probabilities = torch.softmax(masked_logits, dim=1)
                probabilities = probabilities.masked_fill(~masks, 0.0)
        finally:
            self.model.train(was_training)

        prior = _validate_policy(
            probabilities.squeeze(0).detach().cpu().numpy(),
            action_mask,
            context="neural evaluator prior",
        )
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


def _validate_policy(
    policy: np.ndarray,
    action_mask: np.ndarray,
    *,
    context: str,
) -> np.ndarray:
    policy = np.asarray(policy, dtype=np.float64)
    action_mask = np.asarray(action_mask, dtype=np.bool_)
    if policy.shape != action_mask.shape:
        raise ValueError(f"{context} shape must match action mask")
    if not np.all(np.isfinite(policy)):
        raise ValueError(f"{context} must contain only finite values")
    if np.any(policy < -1e-8):
        raise ValueError(f"{context} must not contain negative probabilities")
    if float(np.abs(policy[~action_mask]).sum()) > 1e-8:
        raise ValueError(f"{context} has illegal action mass")

    legal_policy = np.zeros_like(policy, dtype=np.float64)
    legal_policy[action_mask] = np.maximum(policy[action_mask], 0.0)
    total = float(legal_policy.sum())
    if total <= 0.0:
        raise ValueError(f"{context} must have positive probability mass")
    if not np.isclose(total, 1.0, rtol=1e-5, atol=1e-8):
        raise ValueError(f"{context} must be normalized")

    normalized = legal_policy / total
    if not np.isclose(float(normalized.sum()), 1.0):
        raise ValueError(f"{context} could not be normalized")
    return normalized


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
        policy = _validate_policy(
            agent.policy(game, state, player),
            action_mask,
            context="MCTS policy",
        ).astype(np.float32)

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
