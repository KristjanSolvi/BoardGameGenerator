from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


class Evaluator(Protocol):
    def evaluate(self, game: Any, state: Any, player: int) -> tuple[np.ndarray, float]:
        """Return action priors and a value from the given player's perspective."""


class UniformEvaluator:
    """Uniform legal-action priors with a neutral value estimate."""

    def evaluate(self, game: Any, state: Any, player: int) -> tuple[np.ndarray, float]:
        del player
        prior = np.zeros(game.action_size, dtype=np.float64)
        legal_actions = game.legal_actions(state)
        if legal_actions:
            prior[legal_actions] = 1.0 / len(legal_actions)
        return prior, 0.0


@dataclass
class _Node:
    state: Any
    prior: float
    player_to_move: int
    children: dict[int, "_Node"] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0

    @property
    def mean_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


class MCTSAgent:
    name = "mcts"

    def __init__(
        self,
        evaluator: Evaluator | None = None,
        simulations: int = 64,
        c_puct: float = 1.5,
        seed: int | None = None,
    ) -> None:
        if simulations <= 0:
            raise ValueError("simulations must be positive")
        if c_puct < 0.0:
            raise ValueError("c_puct must be non-negative")
        self.evaluator = evaluator if evaluator is not None else UniformEvaluator()
        self.simulations = simulations
        self.c_puct = c_puct
        self.rng = np.random.default_rng(seed)

    def policy(self, game: Any, state: Any, player: int) -> np.ndarray:
        current_player = game.current_player(state)
        if player != current_player:
            raise ValueError("policy player must be the current player")

        root = _Node(
            state=state,
            prior=1.0,
            player_to_move=current_player,
        )

        for _ in range(self.simulations):
            node = root
            path = [node]

            while node.children and not game.is_terminal(node.state):
                _, node = self._select_child(node)
                path.append(node)

            if game.is_terminal(node.state):
                value = self._validate_value(
                    game.result(node.state).value_for_player(node.player_to_move)
                )
            else:
                prior, value = self.evaluator.evaluate(
                    game, node.state, node.player_to_move
                )
                value = self._validate_value(value)
                self._expand(game, node, prior)

            self._backup(path, value)

        return self._root_policy(game, root)

    def choose(self, game: Any, state: Any, player: int) -> int:
        if not game.legal_actions(state):
            raise ValueError("no legal actions")
        policy = self.policy(game, state, player)
        if float(policy.sum()) <= 0.0:
            raise ValueError("policy has no probability mass")
        return int(np.argmax(policy))

    def _expand(self, game: Any, node: _Node, prior: np.ndarray) -> None:
        legal_actions = game.legal_actions(node.state)
        if not legal_actions:
            return

        legal_prior = self._legal_prior(game, prior, legal_actions)
        for action in legal_actions:
            child_state = game.apply_action(node.state, action)
            node.children[action] = _Node(
                state=child_state,
                prior=float(legal_prior[action]),
                player_to_move=game.current_player(child_state),
            )

    def _legal_prior(
        self, game: Any, prior: np.ndarray, legal_actions: list[int]
    ) -> np.ndarray:
        prior = np.asarray(prior, dtype=np.float64)
        if prior.shape != (game.action_size,):
            raise ValueError(
                f"evaluator prior shape {prior.shape} does not match "
                f"action_size {game.action_size}"
            )
        if not np.all(np.isfinite(prior)):
            raise ValueError("evaluator prior must contain only finite values")

        legal_prior = np.zeros(game.action_size, dtype=np.float64)
        legal_values = np.maximum(prior[legal_actions], 0.0)
        total = float(legal_values.sum())
        if total > 0.0:
            legal_prior[legal_actions] = legal_values / total
        else:
            legal_prior[legal_actions] = 1.0 / len(legal_actions)
        return legal_prior

    def _select_child(self, node: _Node) -> tuple[int, _Node]:
        best_score = -math.inf
        best: list[tuple[int, _Node]] = []
        exploration_base = math.sqrt(node.visits + 1)

        for action, child in node.children.items():
            q_value = -child.mean_value
            exploration = (
                self.c_puct * child.prior * exploration_base / (1 + child.visits)
            )
            score = q_value + exploration
            if score > best_score:
                best_score = score
                best = [(action, child)]
            elif math.isclose(score, best_score, rel_tol=1e-12, abs_tol=1e-12):
                best.append((action, child))

        return best[int(self.rng.integers(len(best)))]

    def _backup(self, path: list[_Node], value: float) -> None:
        for node in reversed(path):
            node.visits += 1
            node.value_sum += value
            value = -value

    def _root_policy(self, game: Any, root: _Node) -> np.ndarray:
        policy = np.zeros(game.action_size, dtype=np.float64)
        if not root.children:
            return policy

        for action, child in root.children.items():
            policy[action] = child.visits

        total = float(policy.sum())
        if total > 0.0:
            return policy / total

        for action, child in root.children.items():
            policy[action] = child.prior
        total = float(policy.sum())
        if total > 0.0:
            return policy / total
        return policy

    def _validate_value(self, value: float) -> float:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("evaluator value must be finite")
        return value


__all__ = ["MCTSAgent", "UniformEvaluator"]
