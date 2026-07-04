from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping


PROBABILITY_TOLERANCE = 1e-3


@dataclass(frozen=True)
class OutcomeVector:
    role0: float
    role1: float
    draw: float = 0.0

    def __post_init__(self) -> None:
        values = {
            "role0": self.role0,
            "role1": self.role1,
            "draw": self.draw,
        }
        for name, value in values.items():
            _require_probability(value, name)
        total = self.role0 + self.role1 + self.draw
        if abs(total - 1.0) > PROBABILITY_TOLERANCE:
            raise ValueError("outcome probabilities must sum to 1")


@dataclass(frozen=True)
class RoleSeatMetrics:
    role_bias: float
    seat_bias: float
    role_seat_separation: float


def evaluator_disagreement(outcomes: Mapping[str, OutcomeVector]) -> float:
    if len(outcomes) < 2:
        raise ValueError("at least two evaluator outcomes are required")

    scores = [
        0.5
        * (
            abs(left.role0 - right.role0)
            + abs(left.role1 - right.role1)
            + abs(left.draw - right.draw)
        )
        for left, right in combinations(outcomes.values(), 2)
    ]
    return max(scores)


def role_seat_separation(
    outcome: OutcomeVector,
    *,
    first_player_win_rate: float,
) -> RoleSeatMetrics:
    _require_probability(first_player_win_rate, "first_player_win_rate")
    role_bias = abs(outcome.role0 - outcome.role1)
    seat_bias = abs(2.0 * first_player_win_rate - 1.0)
    return RoleSeatMetrics(
        role_bias=role_bias,
        seat_bias=seat_bias,
        role_seat_separation=role_bias - seat_bias,
    )


def hidden_role_collapse(
    *,
    random_outcome: OutcomeVector,
    planned_outcome: OutcomeVector,
    random_min_role_win_rate: float = 0.25,
) -> float:
    _require_probability(random_min_role_win_rate, "random_min_role_win_rate")
    if min(random_outcome.role0, random_outcome.role1) < random_min_role_win_rate:
        return 0.0
    return abs(planned_outcome.role0 - planned_outcome.role1)


def role_inversion(
    *,
    random_outcome: OutcomeVector,
    planned_outcome: OutcomeVector,
    neutral_role0_win_rate: float = 0.5,
) -> float:
    _require_probability(neutral_role0_win_rate, "neutral_role0_win_rate")
    random_advantage = random_outcome.role0 - neutral_role0_win_rate
    planned_advantage = planned_outcome.role0 - neutral_role0_win_rate
    if random_advantage * planned_advantage >= 0.0:
        return 0.0
    return abs(random_outcome.role0 - planned_outcome.role0)


def architecture_delta(
    *,
    role_heads_win_rate: float,
    shared_heads_win_rate: float,
) -> float:
    _require_probability(role_heads_win_rate, "role_heads_win_rate")
    _require_probability(shared_heads_win_rate, "shared_heads_win_rate")
    return role_heads_win_rate - shared_heads_win_rate


def _require_probability(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    if value > 1.0 + PROBABILITY_TOLERANCE:
        raise ValueError(f"{name} must be at most 1")
