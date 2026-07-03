from __future__ import annotations

from typing import Any

import numpy as np

from research.asymbench.baselines import RandomAgent, evaluate_matchup
from research.asymbench.generation.loader import compile_generated_game
from research.asymbench.generation.specs import GeneratedGameSpec, ValidationReport


def validate_generated_game(
    spec: GeneratedGameSpec,
    random_games: int = 8,
    seed: int = 0,
) -> ValidationReport:
    if random_games <= 0:
        raise ValueError("random_games must be positive")

    reasons: list[str] = []
    game: Any | None = None

    try:
        game = compile_generated_game(spec)
    except ValueError as exc:
        reason = _compile_failure_reason(exc)
        return _invalid_report(spec=spec, reasons=[reason])

    try:
        initial_state = game.initial_state()
        if game.is_terminal(initial_state):
            reasons.append("initial state is terminal")

        legal_actions = game.legal_actions(initial_state)
        if not legal_actions:
            reasons.append("initial state has no legal actions")

        action_mask = np.asarray(game.action_mask(initial_state))
        if (
            action_mask.ndim != 1
            or action_mask.shape != (game.action_size,)
            or action_mask.dtype.kind != "b"
        ):
            reasons.append("action mask does not match legal actions")
        else:
            masked_actions = [
                index for index, is_legal in enumerate(action_mask.tolist()) if is_legal
            ]
            if masked_actions != sorted(legal_actions):
                reasons.append("action mask does not match legal actions")
    except ValueError as exc:
        return _invalid_report(spec=spec, reasons=[f"compile failed: {exc}"])

    if reasons:
        return _invalid_report(
            spec=spec,
            reasons=reasons,
            initial_branching_factor=len(legal_actions),
        )

    summary = evaluate_matchup(
        game,
        agent_factories={
            0: lambda s: RandomAgent(s),
            1: lambda s: RandomAgent(s),
        },
        games=random_games,
        seed=seed,
    )

    terminal_reasons = dict(summary["termination_reasons"])
    reasons = []
    valid = True
    if terminal_reasons.get("max_plies", 0) == random_games:
        valid = False
        reasons.append("all random rollouts ended by max plies")

    return ValidationReport(
        family=spec.family,
        name=spec.name,
        valid=valid,
        reasons=tuple(reasons),
        initial_branching_factor=len(legal_actions),
        random_role_win_rates=dict(summary["role_win_rates"]),
        mcts_role_win_rates={},
        average_random_plies=summary["avg_plies"],
        terminal_reasons=terminal_reasons,
    )


def _invalid_report(
    *,
    spec: GeneratedGameSpec,
    reasons: list[str],
    initial_branching_factor: int = 0,
) -> ValidationReport:
    return ValidationReport(
        family=spec.family,
        name=spec.name,
        valid=False,
        reasons=tuple(reasons),
        initial_branching_factor=initial_branching_factor,
        random_role_win_rates={},
        mcts_role_win_rates={},
        average_random_plies=0.0,
        terminal_reasons={},
    )


def _compile_failure_reason(exc: Exception) -> str:
    message = str(exc)
    if "must not start connected" in message:
        return "initial state is terminal"
    if "key must not start captured" in message:
        return "initial state is terminal"
    if "initial state must have legal actions" in message:
        return "initial state has no legal actions"
    return f"compile failed: {message}"
