from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class IllegalActionError(ValueError):
    """Raised when an action id is not legal in a state."""


@dataclass(frozen=True)
class RoleResult:
    winner: int | None
    reason: str
    plies: int

    def value_for_player(self, player: int) -> float:
        if self.winner is None:
            return 0.0
        return 1.0 if self.winner == player else -1.0


class AsymGame(Protocol):
    name: str
    roles: tuple[str, str]
    board_shape: tuple[int, int]
    action_size: int

    def initial_state(self, seat_roles: tuple[int, int] = (0, 1)): ...
    def current_player(self, state) -> int: ...
    def player_role(self, state, player: int) -> int: ...
    def legal_actions(self, state) -> list[int]: ...
    def apply_action(self, state, action: int): ...
    def is_terminal(self, state) -> bool: ...
    def result(self, state) -> RoleResult: ...
    def observation_tensor(self, state, player: int) -> np.ndarray: ...
    def action_mask(self, state) -> np.ndarray: ...
    def render(self, state) -> str: ...


def make_action_mask(action_size: int, legal_actions: list[int]) -> np.ndarray:
    mask = np.zeros(action_size, dtype=np.bool_)
    for action in legal_actions:
        if action < 0 or action >= action_size:
            raise ValueError(
                f"legal action {action} outside action space 0..{action_size - 1}"
            )
        mask[action] = True
    return mask
