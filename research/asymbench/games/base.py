from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar

import numpy as np


StateT = TypeVar("StateT")


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


class AsymGame(Protocol[StateT]):
    name: str
    roles: tuple[str, str]
    board_shape: tuple[int, int]
    action_size: int

    def initial_state(self, seat_roles: tuple[int, int] = (0, 1)) -> StateT: ...
    def current_player(self, state: StateT) -> int: ...
    def player_role(self, state: StateT, player: int) -> int: ...
    def legal_actions(self, state: StateT) -> list[int]: ...
    def apply_action(self, state: StateT, action: int) -> StateT: ...
    def is_terminal(self, state: StateT) -> bool: ...
    def result(self, state: StateT) -> RoleResult: ...
    def observation_tensor(self, state: StateT, player: int) -> np.ndarray: ...
    def action_mask(self, state: StateT) -> np.ndarray: ...
    def render(self, state: StateT) -> str: ...


def make_action_mask(action_size: int, legal_actions: list[int]) -> np.ndarray:
    mask = np.zeros(action_size, dtype=np.bool_)
    for action in legal_actions:
        if action < 0 or action >= action_size:
            raise ValueError(
                f"legal action {action} outside action space 0..{action_size - 1}"
            )
        mask[action] = True
    return mask
