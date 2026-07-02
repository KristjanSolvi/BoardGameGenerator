"""The fixed interface every generated game engine must implement.

Generated engines subclass GameEngine (they import nothing from this
package at runtime; the validator checks method presence structurally, so
a generated module is self-contained and this file is the reference
contract).

Conventions (enforced by the validator):
  * Players are 0 and 1 (the spec's two asymmetric roles). Player 0
    always moves first.
  * States are immutable and hashable (nested tuples / frozen dataclasses)
    and encode whose turn it is plus anything repetition rules need:
    hash(state) is the repetition key.
  * Moves are hashable tuples of primitives; repr(move) is used in logs.
  * apply() is pure: it returns a NEW state and raises IllegalMoveError
    (any exception type named 'IllegalMoveError', or ValueError) for any
    move not in legal_moves(state, current_player(state)).
  * No I/O, no randomness, no globals mutated.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Hashable

State = Hashable
Move = Hashable


class IllegalMoveError(ValueError):
    """Raised by apply() for any move that is not currently legal."""


class GameEngine(ABC):
    @abstractmethod
    def initial_state(self) -> State:
        """The starting position. Player 0 is to move."""

    @abstractmethod
    def current_player(self, state: State) -> int:
        """Whose turn it is in this state: 0 or 1."""

    @abstractmethod
    def legal_moves(self, state: State, player: int) -> list[Move]:
        """All legal moves for `player`. Empty list if it is not that
        player's turn or the state is terminal. Deterministic order."""

    @abstractmethod
    def apply(self, state: State, move: Move) -> State:
        """Return the successor state. Raises IllegalMoveError if `move`
        is not in legal_moves(state, current_player(state))."""

    @abstractmethod
    def is_terminal(self, state: State) -> bool:
        """True iff the game is over in this state."""

    @abstractmethod
    def result(self, state: State) -> dict[str, Any]:
        """Only valid when is_terminal(state). Returns
        {"winner": 0 | 1 | None, "reason": "<short string>"}
        where winner None means a draw."""

    # Required so the validator can label logs; trivial to implement.
    @abstractmethod
    def render(self, state: State) -> str:
        """ASCII rendering of the state for logs and debugging."""


REQUIRED_METHODS = (
    "initial_state",
    "current_player",
    "legal_moves",
    "apply",
    "is_terminal",
    "result",
    "render",
)
