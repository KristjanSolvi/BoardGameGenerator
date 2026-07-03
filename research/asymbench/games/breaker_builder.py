from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from research.asymbench.games.base import (
    IllegalActionError,
    RoleResult,
    make_action_mask,
)


@dataclass(frozen=True)
class BreakerBuilderState:
    board: tuple[int, ...]
    to_move: int
    seat_roles: tuple[int, int]
    plies: int


class BreakerBuilder:
    """Tiny connection/blocking game with asymmetric action types."""

    BUILDER = 0
    BREAKER = 1

    EMPTY = 0
    BUILDER_MARKER = 1
    BREAKER_BLOCKER = 2

    name = "breaker_builder"
    roles = ("builder", "breaker")
    board_shape = (5, 5)
    action_size = 25 + 25 * 25 + 25 * 25

    _BOARD_SIZE = 5
    _CELL_COUNT = _BOARD_SIZE * _BOARD_SIZE
    _MOVE_OFFSET = _CELL_COUNT
    _REMOVE_OFFSET = _MOVE_OFFSET + _CELL_COUNT * _CELL_COUNT
    _DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))
    _INITIAL_BLOCKERS = ("b2", "d4")

    def __init__(self, max_plies: int = 30) -> None:
        if max_plies <= 0:
            raise ValueError("max_plies must be positive")
        self.max_plies = max_plies

    def initial_state(
        self, seat_roles: tuple[int, int] = (BUILDER, BREAKER)
    ) -> BreakerBuilderState:
        if len(seat_roles) != 2 or set(seat_roles) != {self.BUILDER, self.BREAKER}:
            raise ValueError("seat_roles must assign builder and breaker once each")
        normalized_seat_roles = tuple(seat_roles)

        board = [self.EMPTY] * self._CELL_COUNT
        for cell in self._INITIAL_BLOCKERS:
            board[self.cell_index(cell)] = self.BREAKER_BLOCKER
        return BreakerBuilderState(
            board=tuple(board), to_move=0, seat_roles=normalized_seat_roles, plies=0
        )

    def current_player(self, state: BreakerBuilderState) -> int:
        return state.to_move

    def player_role(self, state: BreakerBuilderState, player: int) -> int:
        if player not in (0, 1):
            raise ValueError(f"player must be 0 or 1: {player}")
        return state.seat_roles[player]

    def encode_place(self, cell: str) -> int:
        return self.cell_index(cell)

    def encode_move(self, from_cell: str, to_cell: str) -> int:
        return (
            self._MOVE_OFFSET
            + self.cell_index(from_cell) * self._CELL_COUNT
            + self.cell_index(to_cell)
        )

    def encode_remove(self, from_cell: str, target_cell: str) -> int:
        return (
            self._REMOVE_OFFSET
            + self.cell_index(from_cell) * self._CELL_COUNT
            + self.cell_index(target_cell)
        )

    def builder_cells(self, state: BreakerBuilderState) -> list[str]:
        return [
            self.index_cell(index)
            for index, piece in enumerate(state.board)
            if piece == self.BUILDER_MARKER
        ]

    def legal_actions(self, state: BreakerBuilderState) -> list[int]:
        if self._has_static_terminal_condition(state):
            return []

        role = self.player_role(state, state.to_move)
        if role == self.BUILDER:
            return [
                index
                for index, piece in enumerate(state.board)
                if piece == self.EMPTY
            ]

        actions: list[int] = []
        for from_index, piece in enumerate(state.board):
            if piece != self.BREAKER_BLOCKER:
                continue
            for to_index in self._orthogonal_neighbors(from_index):
                target_piece = state.board[to_index]
                if target_piece == self.EMPTY:
                    actions.append(
                        self._MOVE_OFFSET + from_index * self._CELL_COUNT + to_index
                    )
                elif target_piece == self.BUILDER_MARKER:
                    actions.append(
                        self._REMOVE_OFFSET + from_index * self._CELL_COUNT + to_index
                    )
        return actions

    def apply_action(
        self, state: BreakerBuilderState, action: int
    ) -> BreakerBuilderState:
        if action not in self.legal_actions(state):
            raise IllegalActionError(
                f"illegal action {action} for player {state.to_move}"
            )

        board = list(state.board)
        role = self.player_role(state, state.to_move)
        if role == self.BUILDER:
            board[action] = self.BUILDER_MARKER
        elif action < self._REMOVE_OFFSET:
            from_index, to_index = divmod(action - self._MOVE_OFFSET, self._CELL_COUNT)
            board[from_index] = self.EMPTY
            board[to_index] = self.BREAKER_BLOCKER
        else:
            _, target_index = divmod(action - self._REMOVE_OFFSET, self._CELL_COUNT)
            board[target_index] = self.EMPTY

        return BreakerBuilderState(
            board=tuple(board),
            to_move=1 - state.to_move,
            seat_roles=state.seat_roles,
            plies=state.plies + 1,
        )

    def is_terminal(self, state: BreakerBuilderState) -> bool:
        return self._has_static_terminal_condition(state) or not self.legal_actions(
            state
        )

    def result(self, state: BreakerBuilderState) -> RoleResult:
        if self._builder_has_connection(state.board):
            return RoleResult(
                winner=self._player_with_role(state, self.BUILDER),
                reason="builder_connection",
                plies=state.plies,
            )
        if state.plies >= self.max_plies:
            return RoleResult(
                winner=self._player_with_role(state, self.BREAKER),
                reason="max_plies",
                plies=state.plies,
            )
        if not self.legal_actions(state):
            return RoleResult(
                winner=1 - state.to_move,
                reason="no_legal_actions",
                plies=state.plies,
            )
        return RoleResult(winner=None, reason="ongoing", plies=state.plies)

    def observation_tensor(
        self, state: BreakerBuilderState, player: int
    ) -> np.ndarray:
        board = np.asarray(state.board, dtype=np.int8).reshape(self.board_shape)
        role = self.player_role(state, player)

        builders = board == self.BUILDER_MARKER
        blockers = board == self.BREAKER_BLOCKER
        if role == self.BUILDER:
            own = builders
            enemy = blockers
        else:
            own = blockers
            enemy = builders

        planes = np.zeros((6, *self.board_shape), dtype=np.float32)
        planes[0] = builders
        planes[1] = blockers
        planes[2] = own
        planes[3] = enemy
        planes[4].fill(1.0 if player == state.to_move else 0.0)
        planes[5].fill(float(role))
        return planes

    def action_mask(self, state: BreakerBuilderState) -> np.ndarray:
        return make_action_mask(self.action_size, self.legal_actions(state))

    def render(self, state: BreakerBuilderState) -> str:
        piece_chars = {
            self.EMPTY: ".",
            self.BUILDER_MARKER: "M",
            self.BREAKER_BLOCKER: "X",
        }
        rows = []
        for row in range(self._BOARD_SIZE - 1, -1, -1):
            cells = [
                piece_chars[state.board[row * self._BOARD_SIZE + col]]
                for col in range(self._BOARD_SIZE)
            ]
            rows.append(f"{row + 1} " + " ".join(cells))
        rows.append("  a b c d e")
        role = self.roles[self.player_role(state, state.to_move)]
        rows.append(f"to_move={state.to_move} role={role}")
        return "\n".join(rows)

    @classmethod
    def cell_index(cls, cell: str) -> int:
        if not isinstance(cell, str) or len(cell) != 2:
            raise ValueError(f"cell must be in a1..e5 form: {cell!r}")

        file_char, rank_char = cell
        if not file_char.isalpha() or not rank_char.isdigit():
            raise ValueError(f"cell must be in a1..e5 form: {cell!r}")

        file_index = ord(file_char.lower()) - ord("a")
        rank_index = int(rank_char) - 1
        if not cls._in_bounds(rank_index, file_index):
            raise ValueError(f"cell outside board: {cell!r}")
        return rank_index * cls._BOARD_SIZE + file_index

    @classmethod
    def index_cell(cls, index: int) -> str:
        if index < 0 or index >= cls._CELL_COUNT:
            raise ValueError(f"cell index outside board: {index}")
        row, col = divmod(index, cls._BOARD_SIZE)
        return f"{chr(ord('a') + col)}{row + 1}"

    @classmethod
    def _in_bounds(cls, row: int, col: int) -> bool:
        return 0 <= row < cls._BOARD_SIZE and 0 <= col < cls._BOARD_SIZE

    def _has_static_terminal_condition(self, state: BreakerBuilderState) -> bool:
        return self._builder_has_connection(state.board) or state.plies >= self.max_plies

    def _builder_has_connection(self, board: tuple[int, ...]) -> bool:
        frontier = deque(
            index
            for index, piece in enumerate(board)
            if piece == self.BUILDER_MARKER and index % self._BOARD_SIZE == 0
        )
        visited = set(frontier)
        while frontier:
            index = frontier.popleft()
            if index % self._BOARD_SIZE == self._BOARD_SIZE - 1:
                return True
            for neighbor in self._orthogonal_neighbors(index):
                if neighbor in visited or board[neighbor] != self.BUILDER_MARKER:
                    continue
                visited.add(neighbor)
                frontier.append(neighbor)
        return False

    def _orthogonal_neighbors(self, index: int) -> list[int]:
        row, col = divmod(index, self._BOARD_SIZE)
        neighbors = []
        for row_delta, col_delta in self._DIRECTIONS:
            neighbor_row = row + row_delta
            neighbor_col = col + col_delta
            if self._in_bounds(neighbor_row, neighbor_col):
                neighbors.append(neighbor_row * self._BOARD_SIZE + neighbor_col)
        return neighbors

    def _player_with_role(self, state: BreakerBuilderState, role: int) -> int:
        return state.seat_roles.index(role)
