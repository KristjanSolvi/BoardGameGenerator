from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from research.asymbench.games.base import (
    IllegalActionError,
    RoleResult,
    make_action_mask,
)


@dataclass(frozen=True)
class MicroTaflState:
    board: tuple[int, ...]
    to_move: int
    seat_roles: tuple[int, int]
    plies: int


class MicroTafl:
    """Tiny 5x5 tafl-like reference game for asymmetric-role experiments."""

    ATTACKER = 0
    DEFENDER = 1

    EMPTY = 0
    ATTACKER_PIECE = 1
    DEFENDER_GUARD = 2
    KING = 3

    name = "micro_tafl"
    roles = ("attacker", "defender")
    board_shape = (5, 5)
    action_size = 25 * 25

    _BOARD_SIZE = 5
    _CORNERS = frozenset((0, 4, 20, 24))
    _DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))

    _INITIAL_ATTACKERS = ("a3", "b1", "b5", "c1", "c5", "d1", "d5", "e3")
    _INITIAL_DEFENDERS = ("b3", "c2", "c4", "d3")
    _INITIAL_KING = "c3"

    def __init__(self, max_plies: int = 80) -> None:
        self.max_plies = max_plies

    def initial_state(
        self, seat_roles: tuple[int, int] = (ATTACKER, DEFENDER)
    ) -> MicroTaflState:
        if len(seat_roles) != 2 or set(seat_roles) != {self.ATTACKER, self.DEFENDER}:
            raise ValueError("seat_roles must assign attacker and defender once each")

        board = [self.EMPTY] * self.action_size_root
        for cell in self._INITIAL_ATTACKERS:
            board[self.cell_index(cell)] = self.ATTACKER_PIECE
        for cell in self._INITIAL_DEFENDERS:
            board[self.cell_index(cell)] = self.DEFENDER_GUARD
        board[self.cell_index(self._INITIAL_KING)] = self.KING
        return MicroTaflState(
            board=tuple(board), to_move=0, seat_roles=seat_roles, plies=0
        )

    @property
    def action_size_root(self) -> int:
        return self._BOARD_SIZE * self._BOARD_SIZE

    def current_player(self, state: MicroTaflState) -> int:
        return state.to_move

    def player_role(self, state: MicroTaflState, player: int) -> int:
        return state.seat_roles[player]

    def legal_actions(self, state: MicroTaflState) -> list[int]:
        if self._has_static_terminal_condition(state):
            return []

        role = self.player_role(state, state.to_move)
        actions: list[int] = []
        for from_index, piece in enumerate(state.board):
            if not self._role_controls_piece(role, piece):
                continue
            from_row, from_col = divmod(from_index, self._BOARD_SIZE)
            for row_delta, col_delta in self._DIRECTIONS:
                row = from_row + row_delta
                col = from_col + col_delta
                while self._in_bounds(row, col):
                    to_index = row * self._BOARD_SIZE + col
                    if state.board[to_index] != self.EMPTY:
                        break
                    actions.append(from_index * self.action_size_root + to_index)
                    row += row_delta
                    col += col_delta
        return actions

    def apply_action(self, state: MicroTaflState, action: int) -> MicroTaflState:
        if action not in self.legal_actions(state):
            raise IllegalActionError(
                f"illegal action {action} for player {state.to_move}"
            )

        from_index, to_index = divmod(action, self.action_size_root)
        moving_piece = state.board[from_index]
        moving_role = self._piece_role(moving_piece)
        board = list(state.board)
        board[from_index] = self.EMPTY
        board[to_index] = moving_piece

        if moving_piece == self.KING and to_index in self._CORNERS:
            return self._next_state(state, board)

        self._capture_sandwiched_non_kings(board, to_index, moving_role)
        if moving_role == self.ATTACKER:
            king_index = self._find_king(board)
            if king_index is not None and self._is_king_captured(board, king_index):
                board[king_index] = self.EMPTY

        return self._next_state(state, board)

    def is_terminal(self, state: MicroTaflState) -> bool:
        return self._has_static_terminal_condition(state) or not self.legal_actions(
            state
        )

    def result(self, state: MicroTaflState) -> RoleResult:
        king_index = self._find_king(state.board)
        if king_index in self._CORNERS:
            return RoleResult(
                winner=self._player_with_role(state, self.DEFENDER),
                reason="king_escape",
                plies=state.plies,
            )
        if king_index is None:
            return RoleResult(
                winner=self._player_with_role(state, self.ATTACKER),
                reason="king_capture",
                plies=state.plies,
            )
        if state.plies >= self.max_plies:
            return RoleResult(winner=None, reason="max_plies", plies=state.plies)
        if not self.legal_actions(state):
            return RoleResult(
                winner=1 - state.to_move,
                reason="no_legal_actions",
                plies=state.plies,
            )
        return RoleResult(winner=None, reason="ongoing", plies=state.plies)

    def observation_tensor(self, state: MicroTaflState, player: int) -> np.ndarray:
        board = np.asarray(state.board, dtype=np.int8).reshape(self.board_shape)
        role = self.player_role(state, player)

        attackers = board == self.ATTACKER_PIECE
        defender_guards = board == self.DEFENDER_GUARD
        king = board == self.KING
        if role == self.ATTACKER:
            own = attackers
            enemy = defender_guards | king
        else:
            own = defender_guards | king
            enemy = attackers

        planes = np.zeros((7, *self.board_shape), dtype=np.float32)
        planes[0] = attackers
        planes[1] = defender_guards
        planes[2] = king
        planes[3] = own
        planes[4] = enemy
        planes[5].fill(1.0 if player == state.to_move else 0.0)
        planes[6].fill(float(role))
        return planes

    def action_mask(self, state: MicroTaflState) -> np.ndarray:
        return make_action_mask(self.action_size, self.legal_actions(state))

    def render(self, state: MicroTaflState) -> str:
        piece_chars = {
            self.EMPTY: ".",
            self.ATTACKER_PIECE: "A",
            self.DEFENDER_GUARD: "D",
            self.KING: "K",
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
        rows.append(
            f"to_move={state.to_move} role={role}"
        )
        return "\n".join(rows)

    @classmethod
    def cell_index(cls, cell: str) -> int:
        file_index = ord(cell[0]) - ord("a")
        rank_index = int(cell[1]) - 1
        if not cls._in_bounds(rank_index, file_index):
            raise ValueError(f"cell outside board: {cell}")
        return rank_index * cls._BOARD_SIZE + file_index

    @classmethod
    def _in_bounds(cls, row: int, col: int) -> bool:
        return 0 <= row < cls._BOARD_SIZE and 0 <= col < cls._BOARD_SIZE

    def _next_state(
        self, state: MicroTaflState, board: list[int]
    ) -> MicroTaflState:
        return MicroTaflState(
            board=tuple(board),
            to_move=1 - state.to_move,
            seat_roles=state.seat_roles,
            plies=state.plies + 1,
        )

    def _has_static_terminal_condition(self, state: MicroTaflState) -> bool:
        king_index = self._find_king(state.board)
        return (
            king_index is None
            or king_index in self._CORNERS
            or state.plies >= self.max_plies
        )

    def _capture_sandwiched_non_kings(
        self, board: list[int], moved_to: int, moving_role: int
    ) -> None:
        moved_row, moved_col = divmod(moved_to, self._BOARD_SIZE)
        for row_delta, col_delta in self._DIRECTIONS:
            adjacent_row = moved_row + row_delta
            adjacent_col = moved_col + col_delta
            anchor_row = moved_row + 2 * row_delta
            anchor_col = moved_col + 2 * col_delta
            if not self._in_bounds(adjacent_row, adjacent_col) or not self._in_bounds(
                anchor_row, anchor_col
            ):
                continue

            adjacent_index = adjacent_row * self._BOARD_SIZE + adjacent_col
            adjacent_piece = board[adjacent_index]
            if adjacent_piece == self.EMPTY or adjacent_piece == self.KING:
                continue
            if self._piece_role(adjacent_piece) == moving_role:
                continue

            anchor_index = anchor_row * self._BOARD_SIZE + anchor_col
            if self._is_capture_anchor(board, anchor_index, moving_role):
                board[adjacent_index] = self.EMPTY

    def _is_capture_anchor(
        self, board: list[int] | tuple[int, ...], index: int, role: int
    ) -> bool:
        piece = board[index]
        if piece != self.EMPTY and self._piece_role(piece) == role:
            return True
        return piece == self.EMPTY and index in self._CORNERS

    def _is_king_captured(self, board: list[int], king_index: int) -> bool:
        king_row, king_col = divmod(king_index, self._BOARD_SIZE)
        return (
            self._king_side_hostile(board, king_row - 1, king_col)
            and self._king_side_hostile(board, king_row + 1, king_col)
        ) or (
            self._king_side_hostile(board, king_row, king_col - 1)
            and self._king_side_hostile(board, king_row, king_col + 1)
        )

    def _king_side_hostile(self, board: list[int], row: int, col: int) -> bool:
        if not self._in_bounds(row, col):
            return False
        index = row * self._BOARD_SIZE + col
        return board[index] == self.ATTACKER_PIECE or (
            board[index] == self.EMPTY and index in self._CORNERS
        )

    def _find_king(self, board: list[int] | tuple[int, ...]) -> int | None:
        try:
            return board.index(self.KING)
        except ValueError:
            return None

    def _player_with_role(self, state: MicroTaflState, role: int) -> int:
        return state.seat_roles.index(role)

    def _role_controls_piece(self, role: int, piece: int) -> bool:
        if role == self.ATTACKER:
            return piece == self.ATTACKER_PIECE
        return piece in (self.DEFENDER_GUARD, self.KING)

    def _piece_role(self, piece: int) -> int:
        if piece == self.ATTACKER_PIECE:
            return self.ATTACKER
        if piece in (self.DEFENDER_GUARD, self.KING):
            return self.DEFENDER
        raise ValueError(f"empty cell has no role: {piece}")
