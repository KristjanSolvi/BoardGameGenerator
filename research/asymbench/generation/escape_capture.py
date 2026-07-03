from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from research.asymbench.games.base import (
    IllegalActionError,
    RoleResult,
    make_action_mask,
)
from research.asymbench.games.grid import (
    coord_to_index,
    in_bounds,
    index_to_coord,
    neighbors,
    replace_cell,
)
from research.asymbench.generation.specs import GeneratedGameSpec


EMPTY = 0
ATTACKER = 1
GUARD = 2
KEY = 3

ROLE_ATTACKER = 0
ROLE_DEFENDER = 1


@dataclass(frozen=True)
class EscapeCaptureState:
    board: tuple[int, ...]
    to_move: int
    seat_roles: tuple[int, int]
    plies: int
    terminal: RoleResult | None = None


class EscapeCaptureGame:
    """Generated-family escape/capture runtime with orthogonal one-step moves."""

    def __init__(self, spec: GeneratedGameSpec) -> None:
        if spec.family != "escape_capture":
            raise ValueError(f"expected escape_capture spec, got {spec.family!r}")
        if spec.actions.get("movement") != "orthogonal_step":
            raise ValueError("escape_capture only supports orthogonal_step movement")

        capture_rule = spec.terminal_rules.get("capture", "opposite_sides")
        if capture_rule != "opposite_sides":
            raise ValueError("escape_capture only supports opposite_sides capture")

        rows = self._require_positive_int(spec.board.get("rows"), "board rows")
        cols = self._require_positive_int(spec.board.get("cols"), "board cols")
        self.name = spec.name
        self.roles = tuple(spec.roles)
        self.board_shape = (rows, cols)
        self._rows = rows
        self._cols = cols
        self._cell_count = rows * cols
        self.action_size = self._cell_count * self._cell_count
        self.max_plies = spec.max_plies

        setup = spec.setup
        self._attackers = self._setup_index_sequence(setup.get("attackers", ()), "attackers")
        self._guards = self._setup_index_sequence(setup.get("guards", ()), "guards")
        self._key = self._setup_key(setup.get("key"))
        self._exits = self._setup_index_sequence(setup.get("exits", ()), "exits")
        self._hostile = self._setup_index_sequence(setup.get("hostile", ()), "hostile")
        self._validate_setup()

    def encode_move(self, from_index: int, to_index: int) -> int:
        self._validate_cell_index(from_index)
        self._validate_cell_index(to_index)
        return from_index * self._cell_count + to_index

    def decode_move(self, action: int) -> tuple[int, int]:
        if type(action) is not int or action < 0 or action >= self.action_size:
            raise ValueError(f"action outside action space: {action!r}")
        return divmod(action, self._cell_count)

    def initial_state(
        self, seat_roles: tuple[int, int] = (ROLE_ATTACKER, ROLE_DEFENDER)
    ) -> EscapeCaptureState:
        if len(seat_roles) != 2 or set(seat_roles) != {ROLE_ATTACKER, ROLE_DEFENDER}:
            raise ValueError("seat_roles must assign attacker and defender once each")
        normalized_seat_roles = tuple(seat_roles)

        board = [EMPTY] * self._cell_count
        for index in self._attackers:
            board[index] = ATTACKER
        for index in self._guards:
            board[index] = GUARD
        board[self._key] = KEY
        return EscapeCaptureState(
            board=tuple(board),
            to_move=0,
            seat_roles=normalized_seat_roles,
            plies=0,
        )

    def current_player(self, state: EscapeCaptureState) -> int:
        return state.to_move

    def player_role(self, state: EscapeCaptureState, player: int) -> int:
        if player not in (0, 1):
            raise ValueError(f"player must be 0 or 1: {player}")
        return state.seat_roles[player]

    def legal_actions(self, state: EscapeCaptureState) -> list[int]:
        if state.terminal is not None:
            return []

        role = self.player_role(state, state.to_move)
        actions: list[int] = []
        for from_index, piece in enumerate(state.board):
            if not self._role_controls_piece(role, piece):
                continue
            from_row, from_col = index_to_coord(from_index, cols=self._cols)
            for to_row, to_col in neighbors(
                from_row,
                from_col,
                rows=self._rows,
                cols=self._cols,
            ):
                to_index = coord_to_index(to_row, to_col, rows=self._rows, cols=self._cols)
                if not self._can_enter(state.board, piece, to_index):
                    continue
                actions.append(self.encode_move(from_index, to_index))
        return actions

    def apply_action(
        self, state: EscapeCaptureState, action: int
    ) -> EscapeCaptureState:
        if action not in self.legal_actions(state):
            raise IllegalActionError(
                f"illegal action {action} for player {state.to_move}"
            )

        from_index, to_index = self.decode_move(action)
        moving_piece = state.board[from_index]
        moving_role = self._piece_role(moving_piece)
        board = replace_cell(state.board, index=from_index, value=EMPTY)
        board = replace_cell(board, index=to_index, value=moving_piece)
        plies = state.plies + 1
        next_player = 1 - state.to_move

        terminal = self._terminal_after_move(
            board=board,
            moving_piece=moving_piece,
            moving_role=moving_role,
            to_index=to_index,
            state=state,
            plies=plies,
        )
        next_state = EscapeCaptureState(
            board=board,
            to_move=next_player,
            seat_roles=state.seat_roles,
            plies=plies,
            terminal=terminal,
        )
        if terminal is None and not self.legal_actions(next_state):
            terminal = RoleResult(
                winner=state.to_move,
                reason="no_legal_actions",
                plies=plies,
            )
            next_state = EscapeCaptureState(
                board=board,
                to_move=next_player,
                seat_roles=state.seat_roles,
                plies=plies,
                terminal=terminal,
            )
        return next_state

    def is_terminal(self, state: EscapeCaptureState) -> bool:
        return state.terminal is not None

    def result(self, state: EscapeCaptureState) -> RoleResult:
        if state.terminal is None:
            raise RuntimeError("result is only available for terminal states")
        return state.terminal

    def observation_tensor(
        self, state: EscapeCaptureState, player: int
    ) -> np.ndarray:
        board = np.asarray(state.board, dtype=np.int8).reshape(self.board_shape)
        role = self.player_role(state, player)

        attackers = board == ATTACKER
        guards = board == GUARD
        key = board == KEY
        exits = np.zeros(self.board_shape, dtype=np.bool_)
        for index in self._exits:
            row, col = index_to_coord(index, cols=self._cols)
            exits[row, col] = True

        if role == ROLE_ATTACKER:
            own = attackers
            enemy = guards | key
        else:
            own = guards | key
            enemy = attackers

        planes = np.zeros((8, *self.board_shape), dtype=np.float32)
        planes[0] = attackers
        planes[1] = guards
        planes[2] = key
        planes[3] = exits
        planes[4] = own
        planes[5] = enemy
        planes[6].fill(1.0 if player == state.to_move else 0.0)
        planes[7].fill(float(role))
        return planes

    def action_mask(self, state: EscapeCaptureState) -> np.ndarray:
        return make_action_mask(self.action_size, self.legal_actions(state))

    def render(self, state: EscapeCaptureState) -> str:
        rows = []
        for row in range(self._rows - 1, -1, -1):
            cells = []
            for col in range(self._cols):
                index = coord_to_index(row, col, rows=self._rows, cols=self._cols)
                cells.append(self._render_cell(state.board[index], index))
            rows.append(f"{row + 1} " + " ".join(cells))
        rows.append("  " + " ".join(chr(ord("a") + col) for col in range(self._cols)))
        role = self.roles[self.player_role(state, state.to_move)]
        rows.append(f"to_move={state.to_move} role={role} plies={state.plies}")
        if state.terminal is not None:
            rows.append(
                f"terminal winner={state.terminal.winner} reason={state.terminal.reason}"
            )
        return "\n".join(rows)

    def _terminal_after_move(
        self,
        *,
        board: tuple[int, ...],
        moving_piece: int,
        moving_role: int,
        to_index: int,
        state: EscapeCaptureState,
        plies: int,
    ) -> RoleResult | None:
        if moving_piece == KEY and to_index in self._exits:
            return RoleResult(
                winner=self._player_with_role(state, ROLE_DEFENDER),
                reason="key_escape",
                plies=plies,
            )
        if moving_role == ROLE_ATTACKER and self._is_key_captured(board):
            return RoleResult(
                winner=self._player_with_role(state, ROLE_ATTACKER),
                reason="key_capture",
                plies=plies,
            )
        if plies >= self.max_plies:
            return RoleResult(winner=None, reason="max_plies", plies=plies)
        return None

    def _is_key_captured(self, board: tuple[int, ...]) -> bool:
        try:
            key_index = board.index(KEY)
        except ValueError:
            return False

        key_row, key_col = index_to_coord(key_index, cols=self._cols)
        return any(
            self._capture_support(board, key_row + dr, key_col + dc)
            and self._capture_support(board, key_row - dr, key_col - dc)
            for dr, dc in ((1, 0), (0, 1))
        )

    def _capture_support(
        self, board: tuple[int, ...], row: int, col: int
    ) -> bool:
        if not in_bounds(row, col, rows=self._rows, cols=self._cols):
            return True
        index = coord_to_index(row, col, rows=self._rows, cols=self._cols)
        return board[index] == ATTACKER or index in self._hostile

    def _can_enter(
        self, board: tuple[int, ...], moving_piece: int, to_index: int
    ) -> bool:
        if board[to_index] != EMPTY:
            return False
        if to_index in self._exits:
            return moving_piece == KEY
        return True

    def _role_controls_piece(self, role: int, piece: int) -> bool:
        if role == ROLE_ATTACKER:
            return piece == ATTACKER
        return piece in (GUARD, KEY)

    def _piece_role(self, piece: int) -> int:
        if piece == ATTACKER:
            return ROLE_ATTACKER
        if piece in (GUARD, KEY):
            return ROLE_DEFENDER
        raise ValueError(f"empty cell has no role: {piece}")

    def _player_with_role(self, state: EscapeCaptureState, role: int) -> int:
        return state.seat_roles.index(role)

    def _render_cell(self, piece: int, index: int) -> str:
        if piece == ATTACKER:
            return "A"
        if piece == GUARD:
            return "G"
        if piece == KEY:
            return "K"
        if index in self._exits and index in self._hostile:
            return "X"
        if index in self._exits:
            return "E"
        if index in self._hostile:
            return "H"
        return "."

    def _validate_setup(self) -> None:
        occupied = self._attackers + self._guards + (self._key,)
        if len(set(occupied)) != len(occupied):
            raise ValueError("occupied setup cells must not collide")
        exits_and_hostile = self._exits + self._hostile
        for index in exits_and_hostile:
            if index in occupied:
                raise ValueError("exits and hostile cells must not overlap occupied cells")

    def _setup_key(self, value: Any) -> int:
        if value is None:
            raise ValueError("setup key must exist exactly once")
        key = self._require_int(value, "key")
        self._validate_cell_index(key)
        return key

    def _setup_index_sequence(self, value: Any, field_name: str) -> tuple[int, ...]:
        if type(value) not in (tuple, list):
            raise ValueError(f"setup {field_name} must be a sequence of indices")
        indices = tuple(self._require_int(index, field_name) for index in value)
        if len(set(indices)) != len(indices):
            raise ValueError(f"setup {field_name} indices must be unique")
        for index in indices:
            self._validate_cell_index(index)
        return indices

    def _validate_cell_index(self, index: int) -> None:
        if type(index) is not int or index < 0 or index >= self._cell_count:
            raise ValueError(f"cell index outside board: {index!r}")

    @staticmethod
    def _require_int(value: Any, field_name: str) -> int:
        if type(value) is not int:
            raise ValueError(f"{field_name} must be an int")
        return value

    @classmethod
    def _require_positive_int(cls, value: Any, field_name: str) -> int:
        number = cls._require_int(value, field_name)
        if number <= 0:
            raise ValueError(f"{field_name} must be positive")
        return number
