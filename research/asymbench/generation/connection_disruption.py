from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

import numpy as np

from research.asymbench.games.base import (
    IllegalActionError,
    RoleResult,
    make_action_mask,
)
from research.asymbench.games.grid import (
    coord_to_index,
    edge_cells,
    index_to_coord,
    neighbors,
    path_exists_between_edges,
    replace_cell,
)
from research.asymbench.generation.specs import (
    GeneratedGameSpec,
    GenerationConstraints,
)


EMPTY = 0
BUILDER = 1
BLOCKER = 2

ROLE_BUILDER = 0
ROLE_BREAKER = 1


class _CandidateRejected(Exception):
    """Expected rejection for sampled generator layouts."""


@dataclass(frozen=True)
class ConnectionDisruptionState:
    board: tuple[int, ...]
    to_move: int
    seat_roles: tuple[int, int]
    plies: int
    terminal: RoleResult | None = None


class ConnectionDisruptionGame:
    """Generated-family connection game with mobile disruptive blockers."""

    def __init__(self, spec: GeneratedGameSpec) -> None:
        if spec.family != "connection_disruption":
            raise ValueError(
                f"expected connection_disruption spec, got {spec.family!r}"
            )
        if spec.roles != ("builder", "breaker"):
            raise ValueError(
                "connection_disruption roles must be ('builder', 'breaker')"
            )
        if spec.actions.get("builder") != "place":
            raise ValueError("connection_disruption builder only supports place")
        breaker_actions = spec.actions.get("breaker")
        if type(breaker_actions) not in (tuple, list) or tuple(breaker_actions) != (
            "orthogonal_step",
            "adjacent_remove",
        ):
            raise ValueError(
                "connection_disruption breaker actions must be "
                "('orthogonal_step', 'adjacent_remove')"
            )
        connect = spec.terminal_rules.get("connect")
        if type(connect) not in (tuple, list) or tuple(connect) not in (
            ("west", "east"),
            ("east", "west"),
            ("north", "south"),
            ("south", "north"),
        ):
            raise ValueError(
                "connection_disruption connect rule must be opposite board edges"
            )

        rows = self._require_positive_int(spec.board.get("rows"), "board rows")
        cols = self._require_positive_int(spec.board.get("cols"), "board cols")
        self.name = spec.name
        self.roles = tuple(spec.roles)
        self.board_shape = (rows, cols)
        self._rows = rows
        self._cols = cols
        self._cell_count = rows * cols
        self._move_offset = self._cell_count
        self._remove_offset = self._move_offset + self._cell_count * self._cell_count
        self.action_size = self._remove_offset + self._cell_count * self._cell_count
        self.max_plies = spec.max_plies
        self._start_edge, self._target_edge = tuple(connect)

        setup = spec.setup
        self._blockers = self._setup_index_sequence(setup.get("blockers", ()), "blockers")
        self._protected = self._setup_index_sequence(
            setup.get("protected", ()), "protected"
        )
        self._builders = self._setup_index_sequence(setup.get("builders", ()), "builders")
        self._validate_setup()

    def encode_place(self, target: int) -> int:
        self._validate_cell_index(target)
        return target

    def encode_move(self, from_index: int, to_index: int) -> int:
        self._validate_cell_index(from_index)
        self._validate_cell_index(to_index)
        return self._move_offset + from_index * self._cell_count + to_index

    def encode_remove(self, from_index: int, target_index: int) -> int:
        self._validate_cell_index(from_index)
        self._validate_cell_index(target_index)
        return self._remove_offset + from_index * self._cell_count + target_index

    def decode_place(self, action: int) -> int:
        if type(action) is not int or action < 0 or action >= self._cell_count:
            raise ValueError(f"place action outside action space: {action!r}")
        return action

    def decode_move(self, action: int) -> tuple[int, int]:
        if (
            type(action) is not int
            or action < self._move_offset
            or action >= self._remove_offset
        ):
            raise ValueError(f"move action outside action space: {action!r}")
        return divmod(action - self._move_offset, self._cell_count)

    def decode_remove(self, action: int) -> tuple[int, int]:
        if (
            type(action) is not int
            or action < self._remove_offset
            or action >= self.action_size
        ):
            raise ValueError(f"remove action outside action space: {action!r}")
        return divmod(action - self._remove_offset, self._cell_count)

    def initial_state(
        self, seat_roles: tuple[int, int] = (ROLE_BUILDER, ROLE_BREAKER)
    ) -> ConnectionDisruptionState:
        if len(seat_roles) != 2 or set(seat_roles) != {ROLE_BUILDER, ROLE_BREAKER}:
            raise ValueError("seat_roles must assign builder and breaker once each")
        return ConnectionDisruptionState(
            board=self._initial_board(),
            to_move=0,
            seat_roles=tuple(seat_roles),
            plies=0,
        )

    def current_player(self, state: ConnectionDisruptionState) -> int:
        return state.to_move

    def player_role(self, state: ConnectionDisruptionState, player: int) -> int:
        if player not in (0, 1):
            raise ValueError(f"player must be 0 or 1: {player}")
        return state.seat_roles[player]

    def legal_actions(self, state: ConnectionDisruptionState) -> list[int]:
        if state.terminal is not None:
            return []

        role = self.player_role(state, state.to_move)
        if role == ROLE_BUILDER:
            return [
                self.encode_place(index)
                for index, piece in enumerate(state.board)
                if piece == EMPTY and index not in self._protected
            ]

        actions: list[int] = []
        for from_index, piece in enumerate(state.board):
            if piece != BLOCKER:
                continue
            from_row, from_col = index_to_coord(from_index, cols=self._cols)
            for to_row, to_col in neighbors(
                from_row,
                from_col,
                rows=self._rows,
                cols=self._cols,
            ):
                target = coord_to_index(to_row, to_col, rows=self._rows, cols=self._cols)
                target_piece = state.board[target]
                if target_piece == EMPTY and target not in self._protected:
                    actions.append(self.encode_move(from_index, target))
                elif target_piece == BUILDER and target not in self._protected:
                    actions.append(self.encode_remove(from_index, target))
        return actions

    def apply_action(
        self, state: ConnectionDisruptionState, action: int
    ) -> ConnectionDisruptionState:
        if action not in self.legal_actions(state):
            raise IllegalActionError(
                f"illegal action {action} for player {state.to_move}"
            )

        board = state.board
        role = self.player_role(state, state.to_move)
        if role == ROLE_BUILDER:
            target = self.decode_place(action)
            board = replace_cell(board, index=target, value=BUILDER)
        elif action < self._remove_offset:
            from_index, to_index = self.decode_move(action)
            board = replace_cell(board, index=from_index, value=EMPTY)
            board = replace_cell(board, index=to_index, value=BLOCKER)
        else:
            _, target_index = self.decode_remove(action)
            board = replace_cell(board, index=target_index, value=EMPTY)

        plies = state.plies + 1
        next_player = 1 - state.to_move
        terminal = self._terminal_after_action(
            board=board,
            state=state,
            plies=plies,
        )
        next_state = ConnectionDisruptionState(
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
            next_state = ConnectionDisruptionState(
                board=board,
                to_move=next_player,
                seat_roles=state.seat_roles,
                plies=plies,
                terminal=terminal,
            )
        return next_state

    def is_terminal(self, state: ConnectionDisruptionState) -> bool:
        return state.terminal is not None

    def result(self, state: ConnectionDisruptionState) -> RoleResult:
        if state.terminal is None:
            raise RuntimeError("result is only available for terminal states")
        return state.terminal

    def observation_tensor(
        self, state: ConnectionDisruptionState, player: int
    ) -> np.ndarray:
        board = np.asarray(state.board, dtype=np.int8).reshape(self.board_shape)
        role = self.player_role(state, player)

        builders = board == BUILDER
        blockers = board == BLOCKER
        protected = np.zeros(self.board_shape, dtype=np.bool_)
        for index in self._protected:
            row, col = index_to_coord(index, cols=self._cols)
            protected[row, col] = True

        if role == ROLE_BUILDER:
            own = builders
            enemy = blockers
        else:
            own = blockers
            enemy = builders

        planes = np.zeros((7, *self.board_shape), dtype=np.float32)
        planes[0] = builders
        planes[1] = blockers
        planes[2] = protected
        planes[3] = own
        planes[4] = enemy
        planes[5].fill(1.0 if player == state.to_move else 0.0)
        planes[6].fill(float(role))
        return planes

    def action_mask(self, state: ConnectionDisruptionState) -> np.ndarray:
        return make_action_mask(self.action_size, self.legal_actions(state))

    def render(self, state: ConnectionDisruptionState) -> str:
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

    def _terminal_after_action(
        self,
        *,
        board: tuple[int, ...],
        state: ConnectionDisruptionState,
        plies: int,
    ) -> RoleResult | None:
        if self._builder_has_connection(board):
            return RoleResult(
                winner=self._player_with_role(state, ROLE_BUILDER),
                reason="builder_connection",
                plies=plies,
            )
        if plies >= self.max_plies:
            return RoleResult(
                winner=self._player_with_role(state, ROLE_BREAKER),
                reason="max_plies",
                plies=plies,
            )
        return None

    def _builder_has_connection(self, board: tuple[int, ...]) -> bool:
        return path_exists_between_edges(
            occupied={
                index for index, piece in enumerate(board) if piece == BUILDER
            },
            rows=self._rows,
            cols=self._cols,
            start_edge=self._start_edge,
            target_edge=self._target_edge,
        )

    def _render_cell(self, piece: int, index: int) -> str:
        if piece == BUILDER:
            return "M"
        if piece == BLOCKER:
            return "X"
        if index in self._protected:
            return "#"
        return "."

    def _validate_setup(self) -> None:
        if len(set(self._blockers)) != len(self._blockers):
            raise ValueError("setup blockers indices must be unique")
        if len(set(self._protected)) != len(self._protected):
            raise ValueError("setup protected indices must be unique")
        if len(set(self._builders)) != len(self._builders):
            raise ValueError("setup builders indices must be unique")

        blockers = set(self._blockers)
        protected = set(self._protected)
        builders = set(self._builders)
        if blockers & protected:
            raise ValueError("blockers must not overlap protected cells")
        if builders & blockers:
            raise ValueError("builders must not overlap blockers")
        if builders & protected:
            raise ValueError("builders must not overlap protected cells")

        board = self._initial_board()
        if self._builder_has_connection(board):
            raise ValueError("builder must not start connected")
        for seat_roles in (
            (ROLE_BUILDER, ROLE_BREAKER),
            (ROLE_BREAKER, ROLE_BUILDER),
        ):
            initial_state = ConnectionDisruptionState(
                board=board,
                to_move=0,
                seat_roles=seat_roles,
                plies=0,
            )
            if not self.legal_actions(initial_state):
                raise ValueError("initial state must have legal actions")

    def _initial_board(self) -> tuple[int, ...]:
        board = [EMPTY] * self._cell_count
        for index in self._builders:
            board[index] = BUILDER
        for index in self._blockers:
            board[index] = BLOCKER
        return tuple(board)

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

    def _player_with_role(self, state: ConnectionDisruptionState, role: int) -> int:
        return state.seat_roles.index(role)

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


class ConnectionDisruptionGenerator:
    family = "connection_disruption"

    def generate(
        self,
        *,
        seed: int,
        constraints: GenerationConstraints,
    ) -> GeneratedGameSpec:
        self._validate_inputs(seed=seed, constraints=constraints)
        rng = random.Random(seed)
        last_rejection: _CandidateRejected | None = None

        for _ in range(constraints.max_attempts):
            try:
                spec = self._candidate_spec(seed=seed, constraints=constraints, rng=rng)
                self._validate_candidate_spec(spec)
                return spec
            except _CandidateRejected as exc:
                last_rejection = exc

        detail = f": {last_rejection}" if last_rejection is not None else ""
        raise RuntimeError(f"failed to generate connection_disruption spec{detail}")

    def compile(self, spec: GeneratedGameSpec) -> ConnectionDisruptionGame:
        if spec.family != self.family:
            raise ValueError(
                f"expected connection_disruption spec, got {spec.family!r}"
            )
        return ConnectionDisruptionGame(spec)

    def _candidate_spec(
        self,
        *,
        seed: int,
        constraints: GenerationConstraints,
        rng: random.Random,
    ) -> GeneratedGameSpec:
        rows, cols = constraints.board_sizes[rng.randrange(len(constraints.board_sizes))]
        min_plies, max_plies = constraints.max_plies_range
        max_plies_value = (
            min_plies if min_plies == max_plies else rng.randint(min_plies, max_plies)
        )

        edge_targets = edge_cells(rows=rows, cols=cols, edge="west") | edge_cells(
            rows=rows, cols=cols, edge="east"
        )
        candidates = [
            index
            for index in range(rows * cols)
            if index not in edge_targets
        ]
        if not candidates:
            raise ValueError("no legal blocker cells")
        blocker_count = rng.randint(1, min(4, len(candidates)))
        blockers = sorted(rng.sample(candidates, blocker_count))

        return GeneratedGameSpec(
            family=self.family,
            name=f"connection_disruption_{rows}x{cols}_seed_{seed}",
            seed=seed,
            board={"rows": rows, "cols": cols},
            roles=("builder", "breaker"),
            setup={"blockers": blockers, "protected": []},
            actions={
                "builder": "place",
                "breaker": ["orthogonal_step", "adjacent_remove"],
            },
            terminal_rules={"connect": ["west", "east"]},
            max_plies=max_plies_value,
        )

    def _validate_inputs(
        self,
        *,
        seed: int,
        constraints: GenerationConstraints,
    ) -> None:
        if type(seed) is not int:
            raise ValueError("seed must be an int")
        for rows, cols in constraints.board_sizes:
            if rows < 5 or cols < 5:
                raise ValueError("board_sizes entries must be at least 5x5")

    def _validate_candidate_spec(self, spec: GeneratedGameSpec) -> None:
        if not spec.setup["blockers"]:
            raise _CandidateRejected("no blockers")
        try:
            game = self.compile(spec)
        except ValueError as exc:
            raise _CandidateRejected(str(exc)) from exc

        state = game.initial_state()
        if game.is_terminal(state):
            raise _CandidateRejected("initial state is terminal")
        if not game.legal_actions(state):
            raise _CandidateRejected("initial state has no legal actions")
