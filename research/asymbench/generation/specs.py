from __future__ import annotations

import copy
from dataclasses import dataclass, field
import math
from typing import Any


VALID_FAMILIES = {"escape_capture", "connection_disruption"}


def _require_string(value: Any, field_name: str, *, allow_empty: bool = False) -> str:
    if type(value) is not str:
        raise ValueError(f"{field_name} must be a string")
    if not allow_empty and value == "":
        raise ValueError(f"{field_name} must not be empty")
    return value


def _require_bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field_name} must be a bool")
    return value


def _require_int(value: Any, field_name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field_name} must be an int")
    return value


def _require_finite_float(value: Any, field_name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{field_name} must be a finite float")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _require_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{field_name} must be a dict")
    for key in value:
        if type(key) is not str:
            raise ValueError(f"{field_name} keys must be strings")
    return copy.deepcopy(value)


@dataclass(frozen=True)
class GenerationConstraints:
    board_sizes: tuple[tuple[int, int], ...] = ((5, 5), (6, 6), (7, 7))
    max_plies_range: tuple[int, int] = (40, 120)
    max_attempts: int = 100

    def __post_init__(self) -> None:
        if type(self.board_sizes) not in (tuple, list) or not self.board_sizes:
            raise ValueError("board_sizes must not be empty")
        board_sizes: list[tuple[int, int]] = []
        for rows, cols in self.board_sizes:
            rows = _require_int(rows, "board_sizes rows")
            cols = _require_int(cols, "board_sizes cols")
            if rows <= 0 or cols <= 0:
                raise ValueError("board sizes must be positive")
            board_sizes.append((rows, cols))
        if type(self.max_plies_range) not in (tuple, list) or len(self.max_plies_range) != 2:
            raise ValueError("max_plies_range must contain two entries")
        min_plies = _require_int(self.max_plies_range[0], "max_plies_range minimum")
        max_plies = _require_int(self.max_plies_range[1], "max_plies_range maximum")
        if min_plies <= 0 or max_plies < min_plies:
            raise ValueError("max_plies_range must be positive and ordered")
        max_attempts = _require_int(self.max_attempts, "max_attempts")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        object.__setattr__(self, "board_sizes", tuple(board_sizes))
        object.__setattr__(self, "max_plies_range", (min_plies, max_plies))
        object.__setattr__(self, "max_attempts", max_attempts)


@dataclass(frozen=True)
class GeneratedGameSpec:
    family: str
    name: str
    seed: int
    board: dict[str, int]
    roles: tuple[str, str]
    setup: dict[str, Any] = field(default_factory=dict)
    actions: dict[str, Any] = field(default_factory=dict)
    terminal_rules: dict[str, Any] = field(default_factory=dict)
    max_plies: int = 80

    def __post_init__(self) -> None:
        family = _require_string(self.family, "family")
        if family not in VALID_FAMILIES:
            raise ValueError(f"unknown family: {family!r}")
        name = _require_string(self.name, "name")
        seed = _require_int(self.seed, "seed")
        board = _require_mapping(self.board, "board")
        rows = _require_int(board.get("rows"), "board rows")
        cols = _require_int(board.get("cols"), "board cols")
        if rows <= 0 or cols <= 0:
            raise ValueError("board rows and cols must be positive")
        board["rows"] = rows
        board["cols"] = cols
        if type(self.roles) not in (tuple, list) or len(self.roles) != 2:
            raise ValueError("roles must contain exactly two entries")
        roles = tuple(_require_string(role, "roles") for role in self.roles)
        max_plies = _require_int(self.max_plies, "max_plies")
        if max_plies <= 0:
            raise ValueError("max_plies must be positive")
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "board", board)
        object.__setattr__(self, "roles", roles)
        object.__setattr__(self, "setup", _require_mapping(self.setup, "setup"))
        object.__setattr__(self, "actions", _require_mapping(self.actions, "actions"))
        object.__setattr__(
            self,
            "terminal_rules",
            _require_mapping(self.terminal_rules, "terminal_rules"),
        )
        object.__setattr__(self, "max_plies", max_plies)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "name": self.name,
            "seed": int(self.seed),
            "board": copy.deepcopy(self.board),
            "roles": list(self.roles),
            "setup": copy.deepcopy(self.setup),
            "actions": copy.deepcopy(self.actions),
            "terminal_rules": copy.deepcopy(self.terminal_rules),
            "max_plies": int(self.max_plies),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GeneratedGameSpec:
        return cls(
            family=data["family"],
            name=data["name"],
            seed=data["seed"],
            board=data["board"],
            roles=data["roles"],
            setup=data.get("setup", {}),
            actions=data.get("actions", {}),
            terminal_rules=data.get("terminal_rules", {}),
            max_plies=data["max_plies"],
        )


@dataclass(frozen=True)
class ValidationReport:
    family: str
    name: str
    valid: bool
    reasons: tuple[str, ...] = ()
    initial_branching_factor: int = 0
    random_role_win_rates: dict[str, float] = field(default_factory=dict)
    mcts_role_win_rates: dict[str, float] = field(default_factory=dict)
    average_random_plies: float = 0.0
    terminal_reasons: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        family = _require_string(self.family, "family")
        if family not in VALID_FAMILIES:
            raise ValueError(f"unknown family: {family!r}")
        name = _require_string(self.name, "name")
        valid = _require_bool(self.valid, "valid")
        if type(self.reasons) not in (tuple, list):
            raise ValueError("reasons must be a tuple or list")
        reasons = tuple(_require_string(reason, "reasons") for reason in self.reasons)
        if not valid and not reasons:
            raise ValueError("invalid reports must include at least one reason")
        initial_branching_factor = _require_int(
            self.initial_branching_factor,
            "initial_branching_factor",
        )
        if initial_branching_factor < 0:
            raise ValueError("initial_branching_factor must be non-negative")
        random_role_win_rates = self._validated_win_rates(
            self.random_role_win_rates,
            "random_role_win_rates",
        )
        mcts_role_win_rates = self._validated_win_rates(
            self.mcts_role_win_rates,
            "mcts_role_win_rates",
        )
        average_random_plies = _require_finite_float(
            self.average_random_plies,
            "average_random_plies",
        )
        if average_random_plies < 0.0:
            raise ValueError("average_random_plies must be non-negative")
        terminal_reasons = self._validated_terminal_reasons(self.terminal_reasons)
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(
            self,
            "initial_branching_factor",
            initial_branching_factor,
        )
        object.__setattr__(self, "random_role_win_rates", random_role_win_rates)
        object.__setattr__(self, "mcts_role_win_rates", mcts_role_win_rates)
        object.__setattr__(self, "average_random_plies", average_random_plies)
        object.__setattr__(self, "terminal_reasons", terminal_reasons)

    @staticmethod
    def _validated_win_rates(value: Any, field_name: str) -> dict[str, float]:
        rates = _require_mapping(value, field_name)
        for key, rate in list(rates.items()):
            rate = _require_finite_float(rate, f"{field_name} win rate")
            if rate < 0.0 or rate > 1.0:
                raise ValueError(f"{field_name} win rate must be in [0.0, 1.0]")
            rates[key] = rate
        return rates

    @staticmethod
    def _validated_terminal_reasons(value: Any) -> dict[str, int]:
        terminal_reasons = _require_mapping(value, "terminal_reasons")
        for key, count in list(terminal_reasons.items()):
            count = _require_int(count, "terminal reason count")
            if count < 0:
                raise ValueError("terminal reason counts must be non-negative")
            terminal_reasons[key] = count
        return terminal_reasons

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "name": self.name,
            "valid": bool(self.valid),
            "reasons": list(self.reasons),
            "initial_branching_factor": int(self.initial_branching_factor),
            "random_role_win_rates": {
                str(key): float(value) for key, value in self.random_role_win_rates.items()
            },
            "mcts_role_win_rates": {
                str(key): float(value) for key, value in self.mcts_role_win_rates.items()
            },
            "average_random_plies": float(self.average_random_plies),
            "terminal_reasons": {
                str(key): int(value) for key, value in self.terminal_reasons.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidationReport:
        return cls(
            family=data["family"],
            name=data["name"],
            valid=data["valid"],
            reasons=data.get("reasons", ()),
            initial_branching_factor=data.get("initial_branching_factor", 0),
            random_role_win_rates=data.get("random_role_win_rates", {}),
            mcts_role_win_rates=data.get("mcts_role_win_rates", {}),
            average_random_plies=data.get("average_random_plies", 0.0),
            terminal_reasons=data.get("terminal_reasons", {}),
        )
