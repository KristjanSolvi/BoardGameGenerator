from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


VALID_FAMILIES = {"escape_capture", "connection_disruption"}


@dataclass(frozen=True)
class GenerationConstraints:
    board_sizes: tuple[tuple[int, int], ...] = ((5, 5), (6, 6), (7, 7))
    max_plies_range: tuple[int, int] = (40, 120)
    max_attempts: int = 100

    def __post_init__(self) -> None:
        if not self.board_sizes:
            raise ValueError("board_sizes must not be empty")
        for rows, cols in self.board_sizes:
            if rows <= 0 or cols <= 0:
                raise ValueError("board sizes must be positive")
        min_plies, max_plies = self.max_plies_range
        if min_plies <= 0 or max_plies < min_plies:
            raise ValueError("max_plies_range must be positive and ordered")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")


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
        if self.family not in VALID_FAMILIES:
            raise ValueError(f"unknown family: {self.family!r}")
        if len(self.roles) != 2:
            raise ValueError("roles must contain exactly two entries")
        rows = int(self.board.get("rows", 0))
        cols = int(self.board.get("cols", 0))
        if rows <= 0 or cols <= 0:
            raise ValueError("board rows and cols must be positive")
        if self.max_plies <= 0:
            raise ValueError("max_plies must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "name": self.name,
            "seed": int(self.seed),
            "board": dict(self.board),
            "roles": list(self.roles),
            "setup": dict(self.setup),
            "actions": dict(self.actions),
            "terminal_rules": dict(self.terminal_rules),
            "max_plies": int(self.max_plies),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GeneratedGameSpec:
        board = dict(data["board"])
        if "rows" in board:
            board["rows"] = int(board["rows"])
        if "cols" in board:
            board["cols"] = int(board["cols"])
        return cls(
            family=str(data["family"]),
            name=str(data["name"]),
            seed=int(data["seed"]),
            board=board,
            roles=tuple(str(role) for role in data["roles"]),
            setup=dict(data.get("setup", {})),
            actions=dict(data.get("actions", {})),
            terminal_rules=dict(data.get("terminal_rules", {})),
            max_plies=int(data["max_plies"]),
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
            family=str(data["family"]),
            name=str(data["name"]),
            valid=bool(data["valid"]),
            reasons=tuple(str(reason) for reason in data.get("reasons", ())),
            initial_branching_factor=int(data.get("initial_branching_factor", 0)),
            random_role_win_rates={
                str(key): float(value)
                for key, value in data.get("random_role_win_rates", {}).items()
            },
            mcts_role_win_rates={
                str(key): float(value)
                for key, value in data.get("mcts_role_win_rates", {}).items()
            },
            average_random_plies=float(data.get("average_random_plies", 0.0)),
            terminal_reasons={
                str(key): int(value)
                for key, value in data.get("terminal_reasons", {}).items()
            },
        )
