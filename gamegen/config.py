"""Load and validate config.yaml."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

DEFAULTS: dict[str, Any] = {
    "backend": "codex",
    "model": "gpt-5.5",
    "reasoning_effort": None,
    "llm": {"timeout_seconds": 900, "max_retries": 2},
    "seed": 42,
    "limits": {"revision_cycles": 4, "repair_rounds": 3, "format_retries": 2},
    "validation": {
        "random_playouts": 1000,
        "move_cap": 400,
        "illegal_move_samples": 200,
    },
    "playtest": {
        "random_vs_random_games": 200,
        "mc_vs_random_games": 40,
        "mc_vs_mc_games": 20,
        "mc_rollout_budget": 96,
        "mc_simulation_move_cap": 200,
        "move_cap": 400,
    },
}


@dataclass
class Config:
    backend: str
    model: str
    reasoning_effort: Optional[str]
    llm: dict[str, Any]
    seed: int
    limits: dict[str, Any]
    validation: dict[str, Any]
    playtest: dict[str, Any]
    source_path: Optional[Path] = field(default=None)

    @classmethod
    def load(cls, path: Path | str = "config.yaml") -> "Config":
        path = Path(path)
        merged = copy.deepcopy(DEFAULTS)
        if path.exists():
            with open(path) as f:
                user_cfg = yaml.safe_load(f) or {}
            if not isinstance(user_cfg, dict):
                raise ValueError(f"{path} must contain a YAML mapping")
            _deep_merge(merged, user_cfg)
        else:
            raise FileNotFoundError(
                f"Config file not found: {path}. Copy the repo's config.yaml "
                "or pass --config."
            )
        if merged["backend"] not in ("codex", "claude"):
            raise ValueError(
                f"Unknown backend {merged['backend']!r}; expected 'codex' or 'claude'"
            )
        return cls(
            backend=merged["backend"],
            model=str(merged["model"]),
            reasoning_effort=merged.get("reasoning_effort"),
            llm=merged["llm"],
            seed=int(merged["seed"]),
            limits=merged["limits"],
            validation=merged["validation"],
            playtest=merged["playtest"],
            source_path=path,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "llm": self.llm,
            "seed": self.seed,
            "limits": self.limits,
            "validation": self.validation,
            "playtest": self.playtest,
        }


def _deep_merge(base: dict, override: dict) -> None:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
