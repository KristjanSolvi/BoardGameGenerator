"""Hard validation of a generated engine (no LLM).

Checks, in order (stopping at the first hard failure so repair feedback
is focused):
  1. module imports and exposes exactly one engine class with the full
     required interface, instantiable with no arguments;
  2. generated pytest suite passes;
  3. N random playouts all terminate under the move cap, with hashable
     states and moves throughout;
  4. move soundness: every move from legal_moves is accepted by apply,
     and sampled moves NOT in the legal set are rejected;
  5. symmetry: mirror_state() gives an isomorphism between player 0's
     opening moves and player 1's opening moves in the mirrored start,
     and the mirror is an involution;
  6. determinism / state hashing: replaying the same seeded playout
     yields the identical hash sequence (this is what repetition rules
     rely on).

Returns a ValidationReport with per-check details for the run log; on
failure, .failure_message is written for the rules-engineer repair prompt.
"""

from __future__ import annotations

import importlib.util
import random
import statistics
import subprocess
import sys
import traceback
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .engine_interface import REQUIRED_METHODS


@dataclass
class ValidationReport:
    ok: bool = False
    failed_check: Optional[str] = None
    failure_message: Optional[str] = None
    checks: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "failed_check": self.failed_check,
            "failure_message": self.failure_message,
            "checks": self.checks,
        }


class ValidationFailure(Exception):
    def __init__(self, check: str, message: str):
        self.check = check
        self.message = message
        super().__init__(f"[{check}] {message}")


# ----------------------------------------------------------------------
def load_engine_class(engine_path: Path):
    """Import the generated module and return the engine class."""
    module_name = f"gamegen_generated_{engine_path.stem}_{id(engine_path)}"
    spec = importlib.util.spec_from_file_location(module_name, engine_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        raise ValidationFailure(
            "import", f"engine module failed to import:\n{traceback.format_exc()}"
        )
    candidates = [
        obj for name, obj in vars(module).items()
        if isinstance(obj, type)
        and all(callable(getattr(obj, m, None)) for m in REQUIRED_METHODS)
        and obj.__module__ == module_name
    ]
    if len(candidates) != 1:
        raise ValidationFailure(
            "import",
            f"expected exactly one class implementing {REQUIRED_METHODS}, "
            f"found {len(candidates)}: {[c.__name__ for c in candidates]}",
        )
    return candidates[0]


def run_pytest(test_path: Path, engine_dir: Path,
               timeout: int = 300) -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_path), "-q", "--no-header"],
        capture_output=True, text=True, timeout=timeout, cwd=engine_dir,
    )
    output = proc.stdout + proc.stderr
    return proc.returncode == 0, output


def _is_illegal_move_error(exc: BaseException) -> bool:
    return isinstance(exc, ValueError) or type(exc).__name__ == "IllegalMoveError"


# ----------------------------------------------------------------------
def validate_engine(
    engine_path: Path,
    test_path: Optional[Path],
    spec: dict,
    seed: int,
    random_playouts: int = 1000,
    move_cap: int = 400,
    illegal_move_samples: int = 200,
) -> ValidationReport:
    report = ValidationReport()
    try:
        engine_cls = load_engine_class(engine_path)
        try:
            engine = engine_cls()
        except Exception:
            raise ValidationFailure(
                "import",
                f"engine class {engine_cls.__name__} could not be "
                f"instantiated with no arguments:\n{traceback.format_exc()}",
            )
        report.checks["import"] = {"ok": True, "class": engine_cls.__name__}

        if test_path is not None:
            passed, output = run_pytest(test_path, engine_path.parent)
            report.checks["pytest"] = {"ok": passed, "output_tail": output[-4000:]}
            if not passed:
                raise ValidationFailure(
                    "pytest", f"generated tests failed:\n{output[-6000:]}"
                )

        _check_playouts(engine, report, seed, random_playouts, move_cap)
        _check_move_soundness(engine, report, seed, illegal_move_samples,
                              move_cap)
        _check_symmetry(engine, report)
        _check_determinism(engine, report, seed, move_cap)

        report.ok = True
    except ValidationFailure as vf:
        report.ok = False
        report.failed_check = vf.check
        report.failure_message = vf.message
        report.checks.setdefault(vf.check, {})["ok"] = False
    except Exception:
        report.ok = False
        report.failed_check = "validator_crash"
        report.failure_message = (
            "the engine raised an unexpected exception during validation:\n"
            + traceback.format_exc()
        )
    return report


# ----------------------------------------------------------------------
def _random_playout(engine, rng: random.Random, move_cap: int,
                    collect: Optional[dict] = None):
    """Play one random game. Returns (terminal_state, plies, hit_cap)."""
    state = engine.initial_state()
    for ply in range(move_cap):
        if engine.is_terminal(state):
            return state, ply, False
        player = engine.current_player(state)
        moves = engine.legal_moves(state, player)
        if not moves:
            raise ValidationFailure(
                "playouts",
                f"non-terminal state with no legal moves for player {player} "
                f"at ply {ply} (the spec's no_legal_move_rule must be encoded "
                f"in the engine, e.g. as a pass move or terminal state):\n"
                f"{engine.render(state)}",
            )
        if collect is not None:
            collect["branching"].append(len(moves))
            for m in moves:
                collect["move_universe"].add(m)
        move = rng.choice(moves)
        try:
            hash(state), hash(move)
        except TypeError:
            raise ValidationFailure(
                "playouts",
                f"state or move is not hashable at ply {ply}: "
                f"state type {type(state).__name__}, move {move!r}",
            )
        state = engine.apply(state, move)
    return state, move_cap, True


def _check_playouts(engine, report, seed, n_playouts, move_cap):
    rng = random.Random(seed ^ 0x5EED)
    lengths = []
    reasons: Counter = Counter()
    collect = {"branching": [], "move_universe": set()}
    for i in range(n_playouts):
        try:
            state, plies, hit_cap = _random_playout(engine, rng, move_cap, collect)
        except ValidationFailure:
            raise
        except Exception:
            raise ValidationFailure(
                "playouts",
                f"engine crashed during random playout {i}:\n"
                + traceback.format_exc(),
            )
        if hit_cap:
            raise ValidationFailure(
                "playouts",
                f"random playout {i} did not terminate within {move_cap} "
                "plies. The game must always end; check for stalemate "
                "loops, missing repetition handling, or unreachable "
                "termination conditions.",
            )
        result = engine.result(state)
        if (not isinstance(result, dict) or "winner" not in result
                or result["winner"] not in (0, 1, None)):
            raise ValidationFailure(
                "playouts",
                f"result() must return {{'winner': 0|1|None, 'reason': str}}, "
                f"got {result!r}",
            )
        lengths.append(plies)
        reasons[str(result.get("reason"))] += 1
    report.checks["playouts"] = {
        "ok": True,
        "n": n_playouts,
        "length_mean": statistics.mean(lengths),
        "length_max": max(lengths),
        "termination_reasons": dict(reasons),
        "branching_mean": statistics.mean(collect["branching"]),
        "branching_max": max(collect["branching"]),
        "distinct_moves_seen": len(collect["move_universe"]),
    }
    report.checks["_move_universe"] = collect["move_universe"]  # for soundness


def _check_move_soundness(engine, report, seed, n_samples, move_cap):
    """Every legal move applies; sampled non-legal moves are rejected."""
    rng = random.Random(seed ^ 0xBADC0DE)
    universe = report.checks.pop("_move_universe")
    tested_legal = 0
    tested_illegal = 0
    for _ in range(20):  # 20 random trajectories, probing along each
        state = engine.initial_state()
        for _ply in range(move_cap):
            if engine.is_terminal(state):
                break
            player = engine.current_player(state)
            moves = engine.legal_moves(state, player)
            if not moves:
                break
            legal_set = set(moves)
            # all legal moves must be accepted
            for m in rng.sample(moves, min(len(moves), 5)):
                try:
                    engine.apply(state, m)
                    tested_legal += 1
                except Exception:
                    raise ValidationFailure(
                        "move_soundness",
                        f"apply() rejected a move returned by legal_moves: "
                        f"{m!r}\n{traceback.format_exc()}\nstate:\n"
                        f"{engine.render(state)}",
                    )
            # sampled moves outside the legal set must be rejected
            outside = [m for m in universe if m not in legal_set]
            for m in rng.sample(outside, min(len(outside), 4)):
                if tested_illegal >= n_samples:
                    break
                try:
                    engine.apply(state, m)
                except Exception as exc:
                    if not _is_illegal_move_error(exc):
                        raise ValidationFailure(
                            "move_soundness",
                            f"apply() raised {type(exc).__name__} instead of "
                            f"IllegalMoveError/ValueError for illegal move "
                            f"{m!r}:\n{traceback.format_exc()}",
                        )
                    tested_illegal += 1
                else:
                    raise ValidationFailure(
                        "move_soundness",
                        f"apply() ACCEPTED a move that legal_moves did not "
                        f"return: {m!r} for player {player} in state:\n"
                        f"{engine.render(state)}",
                    )
            state = engine.apply(state, rng.choice(moves))
    report.checks["move_soundness"] = {
        "ok": True,
        "legal_moves_tested": tested_legal,
        "illegal_moves_tested": tested_illegal,
    }


def _check_symmetry(engine, report):
    """mirror_state() must be an involution mapping player 0's opening
    options one-to-one onto player 1's options in the mirrored start."""
    s0 = engine.initial_state()
    m0 = engine.mirror_state(s0)
    if engine.current_player(s0) != 0:
        raise ValidationFailure(
            "symmetry", "player 0 must be to move in initial_state()"
        )
    if engine.current_player(m0) != 1:
        raise ValidationFailure(
            "symmetry",
            "mirror_state(initial_state()) must have player 1 to move "
            f"(got player {engine.current_player(m0)})",
        )
    if hash(engine.mirror_state(m0)) != hash(s0) or engine.mirror_state(m0) != s0:
        raise ValidationFailure(
            "symmetry",
            "mirror_state must be an involution: "
            "mirror(mirror(s)) != s for the initial state",
        )
    moves0 = engine.legal_moves(s0, 0)
    moves1 = engine.legal_moves(m0, 1)
    if len(moves0) != len(moves1):
        raise ValidationFailure(
            "symmetry",
            f"players have different numbers of opening moves after the "
            f"color swap: {len(moves0)} vs {len(moves1)}. The game (or the "
            f"engine's mirror_state) is asymmetric.",
        )
    succ0 = Counter(hash(engine.mirror_state(engine.apply(s0, m))) for m in moves0)
    succ1 = Counter(hash(engine.apply(m0, m)) for m in moves1)
    if succ0 != succ1:
        raise ValidationFailure(
            "symmetry",
            "opening move sets are not isomorphic under the color swap: the "
            "multiset of mirrored successor states of player 0's opening "
            "moves does not equal the multiset of successor states of "
            "player 1's opening moves in the mirrored start. Either the "
            "rules are asymmetric or mirror_state is implemented wrongly.",
        )
    report.checks["symmetry"] = {
        "ok": True,
        "opening_moves_per_player": len(moves0),
    }


def _check_determinism(engine, report, seed, move_cap):
    """Same seed twice -> identical hash trajectory (repetition hashing)."""
    trajectories = []
    for _ in range(2):
        rng = random.Random(seed ^ 0xD37E12)
        state = engine.initial_state()
        hashes = [hash(state)]
        for _ply in range(move_cap):
            if engine.is_terminal(state):
                break
            moves = engine.legal_moves(state, engine.current_player(state))
            if not moves:
                break
            state = engine.apply(state, rng.choice(moves))
            hashes.append(hash(state))
        trajectories.append(hashes)
    if trajectories[0] != trajectories[1]:
        raise ValidationFailure(
            "determinism",
            "replaying an identical seeded playout produced a different "
            "state-hash sequence. The engine must be deterministic and "
            "states must hash stably (no set iteration order leaks, no "
            "id()-based hashing); repetition rules depend on this.",
        )
    report.checks["determinism"] = {
        "ok": True, "trajectory_length": len(trajectories[0]),
    }
