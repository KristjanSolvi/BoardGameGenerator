from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from research.asymbench.generation.connection_disruption import (
    ConnectionDisruptionGenerator,
)
from research.asymbench.generation.escape_capture import EscapeCaptureGenerator
from research.asymbench.generation.specs import GenerationConstraints
from research.asymbench.generation.validate import validate_generated_game


GENERATOR_REGISTRY = {
    "escape_capture": EscapeCaptureGenerator(),
    "connection_disruption": ConnectionDisruptionGenerator(),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate validated AsymBench grid-game specs."
    )
    parser.add_argument(
        "--family",
        choices=("escape_capture", "connection_disruption", "all"),
        required=True,
    )
    parser.add_argument("--count", type=_positive_int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research_runs/asymbench/generated"),
    )
    parser.add_argument("--random-games", type=_positive_int, default=8)
    args = parser.parse_args(argv)

    constraints = GenerationConstraints()
    families = _families_for_selection(args.family)
    for family in families:
        accepted = _generate_family(
            family=family,
            count=args.count,
            seed=args.seed,
            output=args.output,
            random_games=args.random_games,
            constraints=constraints,
        )
        if accepted < args.count:
            print(
                f"family={family} accepted={accepted} target={args.count}",
                file=sys.stderr,
            )
            return 1

    return 0


def _generate_family(
    *,
    family: str,
    count: int,
    seed: int,
    output: Path,
    random_games: int,
    constraints: GenerationConstraints,
) -> int:
    generator = GENERATOR_REGISTRY[family]
    accepted = 0
    attempts = 0
    candidate_seed = seed

    while accepted < count and attempts < constraints.max_attempts:
        attempts += 1
        try:
            spec = generator.generate(seed=candidate_seed, constraints=constraints)
        except Exception:
            candidate_seed += 1
            continue

        report = validate_generated_game(
            spec,
            random_games=random_games,
            seed=seed ^ candidate_seed,
        )
        if not report.valid:
            candidate_seed += 1
            continue

        run_dir = output / f"{family}_{candidate_seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(run_dir / "spec.json", spec.to_dict())
        _write_json(run_dir / "validation.json", report.to_dict())
        print(f"accepted={run_dir / 'spec.json'}")
        accepted += 1
        candidate_seed += 1

    return accepted


def _families_for_selection(family: str) -> tuple[str, ...]:
    if family == "all":
        return ("escape_capture", "connection_disruption")
    return (family,)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive int") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive int")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
