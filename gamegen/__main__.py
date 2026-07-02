"""CLI entry point.

    python -m gamegen generate --runs 5 --seed 42 [--config config.yaml]
    python -m gamegen replay runs/<run_dir> [--config config.yaml]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .backend import BackendNotAvailableError
from .config import Config
from .pipeline import generate, replay


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gamegen",
        description="Multi-agent LLM pipeline that invents new two-player "
                    "asymmetric abstract board games.",
    )
    parser.add_argument("--config", default="config.yaml",
                        help="path to config.yaml (default: ./config.yaml)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate", help="generate new games")
    p_gen.add_argument("--runs", type=int, default=1,
                       help="number of independent generation runs")
    p_gen.add_argument("--seed", type=int, default=None,
                       help="override the global seed from config.yaml")

    p_rep = sub.add_parser(
        "replay", help="re-run playtests on an existing run's engine")
    p_rep.add_argument("run_dir", type=Path)
    p_rep.add_argument("--seed", type=int, default=None,
                       help="override the global seed from config.yaml")

    args = parser.parse_args(argv)

    try:
        cfg = Config.load(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.seed is not None:
        cfg.seed = args.seed

    root = Path.cwd()
    try:
        if args.command == "generate":
            summaries = generate(cfg, args.runs, root)
            print(json.dumps(summaries, indent=2))
            return 0 if any(s["status"] == "accepted" for s in summaries) else 1
        else:
            run_dir = args.run_dir.resolve()
            if not run_dir.is_dir():
                print(f"error: {run_dir} is not a directory", file=sys.stderr)
                return 2
            replay(run_dir, cfg)
            return 0
    except BackendNotAvailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
