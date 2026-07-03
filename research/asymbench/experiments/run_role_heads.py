from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research.asymbench.games.breaker_builder import BreakerBuilder
from research.asymbench.games.micro_tafl import MicroTafl
from research.asymbench.generation.loader import (
    compile_generated_game,
    load_generated_spec,
)
from research.asymbench.learning.evaluate import evaluate_model_vs_random
from research.asymbench.learning.model import PolicyValueNet
from research.asymbench.learning.replay import ReplayBuffer
from research.asymbench.learning.selfplay import generate_selfplay_game
from research.asymbench.learning.train import train_steps


GAME_CLASSES = {
    "breaker_builder": BreakerBuilder,
    "micro_tafl": MicroTafl,
}

VARIANT_ROLE_HEADS = {
    "shared_heads": False,
    "role_heads": True,
}

# We seed Python, NumPy, Torch, and CUDA/cuDNN where available. Full Torch
# deterministic algorithms are not forced here, so CUDA smoke metrics should be
# treated as approximate even with fixed seeds.


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run AsymBench shared-head versus role-head smoke experiments."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    args = parser.parse_args(argv)

    run_dir = run_experiment(args.config, device_override=args.device)
    print(f"run_dir={run_dir}")
    return 0


def run_experiment(config_path: Path, device_override: str | None = None) -> Path:
    config = _load_config(config_path)
    _validate_config(config)

    game_name, game_factory, generated_metadata = _resolve_game_source(config)
    device_requested = device_override or str(config.get("device", "cpu"))
    device_used = _resolve_device(device_requested)

    output_root = Path(str(config["output_root"]))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = output_root / f"{game_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    written_config = dict(config)
    written_config["config_path"] = str(config_path)
    written_config["device_requested"] = device_requested
    written_config["device_used"] = device_used
    _write_json(run_dir / "config.json", written_config)

    metrics_path = run_dir / "metrics.jsonl"
    all_rows: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []

    for seed in config["seeds"]:
        trial_seed = int(seed)
        for variant in config["model_variants"]:
            variant_name = str(variant)
            model_init_seed = _variant_seed(trial_seed, variant_name)
            variant_seed = model_init_seed
            _seed_everything(model_init_seed)

            variant_dir = run_dir / variant_name / f"seed_{seed}"
            variant_dir.mkdir(parents=True, exist_ok=False)

            game = game_factory()
            model = _create_model(game, role_heads=VARIANT_ROLE_HEADS[variant_name])
            buffer = ReplayBuffer(
                capacity=int(config["replay_capacity"]),
                seed=trial_seed,
            )
            selfplay_games_total = 0

            for iteration in range(1, int(config["iterations"]) + 1):
                selfplay_game_seeds: list[int] = []
                selfplay_seat_role_counts: Counter[str] = Counter()
                for game_index in range(int(config["selfplay_games_per_iteration"])):
                    selfplay_seed = _derived_seed(trial_seed, iteration, game_index)
                    seat_roles = _selfplay_seat_roles(
                        trial_seed=trial_seed,
                        iteration=iteration,
                        game_index=game_index,
                    )
                    examples, _ = generate_selfplay_game(
                        game=game,
                        model=model,
                        device=device_used,
                        simulations=int(config["mcts_simulations"]),
                        seed=selfplay_seed,
                        seat_roles=seat_roles,
                    )
                    selfplay_game_seeds.append(selfplay_seed)
                    selfplay_seat_role_counts[_seat_roles_key(seat_roles)] += 1
                    for example in examples:
                        buffer.add(example)
                    selfplay_games_total += 1

                if len(buffer) == 0:
                    raise RuntimeError(
                        f"self-play produced no replay examples for {variant} seed {seed}"
                    )

                effective_batch_size = min(int(config["batch_size"]), len(buffer))
                train_metrics = train_steps(
                    model=model,
                    buffer=buffer,
                    batch_size=effective_batch_size,
                    steps=int(config["train_steps_per_iteration"]),
                    lr=float(config["learning_rate"]),
                    device=device_used,
                )
                eval_seed = _derived_seed(trial_seed, iteration, 10_000)
                eval_summary = evaluate_model_vs_random(
                    game=game,
                    model=model,
                    device=device_used,
                    games=int(config["eval_games"]),
                    simulations=int(config["eval_simulations"]),
                    seed=eval_seed,
                )

                row = {
                    "iteration": iteration,
                    "game": game_name,
                    "variant": variant_name,
                    "seed": int(seed),
                    "variant_seed": variant_seed,
                    "trial_seed": trial_seed,
                    "model_init_seed": model_init_seed,
                    "eval_seed": eval_seed,
                    "selfplay_game_seeds": selfplay_game_seeds,
                    "selfplay_seat_role_counts": dict(
                        sorted(selfplay_seat_role_counts.items())
                    ),
                    "device_requested": device_requested,
                    "device_used": device_used,
                    "selfplay_games_total": selfplay_games_total,
                    "replay_examples": len(buffer),
                    "train_batch_size": effective_batch_size,
                    "policy_loss": float(train_metrics["policy_loss"]),
                    "value_loss": float(train_metrics["value_loss"]),
                    "train_total_loss": float(train_metrics["total_loss"]),
                    "eval_games": int(config["eval_games"]),
                    "eval_simulations": int(config["eval_simulations"]),
                    "eval_model_win_rate": float(eval_summary["model_win_rate"]),
                    "eval_random_win_rate": float(eval_summary["random_win_rate"]),
                    "eval_role_win_rates": eval_summary["role_win_rates"],
                    "eval_model_role_win_rates": eval_summary[
                        "model_role_win_rates"
                    ],
                    "eval_draw_rate": float(eval_summary["draw_rate"]),
                    "eval_avg_plies": float(eval_summary["avg_plies"]),
                    "eval_termination_reasons": eval_summary[
                        "termination_reasons"
                    ],
                    **generated_metadata,
                }
                _append_jsonl(metrics_path, row)
                all_rows.append(row)

            checkpoint_path = variant_dir / "final_checkpoint.pt"
            checkpoint_payload = {
                "model_state_dict": _cpu_state_dict(model),
                "config": config,
                "game": game_name,
                "variant": variant_name,
                "seed": int(seed),
                "variant_seed": variant_seed,
                "trial_seed": trial_seed,
                "model_init_seed": model_init_seed,
                "device_requested": device_requested,
                "device_used": device_used,
                "input_shape": model.input_shape,
                "action_size": model.action_size,
                "num_roles": model.num_roles,
                "role_heads": model.role_heads,
            }
            if generated_metadata:
                checkpoint_payload["generated_metadata"] = dict(generated_metadata)
            torch.save(checkpoint_payload, checkpoint_path)
            checkpoints.append(
                {
                    "variant": variant_name,
                    "seed": int(seed),
                    "variant_seed": variant_seed,
                    "trial_seed": trial_seed,
                    "model_init_seed": model_init_seed,
                    "path": str(checkpoint_path),
                    **generated_metadata,
                }
            )

    summary = _summarize_run(
        config=config,
        game_name=game_name,
        generated_metadata=generated_metadata,
        run_dir=run_dir,
        device_requested=device_requested,
        device_used=device_used,
        metrics=all_rows,
        checkpoints=checkpoints,
    )
    _write_json(run_dir / "role_summary.json", summary)
    return run_dir


def _load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _validate_config(config: dict[str, Any]) -> None:
    has_game = "game" in config
    has_game_source = "game_source" in config
    if has_game == has_game_source:
        raise ValueError("config must include exactly one of game or game_source")

    if has_game:
        if config.get("game") not in GAME_CLASSES:
            raise ValueError(f"unknown game: {config.get('game')!r}")
    else:
        game_source = config.get("game_source")
        if type(game_source) is not dict:
            raise ValueError("game_source must be a dict")
        if game_source.get("type") != "generated_spec":
            raise ValueError("game_source.type must be 'generated_spec'")
        path = game_source.get("path")
        if type(path) is not str or path == "":
            raise ValueError("game_source.path must be a non-empty string")

    variants = config.get("model_variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("model_variants must be a non-empty list")
    unknown_variants = sorted(set(variants) - set(VARIANT_ROLE_HEADS))
    if unknown_variants:
        raise ValueError(f"unknown model_variants: {unknown_variants}")
    for key in (
        "iterations",
        "selfplay_games_per_iteration",
        "train_steps_per_iteration",
        "batch_size",
        "replay_capacity",
        "mcts_simulations",
        "eval_games",
        "eval_simulations",
    ):
        if int(config[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if float(config["learning_rate"]) <= 0.0:
        raise ValueError("learning_rate must be positive")
    if not config.get("seeds"):
        raise ValueError("seeds must not be empty")
    if config.get("device", "cpu") not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")


def _resolve_device(device_requested: str) -> str:
    if device_requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device_requested


def _resolve_game_source(
    config: dict[str, Any],
) -> tuple[str, Any, dict[str, Any]]:
    game_name = config.get("game")
    if game_name is not None:
        game_name = str(game_name)
        return game_name, lambda: GAME_CLASSES[game_name](), {}

    game_source = config["game_source"]
    spec_path = Path(str(game_source["path"]))
    spec = load_generated_spec(spec_path)
    metadata = {
        "generated_family": spec.family,
        "generated_name": spec.name,
        "generated_seed": int(spec.seed),
        "generated_spec_path": str(spec_path),
    }
    return spec.name, lambda: compile_generated_game(spec), metadata


def _create_model(game: Any, *, role_heads: bool) -> PolicyValueNet:
    state = game.initial_state()
    player = game.current_player(state)
    input_shape = tuple(game.observation_tensor(state, player=player).shape)
    return PolicyValueNet(
        input_shape=input_shape,
        action_size=game.action_size,
        num_roles=len(game.roles),
        role_heads=role_heads,
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _variant_seed(seed: int, variant: str) -> int:
    digest = hashlib.blake2b(
        f"{seed}:{variant}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, byteorder="big") % (2**31)


def _derived_seed(seed: int, iteration: int, index: int) -> int:
    return seed * 1_000_003 + iteration * 10_007 + index


def _selfplay_seat_roles(
    *, trial_seed: int, iteration: int, game_index: int
) -> tuple[int, int]:
    offset = _derived_seed(trial_seed, iteration, 50_000) % 2
    if (offset + game_index) % 2 == 0:
        return (0, 1)
    return (1, 0)


def _seat_roles_key(seat_roles: tuple[int, int]) -> str:
    return f"{seat_roles[0]}-{seat_roles[1]}"


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _summarize_run(
    *,
    config: dict[str, Any],
    game_name: str,
    generated_metadata: dict[str, Any],
    run_dir: Path,
    device_requested: str,
    device_used: str,
    metrics: list[dict[str, Any]],
    checkpoints: list[dict[str, Any]],
) -> dict[str, Any]:
    final_rows = [
        row for row in metrics if row["iteration"] == int(config["iterations"])
    ]
    by_variant = {}
    for variant in config["model_variants"]:
        variant_rows = [row for row in final_rows if row["variant"] == variant]
        if not variant_rows:
            continue
        by_variant[str(variant)] = {
            "final_eval_model_win_rate_mean": _mean(
                row["eval_model_win_rate"] for row in variant_rows
            ),
            "final_eval_random_win_rate_mean": _mean(
                row["eval_random_win_rate"] for row in variant_rows
            ),
            "final_eval_draw_rate_mean": _mean(
                row["eval_draw_rate"] for row in variant_rows
            ),
            "final_eval_avg_plies_mean": _mean(
                row["eval_avg_plies"] for row in variant_rows
            ),
            "final_policy_loss_mean": _mean(row["policy_loss"] for row in variant_rows),
            "final_value_loss_mean": _mean(row["value_loss"] for row in variant_rows),
            "final_total_loss_mean": _mean(
                row["train_total_loss"] for row in variant_rows
            ),
            "final_replay_examples_mean": _mean(
                row["replay_examples"] for row in variant_rows
            ),
            "final_model_role_win_rates_mean": _mean_nested_rates(
                row["eval_model_role_win_rates"] for row in variant_rows
            ),
            "final_termination_reasons": _sum_counter_maps(
                row["eval_termination_reasons"] for row in variant_rows
            ),
            "final_rows": variant_rows,
        }

    return {
        "game": game_name,
        "run_dir": str(run_dir),
        "device_requested": device_requested,
        "device_used": device_used,
        "iterations": int(config["iterations"]),
        "seeds": [int(seed) for seed in config["seeds"]],
        "model_variants": list(config["model_variants"]),
        "metrics_rows": len(metrics),
        "by_variant": by_variant,
        "checkpoints": checkpoints,
        **generated_metadata,
    }


def _mean(values: Any) -> float:
    values = list(values)
    if not values:
        return 0.0
    return round(float(sum(values) / len(values)), 6)


def _mean_nested_rates(rows: Any) -> dict[str, float]:
    totals: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            totals.setdefault(str(key), []).append(float(value))
    return {key: _mean(values) for key, values in sorted(totals.items())}


def _sum_counter_maps(rows: Any) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update({str(key): int(value) for key, value in row.items()})
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    raise SystemExit(main())
