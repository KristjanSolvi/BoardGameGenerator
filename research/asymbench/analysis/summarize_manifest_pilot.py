from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


VARIANTS = ("shared_heads", "role_heads")
BOOTSTRAP_SAMPLES = 2000
SUMMARY_FIELDS = {
    "final_eval_model_win_rate_mean": "win_rate",
    "final_eval_random_win_rate_mean": "random_win_rate",
    "final_eval_draw_rate_mean": "draw_rate",
    "final_eval_avg_plies_mean": "avg_plies",
}


def summarize_manifest_pilot(
    pilot_manifest: Path,
    *,
    runs_root: Path | None = None,
) -> dict[str, Any]:
    pilot_manifest = Path(pilot_manifest)
    pilot = json.loads(pilot_manifest.read_text())
    entries = pilot.get("entries")
    if not isinstance(entries, list):
        raise ValueError("pilot_manifest entries must be a list")
    resolved_runs_root = _runs_root(pilot, pilot_manifest, runs_root)
    role_summaries = _load_role_summaries(resolved_runs_root)

    summarized_entries = []
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, Mapping):
            raise ValueError(f"pilot entry {index} must be an object")
        summarized_entries.append(
            _summarize_entry(
                entry,
                role_summaries=role_summaries,
            )
        )

    completed = [entry for entry in summarized_entries if entry["completed"]]
    return {
        "schema_version": 1,
        "pilot_manifest": str(pilot_manifest),
        "runs_root": str(resolved_runs_root),
        "entries_total": len(summarized_entries),
        "completed_entries": len(completed),
        "missing_entries": len(summarized_entries) - len(completed),
        "entries": summarized_entries,
        "by_bucket": _aggregate_group(summarized_entries, "bucket"),
        "by_family": _aggregate_group(summarized_entries, "family"),
        "by_bucket_family": _aggregate_bucket_family(summarized_entries),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize manifest-driven role-head pilot runs."
    )
    parser.add_argument("pilot_manifest", type=Path)
    parser.add_argument(
        "--runs-root",
        type=Path,
        help="Optional override for the run_role_heads output root.",
    )
    parser.add_argument("--output", type=Path, help="Optional summary JSON path.")
    args = parser.parse_args(argv)

    try:
        summary = summarize_manifest_pilot(
            args.pilot_manifest,
            runs_root=args.runs_root,
        )
        output = json.dumps(summary, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output)
        else:
            print(output, end="")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.output:
        print(
            f"summary={args.output} completed={summary['completed_entries']}/{summary['entries_total']}"
        )
    return 0


def _runs_root(
    pilot: Mapping[str, Any],
    pilot_manifest: Path,
    runs_root: Path | None,
) -> Path:
    if runs_root is not None:
        return Path(runs_root)
    output_root = pilot.get("output_root")
    if not isinstance(output_root, str) or output_root == "":
        raise ValueError("pilot_manifest output_root must be a non-empty string")
    root = Path(output_root)
    if not root.is_absolute():
        workspace_relative = root
        manifest_relative = pilot_manifest.parent / root
        root = (
            workspace_relative
            if workspace_relative.exists() or not manifest_relative.exists()
            else manifest_relative
        )
    return root / "runs"


def _load_role_summaries(runs_root: Path) -> dict[tuple[str, str, int], list[dict[str, Any]]]:
    by_key: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    if not runs_root.is_dir():
        return {}
    for path in sorted(runs_root.glob("*/role_summary.json")):
        summary = json.loads(path.read_text())
        if not isinstance(summary, dict):
            raise ValueError(f"role summary must be an object: {path}")
        family = _string(summary.get("generated_family"), "generated_family")
        name = _string(summary.get("generated_name"), "generated_name")
        seed = _int(summary.get("generated_seed"), "generated_seed")
        summary["_summary_path"] = str(path)
        summary["_summary_mtime"] = path.stat().st_mtime
        by_key[(family, name, seed)].append(summary)
    for summaries in by_key.values():
        summaries.sort(key=lambda item: (float(item["_summary_mtime"]), item["_summary_path"]))
    return by_key


def _summarize_entry(
    entry: Mapping[str, Any],
    *,
    role_summaries: Mapping[tuple[str, str, int], list[dict[str, Any]]],
) -> dict[str, Any]:
    family = _string(entry.get("family"), "family")
    name = _string(entry.get("name"), "name")
    seed = _int(entry.get("seed"), "seed")
    base = {
        "bucket": _string(entry.get("bucket"), "bucket"),
        "family": family,
        "name": name,
        "seed": seed,
        "strata": _string_list(entry.get("strata", []), "strata"),
        "verification_labels": _string_list(
            entry.get("verification_labels", []),
            "verification_labels",
        ),
        "mcts_role_bias": _optional_number(entry.get("mcts_role_bias")),
        "mcts_seat_bias": _optional_number(entry.get("mcts_seat_bias")),
        "mcts_max_ply_rate": _optional_number(entry.get("mcts_max_ply_rate")),
    }
    matches = role_summaries.get((family, name, seed), [])
    if not matches:
        return {
            **base,
            "completed": False,
        }

    summary = matches[-1]
    shared = _variant(summary, "shared_heads")
    role = _variant(summary, "role_heads")
    shared_win = _number(
        shared.get("final_eval_model_win_rate_mean"),
        "shared_heads final_eval_model_win_rate_mean",
    )
    role_win = _number(
        role.get("final_eval_model_win_rate_mean"),
        "role_heads final_eval_model_win_rate_mean",
    )
    result = {
        **base,
        "completed": True,
        "run_dir": _string(summary.get("run_dir"), "run_dir"),
        "role_summary_path": str(summary["_summary_path"]),
        "device_used": summary.get("device_used"),
        "architecture_delta": round(role_win - shared_win, 6),
    }
    for variant_name, variant_summary in (
        ("shared_heads", shared),
        ("role_heads", role),
    ):
        prefix = variant_name
        for source_field, target_field in SUMMARY_FIELDS.items():
            result[f"{prefix}_{target_field}"] = _number(
                variant_summary.get(source_field),
                f"{variant_name} {source_field}",
            )
    seed_deltas = _paired_seed_deltas(summary)
    result["seed_architecture_deltas"] = seed_deltas
    result.update(_seed_delta_summary(seed_deltas))
    return result


def _variant(summary: Mapping[str, Any], variant: str) -> Mapping[str, Any]:
    by_variant = summary.get("by_variant")
    if not isinstance(by_variant, Mapping):
        raise ValueError("role summary by_variant must be an object")
    data = by_variant.get(variant)
    if not isinstance(data, Mapping):
        raise ValueError(f"role summary missing variant: {variant}")
    return data


def _aggregate_group(entries: Iterable[Mapping[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for entry in entries:
        groups[_string(entry.get(field), field)].append(entry)
    return {
        key: _aggregate_entries(value)
        for key, value in sorted(groups.items())
    }


def _aggregate_bucket_family(entries: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for entry in entries:
        key = f"{_string(entry.get('bucket'), 'bucket')}::{_string(entry.get('family'), 'family')}"
        groups[key].append(entry)
    return {
        key: _aggregate_entries(value)
        for key, value in sorted(groups.items())
    }


def _aggregate_entries(entries: list[Mapping[str, Any]]) -> dict[str, Any]:
    completed = [entry for entry in entries if entry.get("completed") is True]
    aggregate = {
        "entries": len(entries),
        "completed": len(completed),
        "missing": len(entries) - len(completed),
    }
    if completed:
        entry_deltas = [float(entry["architecture_delta"]) for entry in completed]
        seed_deltas = [
            float(row["architecture_delta"])
            for entry in completed
            for row in entry.get("seed_architecture_deltas", [])
            if isinstance(row, Mapping) and "architecture_delta" in row
        ]
        aggregate.update(
            {
                "mean_architecture_delta": _mean(
                    entry["architecture_delta"] for entry in completed
                ),
                "mean_shared_heads_win_rate": _mean(
                    entry["shared_heads_win_rate"] for entry in completed
                ),
                "mean_role_heads_win_rate": _mean(
                    entry["role_heads_win_rate"] for entry in completed
                ),
                "mean_shared_heads_draw_rate": _mean(
                    entry["shared_heads_draw_rate"] for entry in completed
                ),
                "mean_role_heads_draw_rate": _mean(
                    entry["role_heads_draw_rate"] for entry in completed
                ),
                "positive_entry_delta_count": sum(
                    1 for value in entry_deltas if value > 0.0
                ),
                "negative_entry_delta_count": sum(
                    1 for value in entry_deltas if value < 0.0
                ),
                "zero_entry_delta_count": sum(
                    1 for value in entry_deltas if value == 0.0
                ),
                "entry_sign_stability": _sign_stability(entry_deltas),
            }
        )
        aggregate.update(
            _value_delta_summary(
                seed_deltas,
                count_key="pooled_seed_delta_count",
                mean_key="mean_pooled_seed_architecture_delta",
                stdev_key="pooled_seed_architecture_delta_stdev",
                ci_key="pooled_seed_architecture_delta_ci95",
                positive_key="pooled_positive_seed_delta_count",
                negative_key="pooled_negative_seed_delta_count",
                zero_key="pooled_zero_seed_delta_count",
                stability_key="pooled_seed_sign_stability",
            )
        )
    return aggregate


def _paired_seed_deltas(summary: Mapping[str, Any]) -> list[dict[str, float | int]]:
    shared_rows = _final_rows(_variant(summary, "shared_heads"), "shared_heads")
    role_rows = _final_rows(_variant(summary, "role_heads"), "role_heads")
    shared_by_seed = {
        _int(row.get("seed"), "shared_heads final row seed"): row
        for row in shared_rows
    }
    role_by_seed = {
        _int(row.get("seed"), "role_heads final row seed"): row
        for row in role_rows
    }
    paired = []
    for seed in sorted(set(shared_by_seed) & set(role_by_seed)):
        shared = shared_by_seed[seed]
        role = role_by_seed[seed]
        shared_win = _number(
            shared.get("eval_model_win_rate"),
            "shared_heads final row eval_model_win_rate",
        )
        role_win = _number(
            role.get("eval_model_win_rate"),
            "role_heads final row eval_model_win_rate",
        )
        paired.append(
            {
                "seed": seed,
                "shared_heads_win_rate": shared_win,
                "role_heads_win_rate": role_win,
                "architecture_delta": round(role_win - shared_win, 6),
                "shared_heads_draw_rate": _number(
                    shared.get("eval_draw_rate"),
                    "shared_heads final row eval_draw_rate",
                ),
                "role_heads_draw_rate": _number(
                    role.get("eval_draw_rate"),
                    "role_heads final row eval_draw_rate",
                ),
            }
        )
    return paired


def _final_rows(
    variant_summary: Mapping[str, Any],
    variant_name: str,
) -> list[Mapping[str, Any]]:
    rows = variant_summary.get("final_rows", [])
    if rows is None:
        return []
    if not isinstance(rows, list):
        raise ValueError(f"{variant_name} final_rows must be a list")
    final_rows = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise ValueError(f"{variant_name} final_rows entry {index} must be an object")
        final_rows.append(row)
    return final_rows


def _seed_delta_summary(
    seed_deltas: list[Mapping[str, Any]],
) -> dict[str, Any]:
    return _value_delta_summary(
        [
            _number(row.get("architecture_delta"), "seed architecture_delta")
            for row in seed_deltas
        ],
        count_key="seed_delta_count",
        mean_key="mean_seed_architecture_delta",
        stdev_key="seed_architecture_delta_stdev",
        ci_key="seed_architecture_delta_ci95",
        positive_key="positive_seed_delta_count",
        negative_key="negative_seed_delta_count",
        zero_key="zero_seed_delta_count",
        stability_key="seed_sign_stability",
    )


def _value_delta_summary(
    values: Iterable[float],
    *,
    count_key: str,
    mean_key: str,
    stdev_key: str,
    ci_key: str,
    positive_key: str,
    negative_key: str,
    zero_key: str,
    stability_key: str,
) -> dict[str, Any]:
    deltas = [float(value) for value in values]
    if not deltas:
        return {
            count_key: 0,
            mean_key: None,
            stdev_key: None,
            ci_key: None,
            positive_key: 0,
            negative_key: 0,
            zero_key: 0,
            stability_key: None,
        }
    return {
        count_key: len(deltas),
        mean_key: _mean(deltas),
        stdev_key: _stdev(deltas),
        ci_key: _bootstrap_mean_ci(deltas),
        positive_key: sum(1 for value in deltas if value > 0.0),
        negative_key: sum(1 for value in deltas if value < 0.0),
        zero_key: sum(1 for value in deltas if value == 0.0),
        stability_key: _sign_stability(deltas),
    }


def _stdev(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return round(math.sqrt(variance), 6)


def _bootstrap_mean_ci(values: Iterable[float]) -> list[float]:
    values = [float(value) for value in values]
    if not values:
        return []
    if len(values) == 1:
        value = round(values[0], 6)
        return [value, value]
    rng = random.Random(0)
    means = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    return [
        round(_quantile(means, 0.025), 6),
        round(_quantile(means, 0.975), 6),
    ]


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    position = probability * (len(values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _sign_stability(values: Iterable[float]) -> float | None:
    values = [float(value) for value in values]
    if not values:
        return None
    positive = sum(1 for value in values if value > 0.0)
    negative = sum(1 for value in values if value < 0.0)
    zero = len(values) - positive - negative
    return round(max(positive, negative, zero) / len(values), 6)


def _mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def _string(value: Any, field: str) -> str:
    if type(value) is not str or value == "":
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return [str(item) for item in value]


def _int(value: Any, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an int")
    return value


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric")
    return float(value)


def _optional_number(value: Any) -> float | None:
    if value is None:
        return None
    return _number(value, "optional metric")


if __name__ == "__main__":
    raise SystemExit(main())
