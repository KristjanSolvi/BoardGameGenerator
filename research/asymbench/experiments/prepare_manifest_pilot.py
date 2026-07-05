from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from research.asymbench.generation.specs import GeneratedGameSpec


BUCKETS = (
    "clean",
    "role_inversion",
    "collapse",
    "seat_sensitive",
    "horizon_stress",
)
STRATA_ORDER = (
    "clean_control",
    "hidden_collapse",
    "horizon_stress",
    "role_collapse",
    "role_inversion",
    "seat_confound",
)
DEFAULT_TEMPLATE: dict[str, Any] = {
    "device": "cuda",
    "seeds": [101, 202],
    "model_variants": ["shared_heads", "role_heads"],
    "iterations": 3,
    "selfplay_games_per_iteration": 4,
    "train_steps_per_iteration": 8,
    "batch_size": 16,
    "replay_capacity": 1024,
    "mcts_simulations": 8,
    "eval_games": 8,
    "eval_simulations": 8,
    "learning_rate": 0.001,
}


@dataclass(frozen=True)
class ManifestCandidate:
    source_manifest: Path
    entry: dict[str, Any]
    family: str
    name: str
    seed: int
    strata: tuple[str, ...]
    verification_labels: tuple[str, ...]

    @property
    def key(self) -> tuple[str, str, int]:
        return self.family, self.name, self.seed

    def metric(self, field: str) -> float:
        verification = self.entry.get("verification_metrics")
        if isinstance(verification, Mapping) and field in verification:
            return _number(verification[field], field)
        if field in self.entry:
            return _number(self.entry[field], field)
        return 0.0


def build_manifest_pilot(
    *,
    manifest_paths: Iterable[Path],
    output_root: Path,
    template: Mapping[str, Any] | None = None,
    per_bucket_per_family: int = 1,
) -> dict[str, Any]:
    if per_bucket_per_family <= 0:
        raise ValueError("per_bucket_per_family must be positive")
    output_root = Path(output_root)
    template_data = _template_copy(template or DEFAULT_TEMPLATE)
    candidates = _load_candidates([Path(path) for path in manifest_paths])
    if not candidates:
        raise ValueError("no manifest candidates found")

    selected = _select_candidates(
        candidates,
        per_bucket_per_family=per_bucket_per_family,
    )
    if not selected:
        raise ValueError("no pilot entries matched the configured buckets")

    entries = []
    for bucket, candidate in selected:
        entries.append(
            _write_candidate_artifacts(
                bucket=bucket,
                candidate=candidate,
                output_root=output_root,
                template=template_data,
            )
        )

    pilot = {
        "schema_version": 1,
        "buckets": list(BUCKETS),
        "per_bucket_per_family": per_bucket_per_family,
        "input_manifests": [str(Path(path)) for path in manifest_paths],
        "output_root": str(output_root),
        "template": template_data,
        "entries": entries,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "pilot_manifest.json", pilot)
    return pilot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare role-head AlphaZero-lite configs from benchmark manifests."
    )
    parser.add_argument(
        "--manifest",
        action="append",
        type=Path,
        required=True,
        dest="manifests",
        help="Committed benchmark manifest JSON with embedded specs.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Directory where specs, configs, and pilot_manifest.json are written.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        help="Optional run_role_heads JSON template. The game field is replaced.",
    )
    parser.add_argument(
        "--per-bucket-per-family",
        type=_positive_int,
        default=1,
        help="Maximum entries to select per bucket and family.",
    )
    args = parser.parse_args(argv)

    try:
        template = _load_template(args.template) if args.template else None
        pilot = build_manifest_pilot(
            manifest_paths=args.manifests,
            output_root=args.output_root,
            template=template,
            per_bucket_per_family=args.per_bucket_per_family,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"pilot_manifest={args.output_root / 'pilot_manifest.json'}")
    print(f"configs={len(pilot['entries'])}")
    return 0


def _load_template(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"template must be a JSON object: {path}")
    return data


def _load_candidates(paths: list[Path]) -> list[ManifestCandidate]:
    merged: dict[tuple[str, str, int], dict[str, Any]] = {}
    for path in paths:
        manifest = json.loads(path.read_text())
        strata = manifest.get("strata")
        if not isinstance(strata, Mapping):
            raise ValueError(f"manifest strata must be an object: {path}")
        for stratum in STRATA_ORDER:
            entries = strata.get(stratum, [])
            if not isinstance(entries, list):
                raise ValueError(f"manifest stratum must be a list: {path} {stratum}")
            for entry in entries:
                if not isinstance(entry, dict):
                    raise ValueError(f"manifest entry must be an object: {path}")
                family = _string(entry.get("family"), "family")
                name = _string(entry.get("name"), "name")
                seed = _int(entry.get("seed"), "seed")
                key = (family, name, seed)
                existing = merged.setdefault(
                    key,
                    {
                        "source_manifest": path,
                        "entry": entry,
                        "strata": set(),
                        "verification_labels": set(),
                    },
                )
                existing["strata"].add(stratum)
                labels = entry.get("verification_labels", [])
                if isinstance(labels, list):
                    existing["verification_labels"].update(
                        str(label) for label in labels
                    )
    candidates = []
    for (family, name, seed), item in merged.items():
        candidates.append(
            ManifestCandidate(
                source_manifest=Path(item["source_manifest"]),
                entry=dict(item["entry"]),
                family=family,
                name=name,
                seed=seed,
                strata=tuple(
                    stratum for stratum in STRATA_ORDER if stratum in item["strata"]
                ),
                verification_labels=tuple(sorted(item["verification_labels"])),
            )
        )
    return candidates


def _select_candidates(
    candidates: list[ManifestCandidate],
    *,
    per_bucket_per_family: int,
) -> list[tuple[str, ManifestCandidate]]:
    families = sorted({candidate.family for candidate in candidates})
    selected: list[tuple[str, ManifestCandidate]] = []
    used: set[tuple[str, str, int]] = set()
    for bucket in BUCKETS:
        for family in families:
            matches = [
                candidate
                for candidate in candidates
                if candidate.family == family and _matches_bucket(candidate, bucket)
            ]
            picked = 0
            for candidate in sorted(matches, key=lambda item: _rank_key(bucket, item)):
                if candidate.key in used:
                    continue
                selected.append((bucket, candidate))
                used.add(candidate.key)
                picked += 1
                if picked >= per_bucket_per_family:
                    break
    return selected


def _matches_bucket(candidate: ManifestCandidate, bucket: str) -> bool:
    labels = set(candidate.verification_labels)
    strata = set(candidate.strata)
    if bucket == "clean":
        return bool(labels & {"strict_clean", "near_clean"}) or "clean_control" in strata
    if bucket == "role_inversion":
        return "verified_role_inversion" in labels or "role_inversion" in strata
    if bucket == "collapse":
        return bool(
            labels & {"high_sim_collapsed", "verified_hidden_collapse"}
        ) or bool(strata & {"hidden_collapse", "role_collapse"})
    if bucket == "seat_sensitive":
        return "seat_sensitive" in labels or "seat_confound" in strata
    if bucket == "horizon_stress":
        return "horizon_stress" in strata
    raise ValueError(f"unknown bucket: {bucket}")


def _rank_key(bucket: str, candidate: ManifestCandidate) -> tuple[float, ...]:
    role_bias = candidate.metric("mcts_role_bias")
    seat_bias = candidate.metric("mcts_seat_bias")
    max_ply_rate = candidate.metric("mcts_max_ply_rate")
    inversion = candidate.metric("role_inversion_score")
    labels = set(candidate.verification_labels)
    if bucket == "clean":
        label_rank = 0 if "strict_clean" in labels else 1 if "near_clean" in labels else 2
        return (label_rank, role_bias, seat_bias, max_ply_rate, float(candidate.seed))
    if bucket == "role_inversion":
        label_rank = 0 if "verified_role_inversion" in labels else 1
        return (label_rank, -inversion, -role_bias, float(candidate.seed))
    if bucket == "collapse":
        label_rank = (
            0
            if "verified_hidden_collapse" in labels
            else 1
            if "high_sim_collapsed" in labels
            else 2
        )
        return (label_rank, -role_bias, seat_bias, float(candidate.seed))
    if bucket == "seat_sensitive":
        label_rank = 0 if "seat_sensitive" in labels else 1
        return (label_rank, -seat_bias, role_bias, -max_ply_rate, float(candidate.seed))
    if bucket == "horizon_stress":
        return (-max_ply_rate, -seat_bias, role_bias, float(candidate.seed))
    return (float(candidate.seed),)


def _write_candidate_artifacts(
    *,
    bucket: str,
    candidate: ManifestCandidate,
    output_root: Path,
    template: Mapping[str, Any],
) -> dict[str, Any]:
    spec_data = candidate.entry.get("spec")
    if not isinstance(spec_data, dict):
        raise ValueError(
            f"selected entry requires an embedded spec: {candidate.name} seed {candidate.seed}"
        )
    spec = GeneratedGameSpec.from_dict(spec_data)

    basename = _safe_filename(f"{candidate.family}_{candidate.seed}_{candidate.name}")
    spec_path = output_root / "specs" / candidate.family / f"{basename}.json"
    config_path = (
        output_root
        / "configs"
        / bucket
        / candidate.family
        / f"{basename}.json"
    )
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(spec_path, spec.to_dict())

    config = _template_copy(template)
    config.pop("game", None)
    config["game_source"] = {
        "type": "generated_spec",
        "path": os.path.relpath(spec_path.resolve(), start=config_path.parent.resolve()),
    }
    config["output_root"] = str((output_root / "runs").resolve())
    config["pilot_metadata"] = {
        "bucket": bucket,
        "source_manifest": str(candidate.source_manifest),
        "family": candidate.family,
        "name": candidate.name,
        "seed": candidate.seed,
        "strata": list(candidate.strata),
        "verification_labels": list(candidate.verification_labels),
    }
    _write_json(config_path, config)

    verification_metrics = candidate.entry.get("verification_metrics", {})
    if not isinstance(verification_metrics, Mapping):
        verification_metrics = {}
    return {
        "bucket": bucket,
        "family": candidate.family,
        "name": candidate.name,
        "seed": candidate.seed,
        "source_manifest": str(candidate.source_manifest),
        "strata": list(candidate.strata),
        "verification_labels": list(candidate.verification_labels),
        "mcts_role_bias": candidate.metric("mcts_role_bias"),
        "mcts_seat_bias": candidate.metric("mcts_seat_bias"),
        "mcts_max_ply_rate": candidate.metric("mcts_max_ply_rate"),
        "role_inversion_score": candidate.metric("role_inversion_score"),
        "verification_metrics": dict(verification_metrics),
        "spec_path": str(spec_path),
        "config_path": str(config_path),
    }


def _template_copy(template: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(template, Mapping):
        raise ValueError("template must be a mapping")
    return copy.deepcopy(dict(template))


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _string(value: Any, field: str) -> str:
    if type(value) is not str or value == "":
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _int(value: Any, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an int")
    return value


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric")
    return float(value)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _safe_filename(value: str) -> str:
    safe = []
    for char in value:
        if char.isalnum() or char in {"-", "_", "."}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe)


if __name__ == "__main__":
    raise SystemExit(main())
