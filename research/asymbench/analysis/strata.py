from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


STRATA = (
    "clean_control",
    "hidden_collapse",
    "role_inversion",
    "seat_confound",
    "horizon_stress",
    "role_collapse",
)


@dataclass(frozen=True)
class ClassifiedValidation:
    family: str
    name: str
    seed: int
    valid: bool
    labels: tuple[str, ...]
    random_role_bias: float
    mcts_role_bias: float
    mcts_seat_bias: float
    random_max_ply_rate: float
    mcts_max_ply_rate: float
    role_inversion_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "name": self.name,
            "seed": self.seed,
            "valid": self.valid,
            "labels": list(self.labels),
            "random_role_bias": self.random_role_bias,
            "mcts_role_bias": self.mcts_role_bias,
            "mcts_seat_bias": self.mcts_seat_bias,
            "random_max_ply_rate": self.random_max_ply_rate,
            "mcts_max_ply_rate": self.mcts_max_ply_rate,
            "role_inversion_score": self.role_inversion_score,
        }


@dataclass(frozen=True)
class ClassifiedValidationEntry:
    record: ClassifiedValidation
    spec_path: Path
    validation_path: Path
    source_root: Path

    def to_manifest_entry(
        self,
        *,
        rank: int,
        embed_spec: bool = False,
        verification: VerificationMetrics | None = None,
        verification_thresholds: VerificationThresholds | None = None,
    ) -> dict[str, Any]:
        entry = self.record.to_dict()
        entry.update(
            {
                "rank": rank,
                "source_root": str(self.source_root),
                "spec_path": str(self.spec_path),
                "validation_path": str(self.validation_path),
            }
        )
        if embed_spec:
            entry["spec"] = json.loads(self.spec_path.read_text())
        if verification is not None:
            if verification_thresholds is None:
                verification_thresholds = VerificationThresholds()
            entry["verification_metrics"] = verification.to_dict()
            entry["verification_labels"] = _verification_labels(
                self.record,
                verification,
                verification_thresholds,
            )
        return entry


@dataclass(frozen=True)
class StrataThresholds:
    main_band_role_bias: float = 0.60
    main_band_seat_bias: float = 0.15
    main_band_mcts_max_ply_rate: float = 0.50
    hidden_collapse_random_min_role_win_rate: float = 0.25
    hidden_collapse_mcts_role_bias: float = 0.90
    seat_confound_bias: float = 0.25
    horizon_stress_rate: float = 0.70


@dataclass(frozen=True)
class VerificationThresholds:
    strict_role_bias: float = 0.25
    strict_seat_bias: float = 0.20
    strict_mcts_max_ply_rate: float = 0.50
    near_role_bias: float = 0.60
    near_seat_bias: float = 0.20
    near_mcts_max_ply_rate: float = 0.60
    seat_sensitive_bias: float = 0.25
    high_sim_collapse_role_bias: float = 0.80
    verified_role_inversion_score: float = 0.40


@dataclass(frozen=True)
class VerificationMetrics:
    name: str
    seed: int
    mcts_role0: float
    mcts_role1: float
    mcts_first_player_win_rate: float
    mcts_role_bias: float
    mcts_seat_bias: float
    mcts_max_ply_rate: float
    role_inversion_score: float
    source_path: Path
    mcts_terminal_reasons: dict[str, int]
    average_mcts_plies: float | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "seed": self.seed,
            "mcts_role_win_rates": {
                "0": self.mcts_role0,
                "1": self.mcts_role1,
            },
            "mcts_first_player_win_rate": self.mcts_first_player_win_rate,
            "mcts_role_bias": self.mcts_role_bias,
            "mcts_seat_bias": self.mcts_seat_bias,
            "mcts_max_ply_rate": self.mcts_max_ply_rate,
            "role_inversion_score": self.role_inversion_score,
            "mcts_terminal_reasons": dict(self.mcts_terminal_reasons),
            "source_path": str(self.source_path),
        }
        if self.average_mcts_plies is not None:
            data["average_mcts_plies"] = self.average_mcts_plies
        return data


def classify_generated_validation(
    *,
    spec: Mapping[str, Any],
    report: Mapping[str, Any],
    thresholds: StrataThresholds = StrataThresholds(),
) -> ClassifiedValidation:
    family = _require_string(spec.get("family"), "spec family")
    name = _require_string(spec.get("name"), "spec name")
    seed = _require_int(spec.get("seed"), "spec seed")
    valid = bool(report.get("valid", False))

    random_role0 = _role_rate(report, "random_role_win_rates", "0")
    random_role1 = _role_rate(report, "random_role_win_rates", "1")
    mcts_role0 = _role_rate(report, "mcts_role_win_rates", "0")
    mcts_role1 = _role_rate(report, "mcts_role_win_rates", "1")
    mcts_first = _probability(report.get("mcts_first_player_win_rate", 0.0), "mcts_first_player_win_rate")

    random_total = max(1, sum(_reason_counts(report.get("terminal_reasons", {})).values()))
    mcts_total = max(1, sum(_reason_counts(report.get("mcts_terminal_reasons", {})).values()))
    random_max_rate = _reason_counts(report.get("terminal_reasons", {})).get("max_plies", 0) / random_total
    mcts_max_rate = _reason_counts(report.get("mcts_terminal_reasons", {})).get("max_plies", 0) / mcts_total

    random_bias = abs(random_role0 - random_role1)
    mcts_bias = abs(mcts_role0 - mcts_role1)
    seat_bias = abs(2.0 * mcts_first - 1.0)
    inversion_score = _role_inversion_score(random_role0, mcts_role0)

    labels = []
    if (
        valid
        and mcts_bias <= thresholds.main_band_role_bias
        and seat_bias <= thresholds.main_band_seat_bias
        and mcts_max_rate <= thresholds.main_band_mcts_max_ply_rate
    ):
        labels.append("clean_control")
    if (
        valid
        and min(random_role0, random_role1)
        >= thresholds.hidden_collapse_random_min_role_win_rate
        and mcts_bias >= thresholds.hidden_collapse_mcts_role_bias
    ):
        labels.append("hidden_collapse")
    if valid and inversion_score > 0.0:
        labels.append("role_inversion")
    if (
        valid
        and seat_bias >= thresholds.seat_confound_bias
        and seat_bias >= mcts_bias
    ):
        labels.append("seat_confound")
    if (
        valid
        and max(random_max_rate, mcts_max_rate) >= thresholds.horizon_stress_rate
    ):
        labels.append("horizon_stress")
    if valid and mcts_bias >= thresholds.hidden_collapse_mcts_role_bias:
        labels.append("role_collapse")

    return ClassifiedValidation(
        family=family,
        name=name,
        seed=seed,
        valid=valid,
        labels=tuple(labels),
        random_role_bias=random_bias,
        mcts_role_bias=mcts_bias,
        mcts_seat_bias=seat_bias,
        random_max_ply_rate=random_max_rate,
        mcts_max_ply_rate=mcts_max_rate,
        role_inversion_score=inversion_score,
    )


def load_classified_validations(root: Path) -> list[ClassifiedValidation]:
    return [
        entry.record
        for entry in load_classified_validation_entries([root], require_mcts=False)
    ]


def load_classified_validation_entries(
    roots: Iterable[Path],
    *,
    require_mcts: bool = True,
    thresholds: StrataThresholds = StrataThresholds(),
) -> list[ClassifiedValidationEntry]:
    records = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"input root is not a directory: {root}")
        for spec_path in sorted(root.glob("*/spec.json")):
            validation_path = spec_path.with_name("validation.json")
            if not validation_path.is_file():
                continue
            spec = json.loads(spec_path.read_text())
            report = json.loads(validation_path.read_text())
            if require_mcts and not _has_mcts_diagnostics(report):
                continue
            records.append(
                ClassifiedValidationEntry(
                    record=classify_generated_validation(
                        spec=spec,
                        report=report,
                        thresholds=thresholds,
                    ),
                    spec_path=spec_path,
                    validation_path=validation_path,
                    source_root=root,
                )
            )
    return records


def build_selection_manifest(
    *,
    input_roots: Iterable[Path],
    limit_per_stratum: int = 10,
    thresholds: StrataThresholds = StrataThresholds(),
    verification_thresholds: VerificationThresholds = VerificationThresholds(),
    verification_reports: Iterable[Path] | None = None,
    require_mcts: bool = True,
    embed_specs: bool = False,
) -> dict[str, Any]:
    input_roots = [Path(root) for root in input_roots]
    verification_report_paths = [
        Path(path) for path in (verification_reports or [])
    ]
    entries = load_classified_validation_entries(
        input_roots,
        require_mcts=require_mcts,
        thresholds=thresholds,
    )
    if not entries:
        raise ValueError("no classified validation records found")
    entries_by_record_id = {id(entry.record): entry for entry in entries}
    ranked = rank_strata(
        (entry.record for entry in entries),
        limit_per_stratum=limit_per_stratum,
    )
    verification_by_key = load_verification_metrics(verification_report_paths)
    return {
        "schema_version": 2,
        "input_roots": [str(root) for root in input_roots],
        "thresholds": asdict(thresholds),
        "verification_thresholds": asdict(verification_thresholds),
        "verification_reports": [str(path) for path in verification_report_paths],
        "limit_per_stratum": limit_per_stratum,
        "require_mcts": require_mcts,
        "embed_specs": embed_specs,
        "strata": {
            stratum: [
                entries_by_record_id[id(record)].to_manifest_entry(
                    rank=rank,
                    embed_spec=embed_specs,
                    verification=verification_by_key.get(
                        _verification_key(record.name, record.seed)
                    ),
                    verification_thresholds=verification_thresholds,
                )
                for rank, record in enumerate(records, start=1)
            ]
            for stratum, records in ranked.items()
        },
    }


def rank_strata(
    records: Iterable[ClassifiedValidation],
    *,
    limit_per_stratum: int = 10,
) -> dict[str, list[ClassifiedValidation]]:
    if limit_per_stratum <= 0:
        raise ValueError("limit_per_stratum must be positive")
    records = list(records)
    return {
        stratum: sorted(
            (record for record in records if stratum in record.labels),
            key=lambda record: _rank_key(stratum, record),
        )[:limit_per_stratum]
        for stratum in STRATA
    }


def _rank_key(stratum: str, record: ClassifiedValidation) -> tuple[float, ...]:
    if stratum == "clean_control":
        return (
            record.mcts_role_bias,
            record.mcts_seat_bias,
            record.mcts_max_ply_rate,
            float(record.seed),
        )
    if stratum == "hidden_collapse":
        return (-record.mcts_role_bias, record.random_role_bias, float(record.seed))
    if stratum == "role_inversion":
        return (-record.role_inversion_score, float(record.seed))
    if stratum == "seat_confound":
        return (-record.mcts_seat_bias, record.mcts_role_bias, float(record.seed))
    if stratum == "horizon_stress":
        return (
            -max(record.random_max_ply_rate, record.mcts_max_ply_rate),
            float(record.seed),
        )
    if stratum == "role_collapse":
        return (-record.mcts_role_bias, record.mcts_seat_bias, float(record.seed))
    return (float(record.seed),)


def write_role_head_configs(
    *,
    manifest: dict[str, Any],
    template_path: Path,
    output_root: Path,
) -> None:
    template = json.loads(Path(template_path).read_text())
    if not isinstance(template, dict):
        raise ValueError("role config template must be a JSON object")
    output_root = Path(output_root)
    for stratum, entries in manifest["strata"].items():
        stratum_dir = output_root / stratum
        for entry in entries:
            config = json.loads(json.dumps(template))
            config.pop("game", None)
            config["game_source"] = {
                "type": "generated_spec",
                "path": entry["spec_path"],
            }
            filename = _safe_filename(f"{entry['rank']:03d}_{entry['name']}.json")
            config_path = stratum_dir / filename
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                json.dumps(config, indent=2, sort_keys=True) + "\n"
            )
            entry["role_config_path"] = str(config_path)


def load_verification_metrics(
    paths: Iterable[Path],
) -> dict[tuple[str, int], VerificationMetrics]:
    metrics = {}
    for path in paths:
        path = Path(path)
        data = json.loads(path.read_text())
        if not isinstance(data, list):
            raise ValueError(f"verification report must be a JSON list: {path}")
        for index, item in enumerate(data, start=1):
            if not isinstance(item, Mapping):
                raise ValueError(
                    f"verification report item {index} must be an object: {path}"
                )
            verification = _verification_metrics_from_mapping(item, source_path=path)
            metrics[_verification_key(verification.name, verification.seed)] = verification
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Select ranked AsymBench validation strata."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        type=Path,
        dest="inputs",
        help="Generated-game output root containing */spec.json and */validation.json.",
    )
    parser.add_argument(
        "--limit-per-stratum",
        type=_positive_int,
        default=10,
        help="Maximum number of entries to export per stratum.",
    )
    parser.add_argument("--output", type=Path, help="Optional manifest JSON path.")
    parser.add_argument(
        "--verification-report",
        action="append",
        type=Path,
        default=[],
        help="Optional high-simulation verification JSON list.",
    )
    parser.add_argument(
        "--embed-specs",
        action="store_true",
        help="Embed spec.json contents in each manifest entry.",
    )
    parser.add_argument(
        "--include-random-only",
        action="store_true",
        help="Allow reports without MCTS diagnostics.",
    )
    parser.add_argument(
        "--role-config-template",
        type=Path,
        help="Optional role-head runner config template to clone per selected entry.",
    )
    parser.add_argument(
        "--role-config-output",
        type=Path,
        help="Output directory for generated role-head configs.",
    )
    args = parser.parse_args(argv)

    try:
        if bool(args.role_config_template) != bool(args.role_config_output):
            raise ValueError(
                "--role-config-template and --role-config-output must be provided together"
            )
        manifest = build_selection_manifest(
            input_roots=args.inputs,
            limit_per_stratum=args.limit_per_stratum,
            require_mcts=not args.include_random_only,
            verification_reports=args.verification_report,
            embed_specs=args.embed_specs,
        )
        if args.role_config_template is not None:
            write_role_head_configs(
                manifest=manifest,
                template_path=args.role_config_template,
                output_root=args.role_config_output,
            )
        output = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        if args.output is None:
            print(output, end="")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _verification_metrics_from_mapping(
    data: Mapping[str, Any],
    *,
    source_path: Path,
) -> VerificationMetrics:
    name = _require_string(data.get("name"), "verification name")
    seed = _require_int(data.get("seed"), "verification seed")
    if "mcts64_role0" in data or "mcts64_role1" in data:
        role0 = _probability(data.get("mcts64_role0"), "mcts64_role0")
        role1 = _probability(data.get("mcts64_role1"), "mcts64_role1")
        first_player = _probability(
            data.get("mcts64_first_player", 0.0),
            "mcts64_first_player",
        )
        terminal_reasons = _reason_counts(data.get("mcts64_terminal_reasons", {}))
        max_rate = _probability(
            data.get(
                "mcts64_max",
                _max_ply_rate(terminal_reasons),
            ),
            "mcts64_max",
        )
        average_plies = _optional_number(data.get("mcts64_avg_plies"))
        random_role0 = _optional_probability(data.get("random_role0_previous"))
    else:
        rates = data.get("mcts_role_win_rates", {})
        if not isinstance(rates, Mapping):
            raise ValueError("mcts_role_win_rates must be a mapping")
        role0 = _probability(rates.get("0"), "mcts_role_win_rates['0']")
        role1 = _probability(rates.get("1"), "mcts_role_win_rates['1']")
        first_player = _probability(
            data.get("mcts_first_player_win_rate", 0.0),
            "mcts_first_player_win_rate",
        )
        terminal_reasons = _reason_counts(data.get("mcts_terminal_reasons", {}))
        max_rate = _probability(
            data.get(
                "mcts_max_ply_rate",
                _max_ply_rate(terminal_reasons),
            ),
            "mcts_max_ply_rate",
        )
        average_plies = _optional_number(data.get("average_mcts_plies"))
        random_role0 = None
        random_rates = data.get("random_role_win_rates", {})
        if isinstance(random_rates, Mapping) and "0" in random_rates:
            random_role0 = _probability(
                random_rates.get("0"),
                "random_role_win_rates['0']",
            )

    role_bias = abs(role0 - role1)
    seat_bias = abs(2.0 * first_player - 1.0)
    inversion_score = (
        _role_inversion_score(random_role0, role0)
        if random_role0 is not None
        else 0.0
    )
    return VerificationMetrics(
        name=name,
        seed=seed,
        mcts_role0=role0,
        mcts_role1=role1,
        mcts_first_player_win_rate=first_player,
        mcts_role_bias=role_bias,
        mcts_seat_bias=seat_bias,
        mcts_max_ply_rate=max_rate,
        role_inversion_score=inversion_score,
        source_path=source_path,
        mcts_terminal_reasons=terminal_reasons,
        average_mcts_plies=average_plies,
    )


def _verification_labels(
    record: ClassifiedValidation,
    metrics: VerificationMetrics,
    thresholds: VerificationThresholds,
) -> list[str]:
    labels = []
    if (
        metrics.mcts_role_bias <= thresholds.strict_role_bias
        and metrics.mcts_seat_bias <= thresholds.strict_seat_bias
        and metrics.mcts_max_ply_rate <= thresholds.strict_mcts_max_ply_rate
    ):
        labels.append("strict_clean")
    elif (
        metrics.mcts_role_bias <= thresholds.near_role_bias
        and metrics.mcts_seat_bias <= thresholds.near_seat_bias
        and metrics.mcts_max_ply_rate <= thresholds.near_mcts_max_ply_rate
    ):
        labels.append("near_clean")
    if (
        metrics.mcts_role_bias <= thresholds.near_role_bias
        and metrics.mcts_seat_bias >= thresholds.seat_sensitive_bias
    ):
        labels.append("seat_sensitive")
    if metrics.mcts_role_bias >= thresholds.high_sim_collapse_role_bias:
        labels.append("high_sim_collapsed")
    if (
        "hidden_collapse" in record.labels
        and metrics.mcts_role_bias >= thresholds.high_sim_collapse_role_bias
    ):
        labels.append("verified_hidden_collapse")
    if metrics.role_inversion_score >= thresholds.verified_role_inversion_score:
        labels.append("verified_role_inversion")
    return labels


def _verification_key(name: str, seed: int) -> tuple[str, int]:
    return name, seed


def _max_ply_rate(terminal_reasons: Mapping[str, int]) -> float:
    total = sum(terminal_reasons.values())
    if total <= 0:
        return 0.0
    return terminal_reasons.get("max_plies", 0) / total


def _has_mcts_diagnostics(report: Mapping[str, Any]) -> bool:
    mcts_rates = report.get("mcts_role_win_rates")
    mcts_reasons = report.get("mcts_terminal_reasons")
    return (
        isinstance(mcts_rates, Mapping)
        and "0" in mcts_rates
        and "1" in mcts_rates
        and isinstance(mcts_reasons, Mapping)
        and bool(mcts_reasons)
    )


def _safe_filename(value: str) -> str:
    safe = []
    for char in value:
        if char.isalnum() or char in {"-", "_", "."}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe)


def _role_inversion_score(random_role0: float, mcts_role0: float) -> float:
    random_advantage = random_role0 - 0.5
    mcts_advantage = mcts_role0 - 0.5
    if random_advantage * mcts_advantage >= 0.0:
        return 0.0
    return abs(random_role0 - mcts_role0)


def _role_rate(report: Mapping[str, Any], field: str, role: str) -> float:
    rates = report.get(field, {})
    if not isinstance(rates, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return _probability(rates.get(role, 0.0), f"{field}[{role!r}]")


def _reason_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError("terminal reasons must be a mapping")
    counts = {}
    for key, count in value.items():
        if not isinstance(key, str):
            raise ValueError("terminal reason keys must be strings")
        if isinstance(count, bool) or not isinstance(count, int):
            raise ValueError("terminal reason counts must be ints")
        if count < 0:
            raise ValueError("terminal reason counts must be non-negative")
        counts[key] = count
    return counts


def _probability(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric")
    value = float(value)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{field} must be in [0.0, 1.0]")
    return value


def _optional_probability(value: Any) -> float | None:
    if value is None:
        return None
    return _probability(value, "optional probability")


def _optional_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("optional number must be numeric")
    return float(value)


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an int")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
