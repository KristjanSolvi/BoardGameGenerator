from __future__ import annotations

import json
from dataclasses import dataclass
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
class StrataThresholds:
    main_band_role_bias: float = 0.60
    main_band_seat_bias: float = 0.15
    main_band_mcts_max_ply_rate: float = 0.50
    hidden_collapse_random_min_role_win_rate: float = 0.25
    hidden_collapse_mcts_role_bias: float = 0.90
    seat_confound_bias: float = 0.25
    horizon_stress_rate: float = 0.70


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
    records = []
    for spec_path in sorted(root.glob("*/spec.json")):
        validation_path = spec_path.with_name("validation.json")
        if not validation_path.is_file():
            continue
        spec = json.loads(spec_path.read_text())
        report = json.loads(validation_path.read_text())
        records.append(classify_generated_validation(spec=spec, report=report))
    return records


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


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an int")
    return value
