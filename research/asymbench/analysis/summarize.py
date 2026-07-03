from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = ("variant", "eval_model_win_rate", "eval_avg_plies")
OPTIONAL_NUMERIC_FIELDS = {
    "eval_random_win_rate": "mean_random_win_rate",
    "eval_draw_rate": "mean_draw_rate",
    "policy_loss": "mean_policy_loss",
    "value_loss": "mean_value_loss",
    "train_total_loss": "mean_train_total_loss",
}


def summarize_metrics(path: Path) -> dict[str, dict[str, Any]]:
    rows_by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rows_seen = 0

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"metrics row {line_number} must be a JSON object")
            _validate_required_fields(row, line_number)
            rows_by_variant[str(row["variant"])].append(row)
            rows_seen += 1

    if rows_seen == 0:
        raise ValueError(f"metrics file is empty: {path}")

    return {
        variant: _summarize_variant(rows)
        for variant, rows in sorted(rows_by_variant.items())
    }


def _validate_required_fields(row: dict[str, Any], line_number: int) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in row]
    if missing:
        raise ValueError(f"metrics row {line_number} missing required fields: {missing}")
    for field in ("eval_model_win_rate", "eval_avg_plies"):
        if not isinstance(row[field], int | float):
            raise ValueError(f"metrics row {line_number} field {field!r} must be numeric")


def _summarize_variant(rows: list[dict[str, Any]]) -> dict[str, Any]:
    last_row = rows[-1]
    summary: dict[str, Any] = {
        "rows": len(rows),
        "mean_model_win_rate": _mean(row["eval_model_win_rate"] for row in rows),
        "mean_avg_plies": _mean(row["eval_avg_plies"] for row in rows),
        "last_model_win_rate": float(last_row["eval_model_win_rate"]),
    }
    for source_field, summary_field in OPTIONAL_NUMERIC_FIELDS.items():
        values = [
            row[source_field]
            for row in rows
            if isinstance(row.get(source_field), int | float)
        ]
        if values:
            summary[summary_field] = _mean(values)
    if "eval_model_role_win_rates" in last_row:
        summary["last_eval_model_role_win_rates"] = last_row[
            "eval_model_role_win_rates"
        ]
    if "eval_termination_reasons" in last_row:
        summary["last_eval_termination_reasons"] = last_row[
            "eval_termination_reasons"
        ]
    return summary


def _mean(values: Any) -> float:
    values = list(values)
    return float(sum(values) / len(values))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize AsymBench metrics JSONL.")
    parser.add_argument("metrics", type=Path)
    args = parser.parse_args(argv)

    print(json.dumps(summarize_metrics(args.metrics), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
