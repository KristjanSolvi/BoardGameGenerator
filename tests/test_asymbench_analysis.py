import json
from pathlib import Path

from research.asymbench.analysis.summarize import summarize_metrics


def test_summarize_metrics_groups_by_variant(tmp_path: Path):
    metrics = tmp_path / "metrics.jsonl"
    rows = [
        {"variant": "shared_heads", "eval_model_win_rate": 0.25, "eval_avg_plies": 8},
        {"variant": "shared_heads", "eval_model_win_rate": 0.75, "eval_avg_plies": 10},
        {"variant": "role_heads", "eval_model_win_rate": 1.0, "eval_avg_plies": 7},
    ]
    metrics.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    summary = summarize_metrics(metrics)
    assert summary["shared_heads"]["mean_model_win_rate"] == 0.5
    assert summary["shared_heads"]["mean_avg_plies"] == 9.0
    assert summary["role_heads"]["mean_model_win_rate"] == 1.0
