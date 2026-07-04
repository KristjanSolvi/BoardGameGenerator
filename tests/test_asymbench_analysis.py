import json
from pathlib import Path

import pytest

from research.asymbench.analysis.disagreement import (
    OutcomeVector,
    architecture_delta,
    evaluator_disagreement,
    hidden_role_collapse,
    role_inversion,
    role_seat_separation,
)
from research.asymbench.analysis.summarize import main, summarize_metrics


def test_evaluator_disagreement_uses_largest_pairwise_outcome_distance():
    score = evaluator_disagreement(
        {
            "random": OutcomeVector(role0=0.031, role1=0.969, draw=0.0),
            "mcts_vs_random": OutcomeVector(role0=0.375, role1=0.625, draw=0.0),
            "mcts_vs_mcts": OutcomeVector(role0=1.0, role1=0.0, draw=0.0),
        }
    )

    assert score == pytest.approx(0.969)


def test_role_seat_separation_distinguishes_role_bias_from_first_player_bias():
    metrics = role_seat_separation(
        OutcomeVector(role0=1.0, role1=0.0, draw=0.0),
        first_player_win_rate=0.5,
    )

    assert metrics.role_bias == pytest.approx(1.0)
    assert metrics.seat_bias == pytest.approx(0.0)
    assert metrics.role_seat_separation == pytest.approx(1.0)


def test_role_seat_separation_exposes_seat_confounds():
    metrics = role_seat_separation(
        OutcomeVector(role0=0.5417, role1=0.4583, draw=0.0),
        first_player_win_rate=0.8333,
    )

    assert metrics.role_bias == pytest.approx(0.0834)
    assert metrics.seat_bias == pytest.approx(0.6666)
    assert metrics.role_seat_separation == pytest.approx(-0.5832)


def test_hidden_role_collapse_flags_random_balanced_planning_collapse():
    score = hidden_role_collapse(
        random_outcome=OutcomeVector(role0=0.531, role1=0.469, draw=0.0),
        planned_outcome=OutcomeVector(role0=1.0, role1=0.0, draw=0.0),
    )

    assert score == pytest.approx(1.0)


def test_hidden_role_collapse_ignores_games_not_balanced_under_random_play():
    score = hidden_role_collapse(
        random_outcome=OutcomeVector(role0=0.031, role1=0.969, draw=0.0),
        planned_outcome=OutcomeVector(role0=1.0, role1=0.0, draw=0.0),
    )

    assert score == 0.0


def test_role_inversion_measures_role0_advantage_reversal_under_skill():
    score = role_inversion(
        random_outcome=OutcomeVector(role0=0.031, role1=0.969, draw=0.0),
        planned_outcome=OutcomeVector(role0=1.0, role1=0.0, draw=0.0),
    )

    assert score == pytest.approx(0.969)


def test_role_inversion_is_zero_without_sign_reversal():
    score = role_inversion(
        random_outcome=OutcomeVector(role0=0.531, role1=0.469, draw=0.0),
        planned_outcome=OutcomeVector(role0=1.0, role1=0.0, draw=0.0),
    )

    assert score == 0.0


def test_architecture_delta_is_role_heads_minus_shared_heads():
    assert architecture_delta(
        role_heads_win_rate=0.611,
        shared_heads_win_rate=0.389,
    ) == pytest.approx(0.222)


def test_outcome_vector_rejects_invalid_probability_mass():
    with pytest.raises(ValueError, match="sum to 1"):
        OutcomeVector(role0=0.5, role1=0.5, draw=0.5)


def test_outcome_vector_rejects_negative_probability():
    with pytest.raises(ValueError, match="non-negative"):
        OutcomeVector(role0=-0.1, role1=1.1, draw=0.0)


def test_summarize_metrics_groups_by_variant(tmp_path: Path):
    metrics = tmp_path / "metrics.jsonl"
    rows = [
        {
            "variant": "shared_heads",
            "eval_model_win_rate": 0.25,
            "eval_avg_plies": 8,
            "eval_random_win_rate": 0.5,
            "eval_draw_rate": 0.25,
            "policy_loss": 2.0,
            "value_loss": 1.0,
            "train_total_loss": 3.0,
        },
        {
            "variant": "shared_heads",
            "eval_model_win_rate": 0.75,
            "eval_avg_plies": 10,
            "eval_random_win_rate": 0.25,
            "eval_draw_rate": 0.0,
            "policy_loss": 4.0,
            "value_loss": 2.0,
            "train_total_loss": 6.0,
            "eval_model_role_win_rates": {"0": 1.0, "1": 0.5},
            "eval_termination_reasons": {"max_plies": 2},
        },
        {
            "variant": "role_heads",
            "eval_model_win_rate": 1.0,
            "eval_avg_plies": 7,
        },
    ]
    metrics.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    summary = summarize_metrics(metrics)
    assert list(summary) == ["role_heads", "shared_heads"]
    assert summary["shared_heads"]["mean_model_win_rate"] == 0.5
    assert summary["shared_heads"]["mean_avg_plies"] == 9.0
    assert summary["shared_heads"]["rows"] == 2
    assert summary["shared_heads"]["last_model_win_rate"] == 0.75
    assert summary["shared_heads"]["mean_random_win_rate"] == 0.375
    assert summary["shared_heads"]["mean_draw_rate"] == 0.125
    assert summary["shared_heads"]["mean_policy_loss"] == 3.0
    assert summary["shared_heads"]["mean_value_loss"] == 1.5
    assert summary["shared_heads"]["mean_train_total_loss"] == 4.5
    assert summary["shared_heads"]["last_eval_model_role_win_rates"] == {
        "0": 1.0,
        "1": 0.5,
    }
    assert summary["shared_heads"]["last_eval_termination_reasons"] == {
        "max_plies": 2,
    }
    assert summary["role_heads"]["rows"] == 1
    assert summary["role_heads"]["mean_model_win_rate"] == 1.0


def test_summarize_metrics_preserves_generated_metadata_when_present(tmp_path: Path):
    metrics = tmp_path / "metrics.jsonl"
    rows = [
        {
            "variant": "shared_heads",
            "eval_model_win_rate": 0.25,
            "eval_avg_plies": 8,
            "generated_family": "escape_capture",
            "generated_name": "escape_capture_v1",
            "generated_seed": 17,
            "generated_spec_path": "research/asymbench/specs/escape_capture.json",
        },
        {
            "variant": "role_heads",
            "eval_model_win_rate": 0.75,
            "eval_avg_plies": 10,
            "generated_family": "escape_capture",
            "generated_name": "escape_capture_v1",
            "generated_seed": 17,
            "generated_spec_path": "research/asymbench/specs/escape_capture.json",
        },
    ]
    metrics.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    summary = summarize_metrics(metrics)
    assert summary["shared_heads"]["generated_family"] == "escape_capture"
    assert summary["shared_heads"]["generated_name"] == "escape_capture_v1"
    assert summary["shared_heads"]["generated_seed"] == 17
    assert (
        summary["shared_heads"]["generated_spec_path"]
        == "research/asymbench/specs/escape_capture.json"
    )
    assert summary["role_heads"]["generated_family"] == "escape_capture"
    assert summary["role_heads"]["generated_name"] == "escape_capture_v1"
    assert summary["role_heads"]["generated_seed"] == 17
    assert (
        summary["role_heads"]["generated_spec_path"]
        == "research/asymbench/specs/escape_capture.json"
    )


def test_summarize_metrics_omits_partial_generated_metadata(tmp_path: Path):
    metrics = tmp_path / "metrics.jsonl"
    rows = [
        {
            "variant": "shared_heads",
            "eval_model_win_rate": 0.25,
            "eval_avg_plies": 8,
            "generated_family": "escape_capture",
        },
        {
            "variant": "shared_heads",
            "eval_model_win_rate": 0.75,
            "eval_avg_plies": 10,
        },
    ]
    metrics.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    summary = summarize_metrics(metrics)
    assert "generated_family" not in summary["shared_heads"]


def test_summarize_metrics_rejects_non_int_generated_seed(tmp_path: Path):
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        json.dumps(
            {
                "variant": "shared_heads",
                "eval_model_win_rate": 0.5,
                "eval_avg_plies": 8,
                "generated_seed": 17.0,
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="generated_seed"):
        summarize_metrics(metrics)


def test_summarize_metrics_rejects_bool_generated_seed(tmp_path: Path):
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        json.dumps(
            {
                "variant": "shared_heads",
                "eval_model_win_rate": 0.5,
                "eval_avg_plies": 8,
                "generated_seed": True,
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="generated_seed"):
        summarize_metrics(metrics)


def test_summarize_metrics_rejects_empty_file(tmp_path: Path):
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text("")

    with pytest.raises(ValueError, match="empty"):
        summarize_metrics(metrics)


def test_summarize_metrics_rejects_invalid_json(tmp_path: Path):
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text('{"variant": "shared_heads"\n')

    with pytest.raises(ValueError, match="invalid JSON on line 1"):
        summarize_metrics(metrics)


def test_summarize_metrics_rejects_missing_required_fields(tmp_path: Path):
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(json.dumps({"variant": "shared_heads"}) + "\n")

    with pytest.raises(ValueError, match="missing required fields"):
        summarize_metrics(metrics)


def test_summarize_metrics_rejects_non_object_row(tmp_path: Path):
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text("[1, 2, 3]\n")

    with pytest.raises(ValueError, match="must be a JSON object"):
        summarize_metrics(metrics)


def test_summarize_metrics_rejects_non_numeric_required_field(tmp_path: Path):
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        json.dumps(
            {
                "variant": "shared_heads",
                "eval_model_win_rate": "0.5",
                "eval_avg_plies": 8,
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="eval_model_win_rate"):
        summarize_metrics(metrics)


def test_summarize_metrics_rejects_bool_required_field(tmp_path: Path):
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        json.dumps(
            {
                "variant": "shared_heads",
                "eval_model_win_rate": True,
                "eval_avg_plies": 8,
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="eval_model_win_rate"):
        summarize_metrics(metrics)


def test_summarize_metrics_rejects_non_numeric_optional_field(tmp_path: Path):
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        json.dumps(
            {
                "variant": "shared_heads",
                "eval_model_win_rate": 0.5,
                "eval_avg_plies": 8,
                "policy_loss": "bad",
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="policy_loss"):
        summarize_metrics(metrics)


def test_summarize_metrics_rejects_bool_optional_field(tmp_path: Path):
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        json.dumps(
            {
                "variant": "shared_heads",
                "eval_model_win_rate": 0.5,
                "eval_avg_plies": 8,
                "policy_loss": False,
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="policy_loss"):
        summarize_metrics(metrics)


def test_cli_success_prints_json_summary(tmp_path: Path, capsys):
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        json.dumps(
            {
                "variant": "shared_heads",
                "eval_model_win_rate": 0.5,
                "eval_avg_plies": 8,
            }
        )
        + "\n"
    )

    assert main([str(metrics)]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["shared_heads"]["rows"] == 1
    assert captured.err == ""


def test_cli_error_prints_concise_message(tmp_path: Path, capsys):
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text("")

    assert main([str(metrics)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error: metrics file is empty" in captured.err


def test_cli_os_error_prints_concise_message(tmp_path: Path, capsys):
    missing_metrics = tmp_path / "missing.jsonl"

    assert main([str(missing_metrics)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error:" in captured.err
    assert "No such file or directory" in captured.err
    assert missing_metrics.name in captured.err
