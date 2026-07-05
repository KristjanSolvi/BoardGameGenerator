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
from research.asymbench.analysis.summarize import main as summarize_main
from research.asymbench.analysis.summarize import summarize_metrics
from research.asymbench.analysis.strata import (
    build_selection_manifest,
    classify_generated_validation,
    load_classified_validation_entries,
    load_classified_validations,
    main as strata_main,
    rank_strata,
)


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


def test_classify_generated_validation_marks_clean_control():
    classified = classify_generated_validation(
        spec={
            "family": "connection_disruption",
            "name": "wall_seed_8003",
            "seed": 8003,
        },
        report={
            "valid": True,
            "random_role_win_rates": {"0": 0.688, "1": 0.312},
            "mcts_role_win_rates": {"0": 0.5, "1": 0.5},
            "mcts_first_player_win_rate": 0.5,
            "terminal_reasons": {"builder_connection": 11, "max_plies": 5},
            "mcts_terminal_reasons": {"builder_connection": 6, "max_plies": 6},
        },
    )

    assert "clean_control" in classified.labels
    assert classified.mcts_role_bias == pytest.approx(0.0)
    assert classified.mcts_seat_bias == pytest.approx(0.0)
    assert classified.mcts_max_ply_rate == pytest.approx(0.5)


def test_classify_generated_validation_marks_diagnostic_stress_labels():
    classified = classify_generated_validation(
        spec={
            "family": "connection_disruption",
            "name": "collapse_seed",
            "seed": 1,
        },
        report={
            "valid": True,
            "random_role_win_rates": {"0": 0.25, "1": 0.75},
            "mcts_role_win_rates": {"0": 1.0, "1": 0.0},
            "mcts_first_player_win_rate": 0.5,
            "terminal_reasons": {"max_plies": 12, "builder_connection": 4},
            "mcts_terminal_reasons": {"builder_connection": 12},
        },
    )

    assert "hidden_collapse" in classified.labels
    assert "role_inversion" in classified.labels
    assert "role_collapse" in classified.labels
    assert "horizon_stress" in classified.labels


def test_classify_generated_validation_marks_seat_confounds():
    classified = classify_generated_validation(
        spec={"family": "connection_disruption", "name": "seat_seed", "seed": 2},
        report={
            "valid": True,
            "random_role_win_rates": {"0": 0.5, "1": 0.5},
            "mcts_role_win_rates": {"0": 0.5, "1": 0.5},
            "mcts_first_player_win_rate": 0.875,
            "terminal_reasons": {"builder_connection": 16},
            "mcts_terminal_reasons": {"builder_connection": 12},
        },
    )

    assert "seat_confound" in classified.labels
    assert classified.mcts_seat_bias == pytest.approx(0.75)


def test_load_and_rank_classified_validations(tmp_path: Path):
    for dirname, spec, report in [
        (
            "clean",
            {"family": "connection_disruption", "name": "clean", "seed": 1},
            {
                "valid": True,
                "random_role_win_rates": {"0": 0.5, "1": 0.5},
                "mcts_role_win_rates": {"0": 0.5, "1": 0.5},
                "mcts_first_player_win_rate": 0.5,
                "terminal_reasons": {"builder_connection": 8, "max_plies": 8},
                "mcts_terminal_reasons": {"builder_connection": 6, "max_plies": 6},
            },
        ),
        (
            "collapse",
            {"family": "connection_disruption", "name": "collapse", "seed": 2},
            {
                "valid": True,
                "random_role_win_rates": {"0": 0.5, "1": 0.5},
                "mcts_role_win_rates": {"0": 1.0, "1": 0.0},
                "mcts_first_player_win_rate": 0.5,
                "terminal_reasons": {"builder_connection": 16},
                "mcts_terminal_reasons": {"builder_connection": 12},
            },
        ),
    ]:
        run_dir = tmp_path / dirname
        run_dir.mkdir()
        (run_dir / "spec.json").write_text(json.dumps(spec))
        (run_dir / "validation.json").write_text(json.dumps(report))

    records = load_classified_validations(tmp_path)
    ranked = rank_strata(records, limit_per_stratum=1)

    assert [record.name for record in records] == ["clean", "collapse"]
    assert ranked["clean_control"][0].name == "clean"
    assert ranked["hidden_collapse"][0].name == "collapse"


def test_build_selection_manifest_exports_ranked_paths_and_thresholds(tmp_path: Path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    _write_generated_validation(
        first_root / "clean",
        spec={"family": "connection_disruption", "name": "clean", "seed": 1},
        report={
            "valid": True,
            "random_role_win_rates": {"0": 0.5, "1": 0.5},
            "mcts_role_win_rates": {"0": 0.5, "1": 0.5},
            "mcts_first_player_win_rate": 0.5,
            "terminal_reasons": {"builder_connection": 8, "max_plies": 8},
            "mcts_terminal_reasons": {"builder_connection": 6, "max_plies": 6},
        },
    )
    _write_generated_validation(
        second_root / "collapse",
        spec={"family": "connection_disruption", "name": "collapse", "seed": 2},
        report={
            "valid": True,
            "random_role_win_rates": {"0": 0.5, "1": 0.5},
            "mcts_role_win_rates": {"0": 1.0, "1": 0.0},
            "mcts_first_player_win_rate": 0.5,
            "terminal_reasons": {"builder_connection": 16},
            "mcts_terminal_reasons": {"builder_connection": 12},
        },
    )

    manifest = build_selection_manifest(
        input_roots=[second_root, first_root],
        limit_per_stratum=1,
    )

    assert manifest["schema_version"] == 2
    assert manifest["limit_per_stratum"] == 1
    assert manifest["input_roots"] == [str(second_root), str(first_root)]
    assert manifest["thresholds"]["main_band_role_bias"] == 0.6
    assert manifest["verification_reports"] == []
    assert manifest["strata"]["clean_control"][0]["name"] == "clean"
    assert manifest["strata"]["clean_control"][0]["rank"] == 1
    assert manifest["strata"]["clean_control"][0]["spec_path"].endswith(
        str(Path("clean") / "spec.json")
    )
    assert manifest["strata"]["hidden_collapse"][0]["name"] == "collapse"


def test_build_selection_manifest_adds_second_stage_verification_labels(
    tmp_path: Path,
):
    root = tmp_path / "root"
    for dirname, seed, first_pass_report in [
        (
            "strict",
            1,
            {
                "valid": True,
                "random_role_win_rates": {"0": 0.5, "1": 0.5},
                "mcts_role_win_rates": {"0": 0.5, "1": 0.5},
                "mcts_first_player_win_rate": 0.5,
                "terminal_reasons": {"builder_connection": 8, "max_plies": 8},
                "mcts_terminal_reasons": {"builder_connection": 6, "max_plies": 6},
            },
        ),
        (
            "near",
            2,
            {
                "valid": True,
                "random_role_win_rates": {"0": 0.5, "1": 0.5},
                "mcts_role_win_rates": {"0": 0.667, "1": 0.333},
                "mcts_first_player_win_rate": 0.5,
                "terminal_reasons": {"builder_connection": 8, "max_plies": 8},
                "mcts_terminal_reasons": {"builder_connection": 8, "max_plies": 4},
            },
        ),
        (
            "seat",
            3,
            {
                "valid": True,
                "random_role_win_rates": {"0": 0.5, "1": 0.5},
                "mcts_role_win_rates": {"0": 0.5, "1": 0.5},
                "mcts_first_player_win_rate": 0.5,
                "terminal_reasons": {"builder_connection": 8, "max_plies": 8},
                "mcts_terminal_reasons": {"builder_connection": 8, "max_plies": 4},
            },
        ),
        (
            "collapsed",
            4,
            {
                "valid": True,
                "random_role_win_rates": {"0": 0.5, "1": 0.5},
                "mcts_role_win_rates": {"0": 0.5, "1": 0.5},
                "mcts_first_player_win_rate": 0.5,
                "terminal_reasons": {"builder_connection": 8, "max_plies": 8},
                "mcts_terminal_reasons": {"builder_connection": 8, "max_plies": 4},
            },
        ),
    ]:
        _write_generated_validation(
            root / dirname,
            spec={"family": "connection_disruption", "name": dirname, "seed": seed},
            report=first_pass_report,
        )
    verification = tmp_path / "verification.json"
    verification.write_text(
        json.dumps(
            [
                {
                    "name": "strict",
                    "seed": 1,
                    "mcts_role_win_rates": {"0": 0.5, "1": 0.5},
                    "mcts_first_player_win_rate": 0.5,
                    "mcts_terminal_reasons": {"builder_connection": 6, "max_plies": 6},
                },
                {
                    "name": "near",
                    "seed": 2,
                    "mcts_role_win_rates": {"0": 0.417, "1": 0.583},
                    "mcts_first_player_win_rate": 0.583,
                    "mcts_terminal_reasons": {"builder_connection": 5, "max_plies": 7},
                },
                {
                    "name": "seat",
                    "seed": 3,
                    "mcts_role_win_rates": {"0": 0.5, "1": 0.5},
                    "mcts_first_player_win_rate": 0.333,
                    "mcts_terminal_reasons": {"builder_connection": 6, "max_plies": 6},
                },
                {
                    "name": "collapsed",
                    "seed": 4,
                    "mcts_role_win_rates": {"0": 0.917, "1": 0.083},
                    "mcts_first_player_win_rate": 0.5,
                    "mcts_terminal_reasons": {"builder_connection": 11, "max_plies": 1},
                },
            ]
        )
    )

    manifest = build_selection_manifest(
        input_roots=[root],
        limit_per_stratum=4,
        verification_reports=[verification],
    )
    by_name = {
        entry["name"]: entry
        for entry in manifest["strata"]["clean_control"]
    }

    assert by_name["strict"]["verification_labels"] == ["strict_clean"]
    assert by_name["near"]["verification_labels"] == ["near_clean"]
    assert by_name["seat"]["verification_labels"] == ["seat_sensitive"]
    assert by_name["collapsed"]["verification_labels"] == ["high_sim_collapsed"]
    assert by_name["strict"]["verification_metrics"]["mcts_role_bias"] == 0.0
    assert by_name["strict"]["verification_metrics"]["source_path"] == str(verification)


def test_build_selection_manifest_marks_verified_inversion_and_hidden_collapse(
    tmp_path: Path,
):
    root = tmp_path / "root"
    _write_generated_validation(
        root / "inversion",
        spec={"family": "connection_disruption", "name": "inversion", "seed": 1},
        report={
            "valid": True,
            "random_role_win_rates": {"0": 0.25, "1": 0.75},
            "mcts_role_win_rates": {"0": 1.0, "1": 0.0},
            "mcts_first_player_win_rate": 0.5,
            "terminal_reasons": {"builder_connection": 4, "max_plies": 12},
            "mcts_terminal_reasons": {"builder_connection": 12},
        },
    )
    verification = tmp_path / "verification.json"
    verification.write_text(
        json.dumps(
            [
                {
                    "name": "inversion",
                    "seed": 1,
                    "random_role_win_rates": {"0": 0.25, "1": 0.75},
                    "mcts_role_win_rates": {"0": 0.917, "1": 0.083},
                    "mcts_first_player_win_rate": 0.5,
                    "mcts_terminal_reasons": {"builder_connection": 11, "max_plies": 1},
                }
            ]
        )
    )

    manifest = build_selection_manifest(
        input_roots=[root],
        limit_per_stratum=1,
        verification_reports=[verification],
    )
    entry = manifest["strata"]["role_inversion"][0]

    assert "verified_hidden_collapse" in entry["verification_labels"]
    assert "verified_role_inversion" in entry["verification_labels"]


def test_build_selection_manifest_supports_mcts64_verification_format(
    tmp_path: Path,
):
    root = tmp_path / "root"
    _write_generated_validation(
        root / "clean",
        spec={"family": "connection_disruption", "name": "clean", "seed": 1},
        report={
            "valid": True,
            "random_role_win_rates": {"0": 0.5, "1": 0.5},
            "mcts_role_win_rates": {"0": 0.5, "1": 0.5},
            "mcts_first_player_win_rate": 0.5,
            "terminal_reasons": {"builder_connection": 8, "max_plies": 8},
            "mcts_terminal_reasons": {"builder_connection": 6, "max_plies": 6},
        },
    )
    verification = tmp_path / "mcts64.json"
    verification.write_text(
        json.dumps(
            [
                {
                    "name": "clean",
                    "seed": 1,
                    "mcts64_role0": 0.5,
                    "mcts64_role1": 0.5,
                    "mcts64_first_player": 0.5,
                    "mcts64_terminal_reasons": {"builder_connection": 6, "max_plies": 6},
                }
            ]
        )
    )

    manifest = build_selection_manifest(
        input_roots=[root],
        limit_per_stratum=1,
        verification_reports=[verification],
    )

    assert manifest["verification_reports"] == [str(verification)]
    assert manifest["strata"]["clean_control"][0]["verification_labels"] == [
        "strict_clean"
    ]


def test_build_selection_manifest_can_embed_specs(tmp_path: Path):
    root = tmp_path / "root"
    spec = {
        "family": "connection_disruption",
        "name": "clean",
        "seed": 1,
        "board": {"rows": 5, "cols": 5},
    }
    _write_generated_validation(
        root / "clean",
        spec=spec,
        report={
            "valid": True,
            "random_role_win_rates": {"0": 0.5, "1": 0.5},
            "mcts_role_win_rates": {"0": 0.5, "1": 0.5},
            "mcts_first_player_win_rate": 0.5,
            "terminal_reasons": {"builder_connection": 8, "max_plies": 8},
            "mcts_terminal_reasons": {"builder_connection": 6, "max_plies": 6},
        },
    )

    manifest = build_selection_manifest(
        input_roots=[root],
        limit_per_stratum=1,
        embed_specs=True,
    )

    assert manifest["embed_specs"] is True
    assert manifest["strata"]["clean_control"][0]["spec"] == spec


def test_load_classified_validation_entries_requires_mcts_fields(tmp_path: Path):
    root = tmp_path / "root"
    _write_generated_validation(
        root / "random_only",
        spec={"family": "connection_disruption", "name": "random_only", "seed": 1},
        report={
            "valid": True,
            "random_role_win_rates": {"0": 0.5, "1": 0.5},
            "terminal_reasons": {"builder_connection": 8},
        },
    )

    assert load_classified_validation_entries([root]) == []
    assert (
        load_classified_validation_entries([root], require_mcts=False)[0].record.name
        == "random_only"
    )


def test_strata_cli_writes_selection_json_and_role_configs(
    tmp_path: Path,
    capsys,
):
    root = tmp_path / "root"
    _write_generated_validation(
        root / "clean",
        spec={"family": "connection_disruption", "name": "clean", "seed": 1},
        report={
            "valid": True,
            "random_role_win_rates": {"0": 0.5, "1": 0.5},
            "mcts_role_win_rates": {"0": 0.5, "1": 0.5},
            "mcts_first_player_win_rate": 0.5,
            "terminal_reasons": {"builder_connection": 8, "max_plies": 8},
            "mcts_terminal_reasons": {"builder_connection": 6, "max_plies": 6},
        },
    )
    template = tmp_path / "template.json"
    template.write_text(
        json.dumps(
            {
                "game": "breaker_builder",
                "device": "cuda",
                "seeds": [1],
                "model_variants": ["shared_heads", "role_heads"],
                "iterations": 1,
                "selfplay_games_per_iteration": 1,
                "train_steps_per_iteration": 1,
                "batch_size": 1,
                "replay_capacity": 8,
                "mcts_simulations": 1,
                "eval_games": 1,
                "eval_simulations": 1,
                "learning_rate": 0.001,
                "output_root": "research_runs/asymbench",
            }
        )
    )
    output = tmp_path / "selection.json"
    config_root = tmp_path / "configs"

    assert (
        strata_main(
            [
                "--input",
                str(root),
                "--limit-per-stratum",
                "1",
                "--output",
                str(output),
                "--embed-specs",
                "--role-config-template",
                str(template),
                "--role-config-output",
                str(config_root),
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    manifest = json.loads(output.read_text())
    config_path = Path(manifest["strata"]["clean_control"][0]["role_config_path"])
    config = json.loads(config_path.read_text())
    assert manifest["strata"]["clean_control"][0]["spec"]["name"] == "clean"
    assert "game" not in config
    assert config["game_source"] == {
        "type": "generated_spec",
        "path": str(root / "clean" / "spec.json"),
    }
    assert config_path.parent == config_root / "clean_control"


def test_strata_cli_missing_input_prints_concise_error(tmp_path: Path, capsys):
    missing = tmp_path / "missing"

    assert strata_main(["--input", str(missing)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error:" in captured.err
    assert missing.name in captured.err


def _write_generated_validation(
    directory: Path,
    *,
    spec: dict[str, object],
    report: dict[str, object],
) -> None:
    directory.mkdir(parents=True)
    (directory / "spec.json").write_text(json.dumps(spec))
    (directory / "validation.json").write_text(json.dumps(report))


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

    assert summarize_main([str(metrics)]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["shared_heads"]["rows"] == 1
    assert captured.err == ""


def test_cli_error_prints_concise_message(tmp_path: Path, capsys):
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text("")

    assert summarize_main([str(metrics)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error: metrics file is empty" in captured.err


def test_cli_os_error_prints_concise_message(tmp_path: Path, capsys):
    missing_metrics = tmp_path / "missing.jsonl"

    assert summarize_main([str(missing_metrics)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error:" in captured.err
    assert "No such file or directory" in captured.err
    assert missing_metrics.name in captured.err
