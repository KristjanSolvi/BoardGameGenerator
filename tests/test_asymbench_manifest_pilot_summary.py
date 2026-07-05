import json
from pathlib import Path

from research.asymbench.analysis.summarize_manifest_pilot import (
    main as summarize_pilot_main,
    summarize_manifest_pilot,
)


def test_summarize_manifest_pilot_computes_architecture_delta(tmp_path: Path):
    pilot = {
        "schema_version": 1,
        "output_root": str(tmp_path / "pilot"),
        "entries": [
            _pilot_entry(
                bucket="clean",
                family="escape_capture",
                name="escape_clean",
                seed=1,
            ),
            _pilot_entry(
                bucket="collapse",
                family="connection_disruption",
                name="connection_collapse",
                seed=2,
            ),
        ],
    }
    pilot_path = tmp_path / "pilot_manifest.json"
    pilot_path.write_text(json.dumps(pilot))
    runs_root = tmp_path / "pilot" / "runs"
    _write_role_summary(
        runs_root / "old_escape",
        family="escape_capture",
        name="escape_clean",
        seed=1,
        shared_win=0.25,
        role_win=0.50,
    )
    latest_escape = runs_root / "latest_escape"
    _write_role_summary(
        latest_escape,
        family="escape_capture",
        name="escape_clean",
        seed=1,
        shared_win=0.25,
        role_win=0.75,
    )
    _write_role_summary(
        runs_root / "connection",
        family="connection_disruption",
        name="connection_collapse",
        seed=2,
        shared_win=1.0,
        role_win=0.25,
    )

    summary = summarize_manifest_pilot(pilot_path)

    assert summary["completed_entries"] == 2
    by_key = {
        (entry["family"], entry["name"], entry["seed"]): entry
        for entry in summary["entries"]
    }
    escape = by_key[("escape_capture", "escape_clean", 1)]
    assert escape["architecture_delta"] == 0.5
    assert escape["run_dir"] == str(latest_escape)
    connection = by_key[("connection_disruption", "connection_collapse", 2)]
    assert connection["architecture_delta"] == -0.75

    assert summary["by_bucket"]["clean"]["mean_architecture_delta"] == 0.5
    assert summary["by_bucket"]["collapse"]["mean_architecture_delta"] == -0.75
    assert summary["by_family"]["escape_capture"]["mean_role_heads_win_rate"] == 0.75


def test_summarize_manifest_pilot_marks_missing_runs(tmp_path: Path):
    pilot = {
        "schema_version": 1,
        "output_root": str(tmp_path / "pilot"),
        "entries": [
            _pilot_entry(
                bucket="clean",
                family="escape_capture",
                name="missing",
                seed=1,
            )
        ],
    }
    pilot_path = tmp_path / "pilot_manifest.json"
    pilot_path.write_text(json.dumps(pilot))

    summary = summarize_manifest_pilot(pilot_path)

    assert summary["completed_entries"] == 0
    assert summary["missing_entries"] == 1
    assert summary["entries"][0]["completed"] is False
    assert summary["by_bucket"]["clean"]["missing"] == 1


def test_summarize_manifest_pilot_resolves_relative_output_root_from_cwd(
    tmp_path: Path,
    monkeypatch,
):
    workspace = tmp_path / "workspace"
    manifest_dir = workspace / "nested" / "pilot"
    manifest_dir.mkdir(parents=True)
    pilot = {
        "schema_version": 1,
        "output_root": "research_runs/asymbench/pilot",
        "entries": [
            _pilot_entry(
                bucket="clean",
                family="escape_capture",
                name="escape_clean",
                seed=1,
            )
        ],
    }
    pilot_path = manifest_dir / "pilot_manifest.json"
    pilot_path.write_text(json.dumps(pilot))
    _write_role_summary(
        workspace / "research_runs" / "asymbench" / "pilot" / "runs" / "escape",
        family="escape_capture",
        name="escape_clean",
        seed=1,
        shared_win=0.25,
        role_win=0.75,
    )
    monkeypatch.chdir(workspace)

    summary = summarize_manifest_pilot(pilot_path)

    assert summary["completed_entries"] == 1
    assert summary["entries"][0]["architecture_delta"] == 0.5


def test_summarize_manifest_pilot_cli_writes_json(tmp_path: Path, capsys):
    pilot = {
        "schema_version": 1,
        "output_root": str(tmp_path / "pilot"),
        "entries": [
            _pilot_entry(
                bucket="clean",
                family="escape_capture",
                name="escape_clean",
                seed=1,
            )
        ],
    }
    pilot_path = tmp_path / "pilot_manifest.json"
    output = tmp_path / "summary.json"
    pilot_path.write_text(json.dumps(pilot))
    _write_role_summary(
        tmp_path / "pilot" / "runs" / "escape",
        family="escape_capture",
        name="escape_clean",
        seed=1,
        shared_win=0.25,
        role_win=0.75,
    )

    assert summarize_pilot_main([str(pilot_path), "--output", str(output)]) == 0
    captured = capsys.readouterr()
    assert "completed=1" in captured.out
    assert captured.err == ""
    assert json.loads(output.read_text())["entries"][0]["architecture_delta"] == 0.5


def _pilot_entry(
    *,
    bucket: str,
    family: str,
    name: str,
    seed: int,
) -> dict[str, object]:
    return {
        "bucket": bucket,
        "family": family,
        "name": name,
        "seed": seed,
        "strata": [bucket],
        "verification_labels": [],
        "mcts_role_bias": 0.0,
        "mcts_seat_bias": 0.0,
        "mcts_max_ply_rate": 0.0,
    }


def _write_role_summary(
    directory: Path,
    *,
    family: str,
    name: str,
    seed: int,
    shared_win: float,
    role_win: float,
) -> None:
    directory.mkdir(parents=True)
    (directory / "role_summary.json").write_text(
        json.dumps(
            {
                "generated_family": family,
                "generated_name": name,
                "generated_seed": seed,
                "device_used": "cuda",
                "run_dir": str(directory),
                "by_variant": {
                    "shared_heads": {
                        "final_eval_model_win_rate_mean": shared_win,
                        "final_eval_random_win_rate_mean": 1.0 - shared_win,
                        "final_eval_draw_rate_mean": 0.0,
                        "final_eval_avg_plies_mean": 10.0,
                    },
                    "role_heads": {
                        "final_eval_model_win_rate_mean": role_win,
                        "final_eval_random_win_rate_mean": 1.0 - role_win,
                        "final_eval_draw_rate_mean": 0.0,
                        "final_eval_avg_plies_mean": 8.0,
                    },
                },
            }
        )
    )
