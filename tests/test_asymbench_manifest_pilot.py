import json
from pathlib import Path

import pytest

from research.asymbench.experiments.prepare_manifest_pilot import (
    build_manifest_pilot,
    main as pilot_main,
)


def test_build_manifest_pilot_writes_specs_configs_and_selection(tmp_path: Path):
    escape_manifest = tmp_path / "escape.json"
    connection_manifest = tmp_path / "connection.json"
    escape_manifest.write_text(
        json.dumps(
            _manifest(
                "escape_capture",
                [
                    _entry(
                        name="escape_clean",
                        seed=1,
                        stratum="clean_control",
                        labels=["strict_clean"],
                        role_bias=0.0,
                        seat_bias=0.0,
                        max_ply_rate=0.0,
                    ),
                    _entry(
                        name="escape_seat",
                        seed=2,
                        stratum="seat_confound",
                        labels=["seat_sensitive"],
                        role_bias=0.1,
                        seat_bias=0.9,
                        max_ply_rate=0.2,
                    ),
                ],
            )
        )
    )
    connection_manifest.write_text(
        json.dumps(
            _manifest(
                "connection_disruption",
                [
                    _entry(
                        name="connection_collapse",
                        seed=3,
                        stratum="role_collapse",
                        labels=["high_sim_collapsed"],
                        role_bias=1.0,
                        seat_bias=0.0,
                        max_ply_rate=0.0,
                    ),
                ],
            )
        )
    )
    template = {
        "game": "breaker_builder",
        "device": "cuda",
        "seeds": [11],
        "model_variants": ["shared_heads", "role_heads"],
        "iterations": 1,
        "selfplay_games_per_iteration": 1,
        "train_steps_per_iteration": 1,
        "batch_size": 2,
        "replay_capacity": 64,
        "mcts_simulations": 1,
        "eval_games": 1,
        "eval_simulations": 1,
        "learning_rate": 0.001,
        "output_root": "research_runs/asymbench",
    }

    pilot = build_manifest_pilot(
        manifest_paths=[escape_manifest, connection_manifest],
        output_root=tmp_path / "pilot",
        template=template,
    )

    assert pilot["schema_version"] == 1
    assert {entry["bucket"] for entry in pilot["entries"]} == {
        "clean",
        "seat_sensitive",
        "collapse",
    }
    for entry in pilot["entries"]:
        spec_path = Path(entry["spec_path"])
        config_path = Path(entry["config_path"])
        assert spec_path.is_file()
        assert config_path.is_file()
        config = json.loads(config_path.read_text())
        assert "game" not in config
        assert config["game_source"]["type"] == "generated_spec"
        resolved_spec_path = (config_path.parent / config["game_source"]["path"]).resolve()
        assert resolved_spec_path == spec_path.resolve()
        assert config["output_root"] == str((tmp_path / "pilot" / "runs").resolve())

    clean = next(entry for entry in pilot["entries"] if entry["bucket"] == "clean")
    assert clean["family"] == "escape_capture"
    assert clean["verification_labels"] == ["strict_clean"]
    assert clean["strata"] == ["clean_control"]

    selection_path = tmp_path / "pilot" / "pilot_manifest.json"
    assert json.loads(selection_path.read_text()) == pilot


def test_manifest_pilot_rejects_entries_without_embedded_specs(tmp_path: Path):
    manifest = _manifest(
        "escape_capture",
        [
            _entry(
                name="missing_spec",
                seed=1,
                stratum="clean_control",
                labels=["strict_clean"],
                role_bias=0.0,
                seat_bias=0.0,
                max_ply_rate=0.0,
                include_spec=False,
            )
        ],
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="embedded spec"):
        build_manifest_pilot(
            manifest_paths=[manifest_path],
            output_root=tmp_path / "pilot",
        )


def test_build_manifest_pilot_can_target_buckets(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            _manifest(
                "escape_capture",
                [
                    _entry(
                        name="escape_clean",
                        seed=1,
                        stratum="clean_control",
                        labels=["strict_clean"],
                        role_bias=0.0,
                        seat_bias=0.0,
                        max_ply_rate=0.0,
                    ),
                    _entry(
                        name="escape_seat",
                        seed=2,
                        stratum="seat_confound",
                        labels=["seat_sensitive"],
                        role_bias=0.1,
                        seat_bias=0.9,
                        max_ply_rate=0.2,
                    ),
                ],
            )
        )
    )

    pilot = build_manifest_pilot(
        manifest_paths=[manifest_path],
        output_root=tmp_path / "pilot",
        buckets=("clean",),
    )

    assert pilot["buckets"] == ["clean"]
    assert [entry["bucket"] for entry in pilot["entries"]] == ["clean"]
    assert pilot["entries"][0]["name"] == "escape_clean"


def test_manifest_pilot_cli_writes_output(tmp_path: Path, capsys):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            _manifest(
                "escape_capture",
                [
                    _entry(
                        name="escape_clean",
                        seed=1,
                        stratum="clean_control",
                        labels=["strict_clean"],
                        role_bias=0.0,
                        seat_bias=0.0,
                        max_ply_rate=0.0,
                    )
                ],
            )
        )
    )

    assert (
        pilot_main(
            [
                "--manifest",
                str(manifest_path),
                "--output-root",
                str(tmp_path / "pilot"),
                "--per-bucket-per-family",
                "1",
                "--bucket",
                "clean",
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert "pilot_manifest=" in captured.out
    assert captured.err == ""
    pilot = json.loads((tmp_path / "pilot" / "pilot_manifest.json").read_text())
    assert pilot["buckets"] == ["clean"]


def _manifest(family: str, entries: list[dict[str, object]]) -> dict[str, object]:
    strata = {
        "clean_control": [],
        "hidden_collapse": [],
        "horizon_stress": [],
        "role_collapse": [],
        "role_inversion": [],
        "seat_confound": [],
    }
    for entry in entries:
        strata[str(entry["primary_stratum"])].append(entry)
    return {
        "schema_version": 2,
        "embed_specs": True,
        "strata": strata,
        "verification_reports": [],
    }


def _entry(
    *,
    name: str,
    seed: int,
    stratum: str,
    labels: list[str],
    role_bias: float,
    seat_bias: float,
    max_ply_rate: float,
    include_spec: bool = True,
) -> dict[str, object]:
    family = "connection_disruption" if name.startswith("connection") else "escape_capture"
    entry: dict[str, object] = {
        "family": family,
        "name": name,
        "seed": seed,
        "rank": 1,
        "labels": [stratum],
        "primary_stratum": stratum,
        "verification_labels": labels,
        "mcts_role_bias": role_bias,
        "mcts_seat_bias": seat_bias,
        "mcts_max_ply_rate": max_ply_rate,
        "role_inversion_score": 0.0,
        "verification_metrics": {
            "mcts_role_bias": role_bias,
            "mcts_seat_bias": seat_bias,
            "mcts_max_ply_rate": max_ply_rate,
            "role_inversion_score": 0.0,
        },
    }
    if include_spec:
        entry["spec"] = {
            "family": family,
            "name": name,
            "seed": seed,
            "board": {"rows": 5, "cols": 5},
            "roles": (
                ["builder", "breaker"]
                if family == "connection_disruption"
                else ["attacker", "defender"]
            ),
            "setup": {},
            "actions": {},
            "terminal_rules": {},
            "max_plies": 40,
        }
    return entry
