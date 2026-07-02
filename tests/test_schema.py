import json
import re
from pathlib import Path

import pytest

from gamegen.extraction import extract_json
from gamegen.schema import SpecError, validate_spec

ROOT = Path(__file__).resolve().parent.parent


def load_example_spec() -> dict:
    """The filled-in example inside the designer prompt must itself pass
    strict validation — otherwise we are teaching the model a broken
    format."""
    text = (ROOT / "prompts" / "designer.md").read_text()
    return extract_json(text)


def test_designer_prompt_example_is_valid():
    validate_spec(load_example_spec())


def test_missing_field_rejected():
    spec = load_example_spec()
    del spec["win_conditions"]
    with pytest.raises(SpecError, match="win_conditions"):
        validate_spec(spec)


def test_unknown_top_level_key_rejected():
    spec = load_example_spec()
    spec["house_rules"] = []
    with pytest.raises(SpecError, match="house_rules"):
        validate_spec(spec)


def test_undeclared_piece_in_placement_rejected():
    spec = load_example_spec()
    spec["setup"]["initial_placements"] = [
        {"player": 0, "piece": "ghost", "cell": "a1"},
        {"player": 1, "piece": "ghost", "cell": "e5"},
    ]
    with pytest.raises(SpecError, match="ghost"):
        validate_spec(spec)


def test_off_board_placement_rejected():
    spec = load_example_spec()
    spec["setup"]["initial_placements"] = [
        {"player": 0, "piece": "stone", "cell": "z9"},
        {"player": 1, "piece": "stone", "cell": "a1"},
    ]
    with pytest.raises(SpecError, match="z9"):
        validate_spec(spec)


def test_setup_exceeding_declared_counts_rejected():
    # the example declares 2 wardens for player 1; placing a third must fail
    spec = load_example_spec()
    spec["setup"]["initial_placements"].append(
        {"player": 1, "piece": "warden", "cell": "c3"}
    )
    with pytest.raises(SpecError, match="declared count"):
        validate_spec(spec)


def test_missing_roles_rejected():
    spec = load_example_spec()
    del spec["roles"]
    with pytest.raises(SpecError, match="roles"):
        validate_spec(spec)


def test_identical_role_names_rejected():
    spec = load_example_spec()
    spec["roles"]["1"]["name"] = spec["roles"]["0"]["name"]
    with pytest.raises(SpecError, match="different names"):
        validate_spec(spec)


def test_bad_player_scope_rejected():
    spec = load_example_spec()
    spec["move_rules"][0]["player"] = 2
    with pytest.raises(SpecError, match="player"):
        validate_spec(spec)


def test_first_player_must_be_zero():
    spec = load_example_spec()
    spec["turn"]["first_player"] = 1
    with pytest.raises(SpecError, match="first_player"):
        validate_spec(spec)


def test_bad_move_category_rejected():
    spec = load_example_spec()
    spec["move_rules"][0]["category"] = "teleport"
    with pytest.raises(SpecError, match="category"):
        validate_spec(spec)


def test_too_few_edge_cases_rejected():
    spec = load_example_spec()
    spec["edge_cases"] = spec["edge_cases"][:2]
    with pytest.raises(SpecError, match="edge_cases"):
        validate_spec(spec)
