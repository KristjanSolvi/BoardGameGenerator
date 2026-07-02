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


def test_asymmetric_setup_rejected():
    spec = load_example_spec()
    spec["setup"]["initial_placements"] = [
        {"player": 0, "piece": "stone", "cell": "a1"},
        {"player": 0, "piece": "stone", "cell": "b1"},
        {"player": 1, "piece": "stone", "cell": "e5"},
    ]
    with pytest.raises(SpecError, match="asymmetric"):
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
