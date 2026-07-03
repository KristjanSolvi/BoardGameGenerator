import json

import numpy as np
import pytest

from research.asymbench.games.base import IllegalActionError
from research.asymbench.generation.escape_capture import (
    EscapeCaptureGame,
    EscapeCaptureGenerator,
)
from research.asymbench.generation.specs import (
    GeneratedGameSpec,
    GenerationConstraints,
    ValidationReport,
)
from research.asymbench.games.grid import (
    connected_component,
    coord_to_index,
    edge_cells,
    index_to_coord,
    in_bounds,
    neighbors,
    path_exists_between_edges,
    ray_cells,
    replace_cell,
)


def test_grid_coordinate_round_trip():
    assert coord_to_index(0, 0, cols=5) == 0
    assert coord_to_index(2, 3, cols=5) == 13
    assert index_to_coord(13, cols=5) == (2, 3)


def test_coord_to_index_rejects_out_of_bounds_coordinates():
    with pytest.raises(ValueError, match="coordinate outside board"):
        coord_to_index(0, 5, cols=5)
    with pytest.raises(ValueError, match="coordinate outside board"):
        coord_to_index(3, 0, rows=3, cols=5)


def test_indices_and_dimensions_are_validated():
    with pytest.raises(ValueError, match="index must be non-negative"):
        index_to_coord(-1, cols=5)
    with pytest.raises(ValueError, match="index outside board"):
        replace_cell((0, 0), index=2, value=1)
    with pytest.raises(ValueError, match="index outside board"):
        connected_component(start=9, occupied={9}, rows=3, cols=3)
    with pytest.raises(ValueError, match="cols must be positive"):
        index_to_coord(0, cols=0)
    with pytest.raises(ValueError, match="rows must be positive"):
        in_bounds(0, 0, rows=0, cols=3)


def test_neighbors_are_in_bounds_and_orthogonal_by_default():
    assert set(neighbors(1, 1, rows=3, cols=3)) == {(0, 1), (2, 1), (1, 0), (1, 2)}
    assert set(neighbors(0, 0, rows=3, cols=3)) == {(1, 0), (0, 1)}


def test_neighbors_can_include_diagonals():
    assert set(neighbors(1, 1, rows=3, cols=3, diagonal=True)) == {
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
    }


def test_ray_cells_walks_until_board_edge():
    assert ray_cells(2, 2, dr=0, dc=1, rows=5, cols=5) == [(2, 3), (2, 4)]
    assert ray_cells(2, 2, dr=-1, dc=0, rows=5, cols=5) == [(1, 2), (0, 2)]


def test_ray_cells_rejects_invalid_directions():
    with pytest.raises(ValueError, match="ray direction must be non-zero"):
        ray_cells(2, 2, dr=0, dc=0, rows=5, cols=5)
    with pytest.raises(ValueError, match="ray direction must be unit"):
        ray_cells(2, 2, dr=0, dc=2, rows=5, cols=5)
    with pytest.raises(ValueError, match="ray direction must be unit"):
        ray_cells(2, 2, dr=-2, dc=1, rows=5, cols=5)


def test_replace_cell_returns_new_tuple():
    board = (0, 0, 0, 0)
    updated = replace_cell(board, index=2, value=7)
    assert board == (0, 0, 0, 0)
    assert updated == (0, 0, 7, 0)


def test_connected_component_and_edge_path():
    occupied = {coord_to_index(1, col, cols=4) for col in range(4)}
    component = connected_component(
        start=coord_to_index(1, 0, cols=4),
        occupied=occupied,
        rows=3,
        cols=4,
    )
    assert component == occupied
    assert path_exists_between_edges(
        occupied=occupied,
        rows=3,
        cols=4,
        start_edge="west",
        target_edge="east",
    )
    assert not path_exists_between_edges(
        occupied=occupied,
        rows=3,
        cols=4,
        start_edge="north",
        target_edge="south",
    )


def test_edge_paths_require_connected_occupied_cells():
    occupied = {
        coord_to_index(1, 0, cols=4),
        coord_to_index(1, 1, cols=4),
        coord_to_index(1, 3, cols=4),
    }
    assert not path_exists_between_edges(
        occupied=occupied,
        rows=3,
        cols=4,
        start_edge="west",
        target_edge="east",
    )


def test_edge_cells_rejects_unknown_edge_name():
    with pytest.raises(ValueError, match="unknown edge"):
        edge_cells(rows=3, cols=3, edge="middle")


def test_generated_game_spec_round_trips_through_json_dict():
    spec = GeneratedGameSpec(
        family="escape_capture",
        name="escape_capture_seed_3",
        seed=3,
        board={"rows": 5, "cols": 5},
        roles=("attacker", "defender"),
        setup={"attackers": [1, 3], "guards": [12], "key": 6, "exits": [0]},
        actions={"movement": "orthogonal_step"},
        terminal_rules={"capture": "sandwich"},
        max_plies=50,
    )
    encoded = json.dumps(spec.to_dict(), sort_keys=True)
    decoded = GeneratedGameSpec.from_dict(json.loads(encoded))
    assert decoded == spec
    assert decoded.roles == ("attacker", "defender")


def test_generation_constraints_defaults_are_small_and_deterministic():
    constraints = GenerationConstraints()
    assert constraints.board_sizes == ((5, 5), (6, 6), (7, 7))
    assert constraints.max_plies_range == (40, 120)
    assert constraints.max_attempts == 100


def test_validation_report_round_trips_and_records_rejections():
    report = ValidationReport(
        family="connection_disruption",
        name="bad_game",
        valid=False,
        reasons=("builder already connected",),
        initial_branching_factor=0,
        random_role_win_rates={"0": 0.0, "1": 1.0},
        mcts_role_win_rates={},
        average_random_plies=3.5,
        terminal_reasons={"builder_connection": 4},
    )
    assert ValidationReport.from_dict(report.to_dict()) == report


def test_generated_game_spec_rejects_bool_numeric_fields():
    base = {
        "family": "escape_capture",
        "name": "escape_capture_seed_3",
        "seed": 3,
        "board": {"rows": 5, "cols": 5},
        "roles": ("attacker", "defender"),
        "max_plies": 50,
    }
    for field, value in (("seed", True), ("max_plies", False)):
        with pytest.raises(ValueError, match=field):
            GeneratedGameSpec(**{**base, field: value})
    for board in ({"rows": True, "cols": 5}, {"rows": 5, "cols": False}):
        with pytest.raises(ValueError, match="board"):
            GeneratedGameSpec(**{**base, "board": board})


def test_generation_constraints_rejects_bool_numeric_fields():
    with pytest.raises(ValueError, match="max_attempts"):
        GenerationConstraints(max_attempts=True)
    with pytest.raises(ValueError, match="board_sizes"):
        GenerationConstraints(board_sizes=((True, 5),))


def test_generation_constraints_rejects_malformed_board_size_entries():
    with pytest.raises(ValueError, match="board_sizes"):
        GenerationConstraints(board_sizes=(5,))
    with pytest.raises(ValueError, match="board_sizes"):
        GenerationConstraints(board_sizes=((1, 2, 3),))


def test_validation_report_from_dict_rejects_non_bool_valid():
    base = ValidationReport(
        family="connection_disruption",
        name="bad_game",
        valid=False,
        reasons=("builder already connected",),
    ).to_dict()
    for value in ("false", 1):
        with pytest.raises(ValueError, match="valid"):
            ValidationReport.from_dict({**base, "valid": value})


def test_validation_report_rejects_invalid_values():
    base = {
        "family": "connection_disruption",
        "name": "bad_game",
        "valid": False,
        "reasons": ("builder already connected",),
    }
    cases = [
        ({**base, "family": "unknown"}, "unknown family"),
        ({**base, "random_role_win_rates": {"0": 1.1}}, "win rate"),
        ({**base, "terminal_reasons": {"builder_connection": -1}}, "terminal"),
        ({**base, "average_random_plies": float("nan")}, "average_random_plies"),
        ({**base, "reasons": ()}, "reason"),
    ]
    for kwargs, match in cases:
        with pytest.raises(ValueError, match=match):
            ValidationReport(**kwargs)


def test_generated_artifacts_do_not_alias_nested_mutable_inputs_or_outputs():
    setup = {"attackers": [1, 3], "nested": {"key": 6}}
    actions = {"movement": {"kind": "orthogonal_step"}}
    spec = GeneratedGameSpec(
        family="escape_capture",
        name="escape_capture_seed_3",
        seed=3,
        board={"rows": 5, "cols": 5},
        roles=("attacker", "defender"),
        setup=setup,
        actions=actions,
        terminal_rules={"capture": {"kind": "sandwich"}},
        max_plies=50,
    )
    setup["attackers"].append(9)
    actions["movement"]["kind"] = "diagonal_step"
    spec_dict = spec.to_dict()
    spec_dict["setup"]["nested"]["key"] = 99
    spec_dict["actions"]["movement"]["kind"] = "slide"
    assert spec.setup["attackers"] == (1, 3)
    assert spec.setup["nested"]["key"] == 6
    assert spec.actions["movement"]["kind"] == "orthogonal_step"

    rates = {"0": 0.5}
    terminals = {"max_plies": 1}
    report = ValidationReport(
        family="escape_capture",
        name="report",
        valid=True,
        random_role_win_rates=rates,
        terminal_reasons=terminals,
    )
    rates["0"] = 1.0
    terminals["max_plies"] = 3
    report_dict = report.to_dict()
    report_dict["random_role_win_rates"]["0"] = 0.0
    report_dict["terminal_reasons"]["max_plies"] = 7
    assert report.random_role_win_rates["0"] == 0.5
    assert report.terminal_reasons["max_plies"] == 1


def test_generated_artifact_fields_are_recursively_immutable():
    spec = GeneratedGameSpec(
        family="escape_capture",
        name="escape_capture_seed_3",
        seed=3,
        board={"rows": 5, "cols": 5},
        roles=("attacker", "defender"),
        setup={"attackers": [1, 3], "nested": {"key": 6}},
        actions={"movement": {"kind": "orthogonal_step"}},
        terminal_rules={"capture": {"kind": "sandwich"}},
        max_plies=50,
    )
    with pytest.raises(TypeError):
        spec.setup["new"] = 1
    with pytest.raises((AttributeError, TypeError)):
        spec.setup["attackers"].append(9)
    with pytest.raises(TypeError):
        spec.actions["movement"]["kind"] = "diagonal_step"

    spec_dict = spec.to_dict()
    spec_dict["setup"]["attackers"].append(9)
    spec_dict["setup"]["nested"]["key"] = 99
    assert spec.setup["attackers"] == (1, 3)
    assert spec.setup["nested"]["key"] == 6

    report = ValidationReport(
        family="escape_capture",
        name="report",
        valid=True,
        terminal_reasons={"max_plies": 1},
    )
    with pytest.raises(TypeError):
        report.terminal_reasons["max_plies"] = 2
    report_dict = report.to_dict()
    report_dict["terminal_reasons"]["max_plies"] = 7
    assert report.terminal_reasons["max_plies"] == 1


def test_generated_artifacts_reject_non_string_text_fields():
    with pytest.raises(ValueError, match="name"):
        GeneratedGameSpec(
            family="escape_capture",
            name=3,
            seed=3,
            board={"rows": 5, "cols": 5},
            roles=("attacker", "defender"),
            max_plies=50,
        )
    with pytest.raises(ValueError, match="roles"):
        GeneratedGameSpec(
            family="escape_capture",
            name="escape_capture_seed_3",
            seed=3,
            board={"rows": 5, "cols": 5},
            roles=("attacker", 4),
            max_plies=50,
        )
    with pytest.raises(ValueError, match="reasons"):
        ValidationReport(
            family="connection_disruption",
            name="bad_game",
            valid=False,
            reasons=(None,),
        )


def _escape_capture_test_spec():
    return GeneratedGameSpec(
        family="escape_capture",
        name="escape_capture_runtime_test",
        seed=101,
        board={"rows": 5, "cols": 5},
        roles=("attacker", "defender"),
        setup={
            "attackers": [1, 3, 21, 23],
            "guards": [7, 17],
            "key": 12,
            "exits": [10],
            "hostile": [],
        },
        actions={"movement": "orthogonal_step"},
        terminal_rules={"capture": "opposite_sides"},
        max_plies=20,
    )


def _escape_capture_spec(
    *,
    board=None,
    roles=("attacker", "defender"),
    setup=None,
    max_plies=20,
):
    return GeneratedGameSpec(
        family="escape_capture",
        name="escape_capture_custom_test",
        seed=202,
        board=board if board is not None else {"rows": 3, "cols": 3},
        roles=roles,
        setup=setup
        if setup is not None
        else {
            "attackers": [1],
            "guards": [],
            "key": 4,
            "exits": [0],
            "hostile": [],
        },
        actions={"movement": "orthogonal_step"},
        terminal_rules={"capture": "opposite_sides"},
        max_plies=max_plies,
    )


def test_escape_capture_runtime_api_and_observation():
    game = EscapeCaptureGame(_escape_capture_test_spec())
    state = game.initial_state()
    assert game.name == "escape_capture_runtime_test"
    assert game.roles == ("attacker", "defender")
    assert game.board_shape == (5, 5)
    assert game.action_size == 625
    assert game.current_player(state) == 0
    assert game.player_role(state, 0) == 0
    assert len(game.legal_actions(state)) > 0
    assert game.action_mask(state).sum() == len(game.legal_actions(state))
    obs = game.observation_tensor(state, player=0)
    assert obs.shape == (9, 5, 5)
    assert obs.dtype == np.float32


def test_escape_capture_observation_includes_hostile_cells():
    spec = _escape_capture_spec(
        setup={
            "attackers": [1],
            "guards": [],
            "key": 4,
            "exits": [8],
            "hostile": [0],
        }
    )
    game = EscapeCaptureGame(spec)
    obs = game.observation_tensor(game.initial_state(), player=0)
    assert obs.shape == (9, 3, 3)
    assert obs[4, 0, 0] == 1.0
    assert obs[4].sum() == 1.0


def test_escape_capture_known_escape_path_reaches_defender_win():
    game = EscapeCaptureGame(_escape_capture_test_spec())
    state = game.initial_state(seat_roles=(1, 0))
    moves = [
        game.encode_move(12, 11),
        game.encode_move(1, 6),
        game.encode_move(11, 10),
    ]
    for action in moves:
        assert action in game.legal_actions(state), game.render(state)
        state = game.apply_action(state, action)
    assert game.is_terminal(state)
    assert game.result(state).winner == 0
    assert game.result(state).reason == "key_escape"


def test_escape_capture_rejects_illegal_move():
    game = EscapeCaptureGame(_escape_capture_test_spec())
    state = game.initial_state()
    illegal = game.encode_move(12, 13)
    assert illegal not in game.legal_actions(state)
    with pytest.raises(IllegalActionError):
        game.apply_action(state, illegal)


def test_escape_capture_rejects_reversed_roles():
    with pytest.raises(ValueError, match="roles"):
        EscapeCaptureGame(_escape_capture_spec(roles=("defender", "attacker")))


def test_escape_capture_rejects_no_attackers():
    with pytest.raises(ValueError, match="attackers"):
        EscapeCaptureGame(
            _escape_capture_spec(
                setup={
                    "attackers": [],
                    "guards": [],
                    "key": 4,
                    "exits": [8],
                    "hostile": [],
                }
            )
        )


def test_escape_capture_rejects_no_exits():
    with pytest.raises(ValueError, match="exits"):
        EscapeCaptureGame(
            _escape_capture_spec(
                setup={
                    "attackers": [1],
                    "guards": [],
                    "key": 4,
                    "exits": [],
                    "hostile": [],
                }
            )
        )


def test_escape_capture_rejects_initially_escaped_key():
    with pytest.raises(ValueError, match="key.*exit"):
        EscapeCaptureGame(
            _escape_capture_spec(
                setup={
                    "attackers": [1],
                    "guards": [],
                    "key": 4,
                    "exits": [4],
                    "hostile": [],
                }
            )
        )


def test_escape_capture_rejects_initially_captured_key():
    with pytest.raises(ValueError, match="captured"):
        EscapeCaptureGame(
            _escape_capture_spec(
                setup={
                    "attackers": [1, 7],
                    "guards": [],
                    "key": 4,
                    "exits": [0],
                    "hostile": [],
                }
            )
        )


def test_escape_capture_rejects_initial_state_with_no_legal_actions():
    with pytest.raises(ValueError, match="legal actions"):
        EscapeCaptureGame(
            _escape_capture_spec(
                setup={
                    "attackers": [0],
                    "guards": [1, 3],
                    "key": 4,
                    "exits": [8],
                    "hostile": [],
                }
            )
        )


def test_escape_capture_key_capture_maps_winner_for_swapped_roles():
    game = EscapeCaptureGame(
        _escape_capture_spec(
            setup={
                "attackers": [1, 8],
                "guards": [3],
                "key": 4,
                "exits": [0],
                "hostile": [],
            }
        )
    )
    state = game.initial_state(seat_roles=(1, 0))
    state = game.apply_action(state, game.encode_move(3, 6))
    state = game.apply_action(state, game.encode_move(8, 7))
    assert game.is_terminal(state)
    assert game.result(state).winner == 1
    assert game.result(state).reason == "key_capture"


def test_escape_capture_hostile_cell_participates_in_capture():
    game = EscapeCaptureGame(
        _escape_capture_spec(
            setup={
                "attackers": [8],
                "guards": [],
                "key": 4,
                "exits": [0],
                "hostile": [1],
            }
        )
    )
    state = game.apply_action(game.initial_state(), game.encode_move(8, 7))
    assert game.is_terminal(state)
    assert game.result(state).winner == 0
    assert game.result(state).reason == "key_capture"


def test_escape_capture_out_of_bounds_participates_in_edge_capture():
    game = EscapeCaptureGame(
        _escape_capture_spec(
            setup={
                "attackers": [5],
                "guards": [],
                "key": 1,
                "exits": [0],
                "hostile": [],
            }
        )
    )
    state = game.apply_action(game.initial_state(), game.encode_move(5, 4))
    assert game.is_terminal(state)
    assert game.result(state).winner == 0
    assert game.result(state).reason == "key_capture"


def test_escape_capture_defender_move_does_not_trigger_key_capture():
    game = EscapeCaptureGame(
        _escape_capture_spec(
            setup={
                "attackers": [1, 7],
                "guards": [],
                "key": 5,
                "exits": [0],
                "hostile": [],
            }
        )
    )
    state = game.initial_state(seat_roles=(1, 0))
    state = game.apply_action(state, game.encode_move(5, 4))
    assert not game.is_terminal(state)


def test_escape_capture_max_plies_is_draw():
    spec_data = _escape_capture_test_spec().to_dict()
    game = EscapeCaptureGame(
        GeneratedGameSpec.from_dict({**spec_data, "max_plies": 1})
    )
    state = game.apply_action(game.initial_state(), game.encode_move(1, 6))
    assert game.is_terminal(state)
    assert game.result(state).winner is None
    assert game.result(state).reason == "max_plies"


def test_escape_capture_generator_is_deterministic_by_seed():
    generator = EscapeCaptureGenerator()
    constraints = GenerationConstraints(board_sizes=((5, 5),), max_plies_range=(40, 40))
    first = generator.generate(seed=17, constraints=constraints)
    second = generator.generate(seed=17, constraints=constraints)
    assert first == second
    assert first.family == "escape_capture"
    assert first.board == {"rows": 5, "cols": 5}
    assert first.max_plies == 40


def test_escape_capture_generator_compiles_to_playable_game():
    generator = EscapeCaptureGenerator()
    spec = generator.generate(
        seed=18,
        constraints=GenerationConstraints(board_sizes=((5, 5),), max_plies_range=(40, 40)),
    )
    game = generator.compile(spec)
    state = game.initial_state()
    assert not game.is_terminal(state)
    assert len(game.legal_actions(state)) > 0
    assert game.action_mask(state).shape == (game.action_size,)


def test_escape_capture_generator_rejects_unsupported_board_sizes_immediately():
    generator = EscapeCaptureGenerator()
    constraints = GenerationConstraints(board_sizes=((3, 3),), max_attempts=1)
    with pytest.raises(ValueError, match="board_sizes.*at least 5"):
        generator.generate(seed=19, constraints=constraints)


@pytest.mark.parametrize("seed", [True, "abc"])
def test_escape_capture_generator_rejects_invalid_seed_types(seed):
    generator = EscapeCaptureGenerator()
    with pytest.raises(ValueError, match="seed"):
        generator.generate(seed=seed, constraints=GenerationConstraints())


def test_escape_capture_generator_seed_range_has_structural_invariants():
    generator = EscapeCaptureGenerator()
    constraints = GenerationConstraints(
        board_sizes=((5, 5), (6, 6), (7, 7)),
        max_plies_range=(40, 40),
    )
    for seed in range(20, 30):
        spec = generator.generate(seed=seed, constraints=constraints)
        rows = spec.board["rows"]
        cols = spec.board["cols"]
        setup = spec.setup
        key = setup["key"]
        key_row, key_col = index_to_coord(key, cols=cols)
        outer_ring = set(EscapeCaptureGenerator._outer_ring(rows, cols))
        occupied = list(setup["attackers"]) + list(setup["guards"]) + [key]

        assert spec.name == f"escape_capture_{rows}x{cols}_seed_{seed}"
        assert set(setup) == {"attackers", "guards", "key", "exits", "hostile"}
        assert len(occupied) == len(set(occupied))
        assert not set(setup["exits"]) & set(occupied)
        assert set(setup["exits"]) <= outer_ring
        assert set(setup["attackers"]) <= outer_ring
        assert all(
            guard in {
                coord_to_index(row, col, rows=rows, cols=cols)
                for row, col in neighbors(key_row, key_col, rows=rows, cols=cols)
            }
            for guard in setup["guards"]
        )
        assert generator._has_escape_path(spec)
        assert generator._has_capture_potential(spec)

        game = generator.compile(spec)
        state = game.initial_state()
        assert not game.is_terminal(state)
        assert len(game.legal_actions(state)) > 0


def test_escape_capture_generator_preserves_guard_count_diversity_on_5x5():
    generator = EscapeCaptureGenerator()
    constraints = GenerationConstraints(board_sizes=((5, 5),), max_plies_range=(40, 40))
    guard_counts = []
    for seed in range(51):
        spec = generator.generate(seed=seed, constraints=constraints)
        guard_counts.append(len(spec.setup["guards"]))

        game = generator.compile(spec)
        state = game.initial_state()
        assert not game.is_terminal(state)
        assert len(game.legal_actions(state)) > 0

    assert all(2 <= count <= 4 for count in guard_counts)
    assert any(count > 2 for count in guard_counts)


def test_escape_capture_capture_potential_treats_guards_as_vacatable():
    generator = EscapeCaptureGenerator()
    spec = GeneratedGameSpec(
        family="escape_capture",
        name="escape_capture_5x5_seed_303",
        seed=303,
        board={"rows": 5, "cols": 5},
        roles=("attacker", "defender"),
        setup={
            "attackers": [0, 4, 20, 24],
            "guards": [7, 11, 13],
            "key": 12,
            "exits": [2],
            "hostile": [],
        },
        actions={"movement": "orthogonal_step"},
        terminal_rules={"capture": "opposite_sides"},
        max_plies=40,
    )

    assert generator._has_capture_potential(spec)
