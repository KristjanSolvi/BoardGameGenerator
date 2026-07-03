import json

import pytest

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
