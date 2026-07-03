import numpy as np

from research.asymbench.games.grid import (
    connected_component,
    coord_to_index,
    index_to_coord,
    neighbors,
    path_exists_between_edges,
    ray_cells,
    replace_cell,
)


def test_grid_coordinate_round_trip():
    assert coord_to_index(0, 0, cols=5) == 0
    assert coord_to_index(2, 3, cols=5) == 13
    assert index_to_coord(13, cols=5) == (2, 3)


def test_neighbors_are_in_bounds_and_orthogonal_by_default():
    assert set(neighbors(1, 1, rows=3, cols=3)) == {(0, 1), (2, 1), (1, 0), (1, 2)}
    assert set(neighbors(0, 0, rows=3, cols=3)) == {(1, 0), (0, 1)}


def test_ray_cells_walks_until_board_edge():
    assert ray_cells(2, 2, dr=0, dc=1, rows=5, cols=5) == [(2, 3), (2, 4)]
    assert ray_cells(2, 2, dr=-1, dc=0, rows=5, cols=5) == [(1, 2), (0, 2)]


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
