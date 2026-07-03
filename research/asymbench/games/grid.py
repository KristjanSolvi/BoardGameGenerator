from __future__ import annotations

from collections import deque
from typing import Iterable


ORTHOGONAL_DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIAGONAL_DELTAS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _require_positive_dimensions(*, rows: int | None = None, cols: int) -> None:
    if cols <= 0:
        raise ValueError("cols must be positive")
    if rows is not None and rows <= 0:
        raise ValueError("rows must be positive")


def coord_to_index(row: int, col: int, *, cols: int, rows: int | None = None) -> int:
    _require_positive_dimensions(rows=rows, cols=cols)
    if row < 0 or (rows is not None and row >= rows) or col < 0 or col >= cols:
        raise ValueError("coordinate outside board")
    return row * cols + col


def index_to_coord(index: int, *, cols: int) -> tuple[int, int]:
    _require_positive_dimensions(cols=cols)
    if index < 0:
        raise ValueError("index must be non-negative")
    return divmod(index, cols)


def in_bounds(row: int, col: int, *, rows: int, cols: int) -> bool:
    _require_positive_dimensions(rows=rows, cols=cols)
    return 0 <= row < rows and 0 <= col < cols


def neighbors(
    row: int,
    col: int,
    *,
    rows: int,
    cols: int,
    diagonal: bool = False,
) -> list[tuple[int, int]]:
    if not in_bounds(row, col, rows=rows, cols=cols):
        raise ValueError("coordinate outside board")
    deltas = ORTHOGONAL_DELTAS + (DIAGONAL_DELTAS if diagonal else ())
    return [
        (row + dr, col + dc)
        for dr, dc in deltas
        if in_bounds(row + dr, col + dc, rows=rows, cols=cols)
    ]


def ray_cells(
    row: int,
    col: int,
    *,
    dr: int,
    dc: int,
    rows: int,
    cols: int,
) -> list[tuple[int, int]]:
    if not in_bounds(row, col, rows=rows, cols=cols):
        raise ValueError("coordinate outside board")
    if dr == 0 and dc == 0:
        raise ValueError("ray direction must be non-zero")
    if dr not in {-1, 0, 1} or dc not in {-1, 0, 1}:
        raise ValueError("ray direction must be unit")

    cells: list[tuple[int, int]] = []
    row += dr
    col += dc
    while in_bounds(row, col, rows=rows, cols=cols):
        cells.append((row, col))
        row += dr
        col += dc
    return cells


def replace_cell(board: tuple[int, ...], *, index: int, value: int) -> tuple[int, ...]:
    if index < 0 or index >= len(board):
        raise ValueError("index outside board")
    updated = list(board)
    updated[index] = value
    return tuple(updated)


def _validate_index(index: int, *, rows: int, cols: int) -> None:
    if index < 0 or index >= rows * cols:
        raise ValueError("index outside board")


def connected_component(
    *,
    start: int,
    occupied: set[int],
    rows: int,
    cols: int,
) -> set[int]:
    _require_positive_dimensions(rows=rows, cols=cols)
    _validate_index(start, rows=rows, cols=cols)
    for cell in occupied:
        _validate_index(cell, rows=rows, cols=cols)

    if start not in occupied:
        return set()

    seen = {start}
    queue: deque[int] = deque([start])
    while queue:
        current = queue.popleft()
        row, col = index_to_coord(current, cols=cols)
        for n_row, n_col in neighbors(row, col, rows=rows, cols=cols):
            neighbor = coord_to_index(n_row, n_col, cols=cols)
            if neighbor in occupied and neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return seen


def edge_cells(*, rows: int, cols: int, edge: str) -> set[int]:
    _require_positive_dimensions(rows=rows, cols=cols)
    if edge == "north":
        return {coord_to_index(0, col, cols=cols) for col in range(cols)}
    if edge == "south":
        return {coord_to_index(rows - 1, col, cols=cols) for col in range(cols)}
    if edge == "west":
        return {coord_to_index(row, 0, cols=cols) for row in range(rows)}
    if edge == "east":
        return {coord_to_index(row, cols - 1, cols=cols) for row in range(rows)}
    raise ValueError(f"unknown edge: {edge!r}")


def path_exists_between_edges(
    *,
    occupied: Iterable[int],
    rows: int,
    cols: int,
    start_edge: str,
    target_edge: str,
) -> bool:
    _require_positive_dimensions(rows=rows, cols=cols)
    occupied_set = set(occupied)
    for cell in occupied_set:
        _validate_index(cell, rows=rows, cols=cols)

    starts = edge_cells(rows=rows, cols=cols, edge=start_edge) & occupied_set
    targets = edge_cells(rows=rows, cols=cols, edge=target_edge) & occupied_set
    for start in starts:
        if connected_component(start=start, occupied=occupied_set, rows=rows, cols=cols) & targets:
            return True
    return False
