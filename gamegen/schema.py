"""Strict validation of the game spec JSON (schema v1.0).

Standard-library only (no jsonschema dependency). Every check failure is
collected so the designer agent gets one complete error report per
attempt instead of a drip-feed.
"""

from __future__ import annotations

from typing import Any

SPEC_VERSION = "1.0"

TOP_LEVEL_REQUIRED = {
    "spec_version": str,
    "name": str,
    "tagline": str,
    "board": dict,
    "pieces": list,
    "setup": dict,
    "turn": dict,
    "move_rules": list,
    "win_conditions": list,
    "draw_conditions": list,
    "repetition_rule": dict,
    "edge_cases": list,
    "example_turn": str,
    "symmetry_statement": str,
    "design_rationale": str,
}
TOP_LEVEL_OPTIONAL = {"loss_conditions": list}

TOPOLOGIES = {"square", "hex", "graph"}
MOVE_CATEGORIES = {
    "place",
    "step",
    "jump",
    "slide",
    "push",
    "swap",
    "capture_replace",
    "capture_jump",
    "capture_custodial",
    "remove",
    "transform",
    "compound",
}
WIN_TYPES = {
    "connection",
    "alignment",
    "elimination",
    "territory",
    "race",
    "immobilization",
    "capture_count",
    "custom",
}
DRAW_TYPES = {"repetition", "mutual_immobility", "move_cap", "custom"}
NO_LEGAL_MOVE_RULES = {"pass", "lose", "draw"}
REPETITION_OUTCOMES = {"draw", "loss_for_mover"}


class SpecError(ValueError):
    """Spec failed strict validation; .errors lists every problem."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__(
            "Game spec failed validation:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )


def validate_spec(spec: Any) -> None:
    """Raise SpecError listing every problem, or return None if valid."""
    errors: list[str] = []
    if not isinstance(spec, dict):
        raise SpecError(["spec must be a JSON object"])

    known = set(TOP_LEVEL_REQUIRED) | set(TOP_LEVEL_OPTIONAL)
    for key in spec:
        if key not in known:
            errors.append(f"unknown top-level key: {key!r}")
    for key, typ in TOP_LEVEL_REQUIRED.items():
        if key not in spec:
            errors.append(f"missing required top-level key: {key!r}")
        elif not isinstance(spec[key], typ):
            errors.append(f"{key!r} must be a {typ.__name__}")
    for key, typ in TOP_LEVEL_OPTIONAL.items():
        if key in spec and not isinstance(spec[key], typ):
            errors.append(f"{key!r} must be a {typ.__name__}")
    if errors:
        raise SpecError(errors)  # structure too broken for deeper checks

    if spec["spec_version"] != SPEC_VERSION:
        errors.append(f"spec_version must be {SPEC_VERSION!r}")

    cells = _check_board(spec["board"], errors)
    piece_ids = _check_pieces(spec["pieces"], errors)
    _check_setup(spec["setup"], piece_ids, cells, errors)
    _check_turn(spec["turn"], errors)
    _check_move_rules(spec["move_rules"], piece_ids, errors)
    _check_conditions(spec["win_conditions"], "win_conditions", WIN_TYPES, errors,
                      required=True)
    _check_conditions(spec.get("loss_conditions", []), "loss_conditions",
                      WIN_TYPES, errors, required=False)
    _check_conditions(spec["draw_conditions"], "draw_conditions", DRAW_TYPES,
                      errors, required=False)
    _check_repetition(spec["repetition_rule"], errors)
    _check_edge_cases(spec["edge_cases"], errors)

    for key in ("example_turn", "symmetry_statement", "design_rationale",
                "name", "tagline"):
        if not spec[key].strip():
            errors.append(f"{key!r} must be a non-empty string")

    if errors:
        raise SpecError(errors)


# ----------------------------------------------------------------------
def _check_board(board: dict, errors: list[str]) -> set[str] | None:
    """Returns the set of cell names if enumerable, else None."""
    topo = board.get("topology")
    if topo not in TOPOLOGIES:
        errors.append(f"board.topology must be one of {sorted(TOPOLOGIES)}")
        return None
    if not isinstance(board.get("cell_notation"), str) or not board["cell_notation"].strip():
        errors.append("board.cell_notation must be a non-empty string")

    if topo == "square":
        for key in ("rows", "cols"):
            if not isinstance(board.get(key), int) or board[key] < 2:
                errors.append(f"square board needs integer {key} >= 2")
                return None
        if board["rows"] > 26 or board["cols"] > 26:
            errors.append("square board rows/cols must be <= 26")
        # canonical notation: columns a.., rows 1..
        return {
            f"{chr(ord('a') + c)}{r + 1}"
            for r in range(board["rows"])
            for c in range(board["cols"])
        }
    if topo == "hex":
        if not isinstance(board.get("size"), int) or board["size"] < 2:
            errors.append("hex board needs integer size >= 2")
        if board.get("shape") not in ("hexagon", "rhombus"):
            errors.append("hex board needs shape 'hexagon' or 'rhombus'")
        return None  # cell enumeration is engine-side for hex
    # graph
    nodes = board.get("nodes")
    edges = board.get("edges")
    if not isinstance(nodes, list) or not nodes or not all(
        isinstance(n, str) and n for n in nodes
    ):
        errors.append("graph board needs a non-empty list of string nodes")
        return None
    if len(set(nodes)) != len(nodes):
        errors.append("graph board nodes must be unique")
    if not isinstance(edges, list) or not edges:
        errors.append("graph board needs a non-empty list of edges")
        return set(nodes)
    node_set = set(nodes)
    for e in edges:
        if (not isinstance(e, list) or len(e) != 2
                or any(n not in node_set for n in e)):
            errors.append(f"bad edge {e!r}: must be [node, node] over declared nodes")
    return node_set


def _check_pieces(pieces: list, errors: list[str]) -> set[str]:
    ids: set[str] = set()
    if not pieces:
        errors.append("pieces must be non-empty")
    for i, p in enumerate(pieces):
        if not isinstance(p, dict):
            errors.append(f"pieces[{i}] must be an object")
            continue
        pid = p.get("id")
        if not isinstance(pid, str) or not pid:
            errors.append(f"pieces[{i}].id must be a non-empty string")
            continue
        if pid in ids:
            errors.append(f"duplicate piece id {pid!r}")
        ids.add(pid)
        if not isinstance(p.get("name"), str) or not p["name"]:
            errors.append(f"piece {pid!r}: missing name")
        cnt = p.get("per_player_count")
        if not isinstance(cnt, int) or cnt < 0:
            errors.append(f"piece {pid!r}: per_player_count must be an int >= 0")
        if not isinstance(p.get("physical"), str) or not p["physical"]:
            errors.append(f"piece {pid!r}: missing 'physical' description")
    return ids


def _check_setup(setup: dict, piece_ids: set[str], cells: set[str] | None,
                 errors: list[str]) -> None:
    if not isinstance(setup.get("description"), str) or not setup["description"].strip():
        errors.append("setup.description must be a non-empty string")
    placements = setup.get("initial_placements")
    if not isinstance(placements, list):
        errors.append("setup.initial_placements must be a list (may be empty)")
        return
    per_player: dict[int, list[str]] = {0: [], 1: []}
    for i, pl in enumerate(placements):
        if not isinstance(pl, dict):
            errors.append(f"initial_placements[{i}] must be an object")
            continue
        player = pl.get("player")
        piece = pl.get("piece")
        cell = pl.get("cell")
        if player not in (0, 1):
            errors.append(f"initial_placements[{i}].player must be 0 or 1")
            continue
        if piece not in piece_ids:
            errors.append(f"initial_placements[{i}].piece {piece!r} not declared")
        if not isinstance(cell, str) or not cell:
            errors.append(f"initial_placements[{i}].cell must be a string")
            continue
        if cells is not None and cell not in cells:
            errors.append(
                f"initial_placements[{i}].cell {cell!r} is not on the board "
                "(square boards use canonical notation a1..)"
            )
        per_player[player].append(piece)
    if sorted(per_player[0]) != sorted(per_player[1]):
        errors.append(
            "setup is asymmetric: players 0 and 1 must place the same "
            f"multiset of pieces (got {sorted(per_player[0])} vs "
            f"{sorted(per_player[1])})"
        )


def _check_turn(turn: dict, errors: list[str]) -> None:
    if not isinstance(turn.get("structure"), str) or not turn["structure"].strip():
        errors.append("turn.structure must be a non-empty string")
    if turn.get("first_player") != 0:
        errors.append("turn.first_player must be 0 (fixed convention)")
    if not isinstance(turn.get("pass_allowed"), bool):
        errors.append("turn.pass_allowed must be a boolean")
    if turn.get("no_legal_move_rule") not in NO_LEGAL_MOVE_RULES:
        errors.append(
            f"turn.no_legal_move_rule must be one of {sorted(NO_LEGAL_MOVE_RULES)}"
        )


def _check_move_rules(rules: list, piece_ids: set[str], errors: list[str]) -> None:
    if not rules:
        errors.append("move_rules must be non-empty")
    seen_ids: set[str] = set()
    for i, r in enumerate(rules):
        if not isinstance(r, dict):
            errors.append(f"move_rules[{i}] must be an object")
            continue
        rid = r.get("id")
        if not isinstance(rid, str) or not rid:
            errors.append(f"move_rules[{i}].id must be a non-empty string")
        elif rid in seen_ids:
            errors.append(f"duplicate move rule id {rid!r}")
        else:
            seen_ids.add(rid)
        applies = r.get("applies_to")
        if (not isinstance(applies, list) or not applies
                or any(p not in piece_ids for p in applies)):
            errors.append(
                f"move_rules[{i}].applies_to must list declared piece ids"
            )
        if r.get("category") not in MOVE_CATEGORIES:
            errors.append(
                f"move_rules[{i}].category must be one of {sorted(MOVE_CATEGORIES)}"
            )
        if not isinstance(r.get("parameters"), dict):
            errors.append(f"move_rules[{i}].parameters must be an object")
        if not isinstance(r.get("text"), str) or len(r.get("text", "")) < 20:
            errors.append(
                f"move_rules[{i}].text must be a precise English rule "
                "(at least 20 characters)"
            )


def _check_conditions(conds: list, field: str, types: set[str],
                      errors: list[str], required: bool) -> None:
    if required and not conds:
        errors.append(f"{field} must be non-empty")
    for i, c in enumerate(conds):
        if not isinstance(c, dict):
            errors.append(f"{field}[{i}] must be an object")
            continue
        if c.get("type") not in types:
            errors.append(f"{field}[{i}].type must be one of {sorted(types)}")
        if not isinstance(c.get("text"), str) or len(c.get("text", "")) < 10:
            errors.append(f"{field}[{i}].text must be a precise English condition")


def _check_repetition(rep: dict, errors: list[str]) -> None:
    if not isinstance(rep.get("enabled"), bool):
        errors.append("repetition_rule.enabled must be a boolean")
        return
    if rep["enabled"]:
        if not isinstance(rep.get("count"), int) or rep["count"] < 2:
            errors.append("repetition_rule.count must be an int >= 2")
        if rep.get("outcome") not in REPETITION_OUTCOMES:
            errors.append(
                f"repetition_rule.outcome must be one of {sorted(REPETITION_OUTCOMES)}"
            )


def _check_edge_cases(cases: list, errors: list[str]) -> None:
    if len(cases) < 3:
        errors.append("edge_cases must contain at least 3 entries")
    for i, c in enumerate(cases):
        if (not isinstance(c, dict)
                or not isinstance(c.get("situation"), str)
                or not isinstance(c.get("ruling"), str)
                or not c.get("situation", "").strip()
                or not c.get("ruling", "").strip()):
            errors.append(
                f"edge_cases[{i}] must be an object with non-empty "
                "'situation' and 'ruling' strings"
            )
