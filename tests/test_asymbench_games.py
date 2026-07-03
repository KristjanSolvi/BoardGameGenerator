import numpy as np
import pytest

from research.asymbench.games import MicroTafl
from research.asymbench.games.base import (
    IllegalActionError,
    RoleResult,
    make_action_mask,
)
from research.asymbench.games.breaker_builder import BreakerBuilder
from research.asymbench.games.micro_tafl import MicroTaflState


def test_make_action_mask_marks_only_legal_actions():
    mask = make_action_mask(action_size=6, legal_actions=[0, 2, 5])
    assert mask.dtype == np.bool_
    assert mask.tolist() == [True, False, True, False, False, True]


def test_make_action_mask_rejects_out_of_range_action():
    with pytest.raises(ValueError, match="outside action space"):
        make_action_mask(action_size=3, legal_actions=[0, 3])


def test_role_result_value_for_player():
    result = RoleResult(winner=1, reason="escape", plies=17)
    assert result.value_for_player(1) == 1.0
    assert result.value_for_player(0) == -1.0
    assert RoleResult(winner=None, reason="draw", plies=20).value_for_player(0) == 0.0


def _micro_tafl_state(
    pieces: dict[str, int],
    *,
    to_move: int = 0,
    seat_roles: tuple[int, int] = (MicroTafl.ATTACKER, MicroTafl.DEFENDER),
    plies: int = 0,
) -> MicroTaflState:
    board = [MicroTafl.EMPTY] * 25
    for cell, piece in pieces.items():
        board[MicroTafl.cell_index(cell)] = piece
    return MicroTaflState(
        board=tuple(board), to_move=to_move, seat_roles=seat_roles, plies=plies
    )


def test_micro_tafl_encode_slide_uses_public_cell_validation():
    game = MicroTafl()

    assert game.encode_slide("a1", "e5") == 24
    with pytest.raises(ValueError, match="cell must be in"):
        game.encode_slide("a", "e5")
    with pytest.raises(ValueError, match="cell outside board"):
        game.encode_slide("a0", "e5")


def test_micro_tafl_initial_state_roles_and_legal_moves():
    game = MicroTafl()
    state = game.initial_state()

    assert game.current_player(state) == 0
    assert game.player_role(state, 0) == MicroTafl.ATTACKER
    assert game.player_role(state, 1) == MicroTafl.DEFENDER
    assert not game.is_terminal(state)
    assert len(game.legal_actions(state)) > 0
    assert game.action_mask(state).sum() == len(game.legal_actions(state))


def test_micro_tafl_initial_state_accepts_seat_role_swap():
    game = MicroTafl()
    state = game.initial_state(
        seat_roles=(MicroTafl.DEFENDER, MicroTafl.ATTACKER)
    )

    assert game.player_role(state, 0) == MicroTafl.DEFENDER
    assert game.player_role(state, 1) == MicroTafl.ATTACKER


def test_micro_tafl_observation_shape_and_action_mask():
    game = MicroTafl()
    state = game.initial_state()

    obs = game.observation_tensor(state, player=0)
    mask = game.action_mask(state)

    assert obs.shape == (7, 5, 5)
    assert obs.dtype == np.float32
    assert mask.shape == (game.action_size,)


def test_micro_tafl_known_escape_sequence_reaches_defender_win():
    game = MicroTafl()
    state = game.initial_state(
        seat_roles=(MicroTafl.DEFENDER, MicroTafl.ATTACKER)
    )
    actions = [
        game.encode_slide("b3", "b2"),
        game.encode_slide("a3", "a4"),
        game.encode_slide("c3", "a3"),
        game.encode_slide("e3", "e4"),
        game.encode_slide("a3", "a1"),
    ]

    for action in actions:
        assert action in game.legal_actions(state)
        state = game.apply_action(state, action)

    result = game.result(state)
    defender_player = state.seat_roles.index(MicroTafl.DEFENDER)
    assert game.is_terminal(state)
    assert result.winner == defender_player
    assert result.reason == "king_escape"


def test_micro_tafl_rejects_illegal_action():
    game = MicroTafl()
    state = game.initial_state()
    illegal_action = game.encode_slide("a1", "a2")

    assert illegal_action not in game.legal_actions(state)
    with pytest.raises(IllegalActionError):
        game.apply_action(state, illegal_action)


def test_micro_tafl_captures_defender_guard_against_attacker_anchor():
    game = MicroTafl()
    state = _micro_tafl_state(
        {
            "a3": MicroTafl.ATTACKER_PIECE,
            "c3": MicroTafl.DEFENDER_GUARD,
            "d3": MicroTafl.ATTACKER_PIECE,
            "e4": MicroTafl.KING,
        }
    )

    state = game.apply_action(state, game.encode_slide("a3", "b3"))

    assert state.board[MicroTafl.cell_index("c3")] == MicroTafl.EMPTY


def test_micro_tafl_captures_non_king_piece_against_hostile_corner():
    game = MicroTafl()
    state = _micro_tafl_state(
        {
            "b1": MicroTafl.DEFENDER_GUARD,
            "d1": MicroTafl.ATTACKER_PIECE,
            "e4": MicroTafl.KING,
        }
    )

    state = game.apply_action(state, game.encode_slide("d1", "c1"))

    assert state.board[MicroTafl.cell_index("b1")] == MicroTafl.EMPTY


def test_micro_tafl_king_capture_by_opposite_attackers():
    game = MicroTafl()
    state = _micro_tafl_state(
        {
            "c1": MicroTafl.ATTACKER_PIECE,
            "c3": MicroTafl.KING,
            "c4": MicroTafl.ATTACKER_PIECE,
        }
    )

    state = game.apply_action(state, game.encode_slide("c1", "c2"))
    result = game.result(state)

    assert game.is_terminal(state)
    assert result.winner == 0
    assert result.reason == "king_capture"


def test_micro_tafl_king_capture_can_use_hostile_corner():
    game = MicroTafl()
    state = _micro_tafl_state(
        {
            "b1": MicroTafl.KING,
            "d1": MicroTafl.ATTACKER_PIECE,
        }
    )

    state = game.apply_action(state, game.encode_slide("d1", "c1"))
    result = game.result(state)

    assert game.is_terminal(state)
    assert result.winner == 0
    assert result.reason == "king_capture"


def test_micro_tafl_max_plies_is_draw():
    game = MicroTafl(max_plies=3)
    state = _micro_tafl_state(
        {
            "a3": MicroTafl.ATTACKER_PIECE,
            "c3": MicroTafl.KING,
        },
        plies=3,
    )

    result = game.result(state)

    assert game.is_terminal(state)
    assert result.winner is None
    assert result.reason == "max_plies"


def test_micro_tafl_no_legal_action_awards_win_to_opponent():
    game = MicroTafl()
    state = _micro_tafl_state(
        {
            "a1": MicroTafl.ATTACKER_PIECE,
            "b1": MicroTafl.DEFENDER_GUARD,
            "c1": MicroTafl.ATTACKER_PIECE,
            "d1": MicroTafl.DEFENDER_GUARD,
            "e1": MicroTafl.ATTACKER_PIECE,
            "a2": MicroTafl.DEFENDER_GUARD,
            "b2": MicroTafl.ATTACKER_PIECE,
            "c2": MicroTafl.DEFENDER_GUARD,
            "d2": MicroTafl.ATTACKER_PIECE,
            "e2": MicroTafl.DEFENDER_GUARD,
            "a3": MicroTafl.ATTACKER_PIECE,
            "b3": MicroTafl.DEFENDER_GUARD,
            "c3": MicroTafl.KING,
            "d3": MicroTafl.DEFENDER_GUARD,
            "e3": MicroTafl.ATTACKER_PIECE,
            "a4": MicroTafl.DEFENDER_GUARD,
            "b4": MicroTafl.ATTACKER_PIECE,
            "c4": MicroTafl.DEFENDER_GUARD,
            "d4": MicroTafl.ATTACKER_PIECE,
            "e4": MicroTafl.DEFENDER_GUARD,
            "a5": MicroTafl.ATTACKER_PIECE,
            "b5": MicroTafl.DEFENDER_GUARD,
            "c5": MicroTafl.ATTACKER_PIECE,
            "d5": MicroTafl.DEFENDER_GUARD,
            "e5": MicroTafl.ATTACKER_PIECE,
        }
    )

    result = game.result(state)

    assert game.legal_actions(state) == []
    assert game.is_terminal(state)
    assert result.winner == 1
    assert result.reason == "no_legal_actions"


def test_breaker_builder_initial_state_roles_and_actions():
    game = BreakerBuilder()
    state = game.initial_state()

    assert game.current_player(state) == 0
    assert game.player_role(state, 0) == BreakerBuilder.BUILDER
    assert game.player_role(state, 1) == BreakerBuilder.BREAKER
    assert len(game.legal_actions(state)) == 23
    assert game.action_mask(state).sum() == 23


def test_breaker_builder_observation_shape():
    game = BreakerBuilder()
    state = game.initial_state()
    state = game.apply_action(state, game.encode_place("a3"))

    builder_obs = game.observation_tensor(state, player=0)
    breaker_obs = game.observation_tensor(state, player=1)
    builder_row, builder_col = divmod(game.cell_index("a3"), 5)
    blocker_row, blocker_col = divmod(game.cell_index("b2"), 5)

    assert builder_obs.shape == (6, 5, 5)
    assert builder_obs.dtype == np.float32
    assert builder_obs[0, builder_row, builder_col] == 1.0
    assert builder_obs[1, blocker_row, blocker_col] == 1.0
    assert builder_obs[2, builder_row, builder_col] == 1.0
    assert builder_obs[3, blocker_row, blocker_col] == 1.0
    assert builder_obs[5].max() == 0.0

    assert breaker_obs[2, blocker_row, blocker_col] == 1.0
    assert breaker_obs[3, builder_row, builder_col] == 1.0
    assert breaker_obs[5].min() == 1.0


def test_breaker_builder_swapped_role_observation_perspective():
    game = BreakerBuilder()
    state = game.initial_state(
        seat_roles=(BreakerBuilder.BREAKER, BreakerBuilder.BUILDER)
    )
    blocker_row, blocker_col = divmod(game.cell_index("d4"), 5)

    obs = game.observation_tensor(state, player=0)

    assert obs[1, blocker_row, blocker_col] == 1.0
    assert obs[2, blocker_row, blocker_col] == 1.0
    assert obs[3].sum() == 0.0
    assert obs[4].min() == 1.0
    assert obs[5].min() == 1.0


def test_breaker_builder_known_connection_win():
    game = BreakerBuilder()
    state = game.initial_state()
    moves = [
        game.encode_place("a3"),
        game.encode_move("b2", "b1"),
        game.encode_place("b3"),
        game.encode_move("d4", "d5"),
        game.encode_place("c3"),
        game.encode_move("b1", "a1"),
        game.encode_place("d3"),
        game.encode_move("d5", "e5"),
        game.encode_place("e3"),
    ]

    for action in moves:
        assert action in game.legal_actions(state), game.render(state)
        state = game.apply_action(state, action)

    assert game.is_terminal(state)
    assert game.result(state).winner == 0
    assert game.result(state).reason == "builder_connection"


def test_breaker_builder_breaker_can_remove_adjacent_marker():
    game = BreakerBuilder()
    state = game.initial_state()
    state = game.apply_action(state, game.encode_place("b3"))

    remove = game.encode_remove("b2", "b3")

    assert remove in game.legal_actions(state)
    state = game.apply_action(state, remove)
    assert "b3" not in game.builder_cells(state)


def test_breaker_builder_illegal_builder_place_on_blocker_rejected():
    game = BreakerBuilder()
    state = game.initial_state()
    action = game.encode_place("b2")

    assert action not in game.legal_actions(state)
    with pytest.raises(IllegalActionError):
        game.apply_action(state, action)


def test_breaker_builder_initial_state_accepts_seat_role_swap():
    game = BreakerBuilder()
    state = game.initial_state(
        seat_roles=(BreakerBuilder.BREAKER, BreakerBuilder.BUILDER)
    )

    assert game.current_player(state) == 0
    assert game.player_role(state, 0) == BreakerBuilder.BREAKER
    assert game.player_role(state, 1) == BreakerBuilder.BUILDER
    assert all(action >= 25 for action in game.legal_actions(state))


def test_breaker_builder_initial_state_normalizes_list_seat_roles():
    game = BreakerBuilder()

    state = game.initial_state(
        seat_roles=[BreakerBuilder.BUILDER, BreakerBuilder.BREAKER]
    )

    assert state.seat_roles == (BreakerBuilder.BUILDER, BreakerBuilder.BREAKER)
    assert isinstance(state.seat_roles, tuple)


def test_breaker_builder_swapped_role_max_plies_winner_is_breaker_player():
    game = BreakerBuilder(max_plies=1)
    state = game.initial_state(
        seat_roles=(BreakerBuilder.BREAKER, BreakerBuilder.BUILDER)
    )
    state = game.apply_action(state, game.encode_move("b2", "b1"))

    result = game.result(state)

    assert game.is_terminal(state)
    assert result.winner == 0
    assert result.reason == "max_plies"


def test_breaker_builder_cell_validation_rejects_malformed_cells():
    game = BreakerBuilder()

    with pytest.raises(ValueError, match="cell must be in"):
        game.encode_place("a")
    with pytest.raises(ValueError, match="cell outside board"):
        game.encode_move("a0", "a1")
