import numpy as np
import pytest

from research.asymbench.games import MicroTafl
from research.asymbench.games.base import IllegalActionError, make_action_mask, RoleResult


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


def _micro_tafl_action(from_cell: str, to_cell: str) -> int:
    def index(cell: str) -> int:
        file_index = ord(cell[0]) - ord("a")
        rank_index = int(cell[1]) - 1
        return rank_index * 5 + file_index

    return index(from_cell) * 25 + index(to_cell)


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
        _micro_tafl_action("b3", "b2"),
        _micro_tafl_action("a3", "a4"),
        _micro_tafl_action("c3", "a3"),
        _micro_tafl_action("e3", "e4"),
        _micro_tafl_action("a3", "a1"),
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
    illegal_action = _micro_tafl_action("a1", "a2")

    assert illegal_action not in game.legal_actions(state)
    with pytest.raises(IllegalActionError):
        game.apply_action(state, illegal_action)
