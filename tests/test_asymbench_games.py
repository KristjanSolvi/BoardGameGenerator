import numpy as np

from research.asymbench.games.base import make_action_mask, RoleResult


def test_make_action_mask_marks_only_legal_actions():
    mask = make_action_mask(action_size=6, legal_actions=[0, 2, 5])
    assert mask.dtype == np.bool_
    assert mask.tolist() == [True, False, True, False, False, True]


def test_make_action_mask_rejects_out_of_range_action():
    try:
        make_action_mask(action_size=3, legal_actions=[0, 3])
    except ValueError as exc:
        assert "outside action space" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_role_result_value_for_player():
    result = RoleResult(winner=1, reason="escape", plies=17)
    assert result.value_for_player(1) == 1.0
    assert result.value_for_player(0) == -1.0
    assert RoleResult(winner=None, reason="draw", plies=20).value_for_player(0) == 0.0
