import pytest

from research.asymbench.baselines import RandomAgent, evaluate_matchup, play_game
from research.asymbench.games.base import RoleResult
from research.asymbench.games.micro_tafl import MicroTafl


class AlwaysFirstPlayerWinsGame:
    roles = ("left", "right")
    max_plies = 4

    def initial_state(self, seat_roles=(0, 1)):
        return {"seat_roles": tuple(seat_roles), "plies": 0}

    def current_player(self, state):
        return state["plies"] % 2

    def player_role(self, state, player: int) -> int:
        return state["seat_roles"][player]

    def legal_actions(self, state):
        return [0]

    def apply_action(self, state, action):
        return {**state, "plies": state["plies"] + 1}

    def is_terminal(self, state):
        return state["plies"] >= 1

    def result(self, state):
        return RoleResult(winner=0, reason="scripted_first_player_win", plies=1)


class NeverTerminalGame(AlwaysFirstPlayerWinsGame):
    max_plies = 3

    def is_terminal(self, state):
        return False


def test_play_game_returns_role_aware_result():
    game = MicroTafl(max_plies=20)
    result = play_game(
        game,
        agents={0: RandomAgent(seed=1), 1: RandomAgent(seed=2)},
        seed=3,
        seat_roles=(MicroTafl.ATTACKER, MicroTafl.DEFENDER),
    )
    assert set(result) >= {"winner", "reason", "plies", "winner_role", "seat_roles"}
    assert result["seat_roles"] == [MicroTafl.ATTACKER, MicroTafl.DEFENDER]


def test_evaluate_matchup_is_reproducible():
    game = MicroTafl(max_plies=20)
    first = evaluate_matchup(
        game,
        agent_factories={
            0: lambda seed: RandomAgent(seed),
            1: lambda seed: RandomAgent(seed),
        },
        games=8,
        seed=11,
    )
    second = evaluate_matchup(
        game,
        agent_factories={
            0: lambda seed: RandomAgent(seed),
            1: lambda seed: RandomAgent(seed),
        },
        games=8,
        seed=11,
    )
    assert first == second
    assert first["games"] == 8
    assert "role_win_rates" in first
    assert "first_player_win_rate" in first


def test_evaluate_matchup_tracks_roles_under_seat_role_alternation():
    game = AlwaysFirstPlayerWinsGame()

    result = evaluate_matchup(
        game,
        agent_factories={
            0: lambda seed: RandomAgent(seed),
            1: lambda seed: RandomAgent(seed),
        },
        games=4,
        seed=23,
    )

    assert result["p0_win_rate"] == 1.0
    assert result["p1_win_rate"] == 0.0
    assert result["draw_rate"] == 0.0
    assert result["first_player_win_rate"] == 1.0
    assert result["role_win_rates"] == {"0": 0.5, "1": 0.5}
    assert result["termination_reasons"] == {"scripted_first_player_win": 4}
    assert [outcome["winner_role"] for outcome in result["outcomes"]] == [
        0,
        1,
        0,
        1,
    ]
    assert [outcome["seat_roles"] for outcome in result["outcomes"]] == [
        [0, 1],
        [1, 0],
        [0, 1],
        [1, 0],
    ]


def test_evaluate_matchup_rates_are_bounded_and_tracks_outcomes():
    game = MicroTafl(max_plies=20)
    summary = evaluate_matchup(
        game,
        agent_factories={
            0: lambda seed: RandomAgent(seed),
            1: lambda seed: RandomAgent(seed),
        },
        games=8,
        seed=17,
    )

    assert len(summary["outcomes"]) == 8
    for key in ("p0_win_rate", "p1_win_rate", "draw_rate", "first_player_win_rate"):
        assert 0.0 <= summary[key] <= 1.0
    assert all(0.0 <= rate <= 1.0 for rate in summary["role_win_rates"].values())


def test_play_game_raises_when_defensive_play_limit_is_reached():
    game = NeverTerminalGame()

    with pytest.raises(RuntimeError, match="exceeded max_steps"):
        play_game(
            game,
            agents={0: RandomAgent(seed=1), 1: RandomAgent(seed=2)},
            seed=3,
            max_steps=2,
        )


def test_evaluate_matchup_rejects_empty_seat_roles_list():
    game = AlwaysFirstPlayerWinsGame()

    with pytest.raises(ValueError, match="seat_roles_list must not be empty"):
        evaluate_matchup(
            game,
            agent_factories={
                0: lambda seed: RandomAgent(seed),
                1: lambda seed: RandomAgent(seed),
            },
            games=1,
            seed=5,
            seat_roles_list=(),
        )
