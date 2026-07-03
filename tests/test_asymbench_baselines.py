from research.asymbench.baselines import RandomAgent, evaluate_matchup, play_game
from research.asymbench.games.micro_tafl import MicroTafl


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
