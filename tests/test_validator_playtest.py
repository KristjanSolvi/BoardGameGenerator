"""Validator + playtest harness exercised against the hand-written
reference engine and against deliberately broken engines."""

import shutil
from pathlib import Path

from gamegen.playtest import FlatMCAgent, RandomAgent, run_playtests
from gamegen.validator import load_engine_class, validate_engine

FIXTURE = Path(__file__).resolve().parent / "ref_engine.py"

PLAYTEST_CFG = {
    "random_vs_random_games": 40,
    "mc_vs_random_games": 6,
    "mc_vs_mc_games": 4,
    "mc_rollout_budget": 64,
    "mc_simulation_move_cap": 30,
    "move_cap": 30,
}


def _copy_fixture(tmp_path: Path, mutate=None) -> Path:
    dst = tmp_path / "engine.py"
    code = FIXTURE.read_text()
    if mutate:
        code = mutate(code)
    dst.write_text(code)
    return dst


def test_reference_engine_passes_validation(tmp_path):
    engine_path = _copy_fixture(tmp_path)
    report = validate_engine(engine_path, test_path=None, spec={}, seed=7,
                             random_playouts=100, move_cap=30,
                             illegal_move_samples=50)
    assert report.ok, report.failure_message
    assert report.checks["symmetry"]["opening_moves_per_player"] == 9
    assert report.checks["playouts"]["length_max"] <= 9


def test_asymmetric_engine_fails_symmetry(tmp_path):
    # break mirror_state: stop flipping the side to move
    engine_path = _copy_fixture(
        tmp_path,
        mutate=lambda code: code.replace(
            "return tuple(swap[c] for c in state[:9]) + (1 - state[9],)",
            "return tuple(swap[c] for c in state[:9]) + (state[9],)",
        ),
    )
    report = validate_engine(engine_path, test_path=None, spec={}, seed=7,
                             random_playouts=20, move_cap=30)
    assert not report.ok
    assert report.failed_check == "symmetry"


def test_unsound_apply_fails(tmp_path):
    # apply() stops checking legality -> illegal moves get accepted
    engine_path = _copy_fixture(
        tmp_path,
        mutate=lambda code: code.replace(
            "        if move not in self.legal_moves(state, state[9]):\n"
            "            raise IllegalMoveError(move)\n",
            "",
        ),
    )
    report = validate_engine(engine_path, test_path=None, spec={}, seed=7,
                             random_playouts=20, move_cap=30)
    assert not report.ok
    assert report.failed_check == "move_soundness"


def test_non_terminating_engine_fails(tmp_path):
    # game never ends (no terminal states, pass always legal) -> the
    # playout check must flag non-termination at the move cap
    engine_path = _copy_fixture(
        tmp_path,
        mutate=lambda code: code.replace(
            "        return [(\"place\", i) for i in range(9) if state[i] is None]\n",
            "        empty = [(\"place\", i) for i in range(9) if state[i] is None]\n"
            "        return empty + [(\"pass\",)]\n",
        ).replace(
            "        _, cell = move\n",
            "        if move == (\"pass\",):\n"
            "            return state[:9] + (1 - state[9],)\n"
            "        _, cell = move\n",
        ).replace(
            "        return self._winner(state) is not None or all(\n"
            "            cell is not None for cell in state[:9]\n"
            "        )\n",
            "        return False\n",
        ),
    )
    report = validate_engine(engine_path, test_path=None, spec={}, seed=7,
                             random_playouts=50, move_cap=30)
    assert not report.ok
    assert report.failed_check in ("playouts", "determinism")


def test_terminal_initial_state_fails_cleanly(tmp_path):
    # regression: an engine whose starting position is already terminal
    # must produce a clear 'playouts' failure, not a validator crash
    engine_path = _copy_fixture(
        tmp_path,
        mutate=lambda code: code.replace(
            "        return self._winner(state) is not None or all(",
            "        return True or all(",
        ),
    )
    report = validate_engine(engine_path, test_path=None, spec={}, seed=7,
                             random_playouts=20, move_cap=30)
    assert not report.ok
    assert report.failed_check == "playouts"
    assert "initial_state() is already terminal" in report.failure_message


def test_playtest_report_shape_and_reproducibility(tmp_path):
    engine_path = _copy_fixture(tmp_path)
    engine = load_engine_class(engine_path)()
    r1 = run_playtests(engine, seed=11, cfg=dict(PLAYTEST_CFG))
    r2 = run_playtests(engine, seed=11, cfg=dict(PLAYTEST_CFG))
    assert r1 == r2  # same seed -> byte-identical report
    head = r1["headline"]
    assert 0.0 <= head["first_player_win_rate_random"] <= 1.0
    assert 0.0 <= head["decisiveness_mc_vs_random"] <= 1.0
    assert r1["random_vs_random"]["games"] == 40
    assert r1["branching_factor"]["max"] == 9


def test_mc_agent_beats_random_at_tictactoe(tmp_path):
    # sanity: the generic search agent must show a skill signal on a
    # solved trivial game, otherwise decisiveness is meaningless
    engine_path = _copy_fixture(tmp_path)
    engine = load_engine_class(engine_path)()
    report = run_playtests(engine, seed=13, cfg=dict(PLAYTEST_CFG))
    assert report["mc_vs_random"]["decisiveness_mc_win_rate"] >= 0.6
