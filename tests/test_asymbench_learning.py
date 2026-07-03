import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from research.asymbench.experiments.run_role_heads import run_experiment
from research.asymbench.generation.escape_capture import EscapeCaptureGenerator
from research.asymbench.generation.specs import GenerationConstraints
from research.asymbench.games.base import RoleResult
from research.asymbench.games.breaker_builder import BreakerBuilder
from research.asymbench.learning import selfplay as selfplay_module
from research.asymbench.learning.evaluate import evaluate_model_vs_random
from research.asymbench.learning.model import PolicyValueNet
from research.asymbench.learning.replay import ReplayBuffer, TrainingExample
from research.asymbench.learning.selfplay import NeuralEvaluator, generate_selfplay_game
from research.asymbench.learning.train import train_steps


class _TinyEvaluatorGame:
    action_size = 2

    def __init__(self, terminal: bool = False) -> None:
        self.terminal = terminal

    def is_terminal(self, state):
        del state
        return self.terminal

    def legal_actions(self, state):
        del state
        return [0]

    def observation_tensor(self, state, player: int):
        del state, player
        return np.zeros((1, 1, 1), dtype=np.float32)

    def player_role(self, state, player: int):
        del state, player
        return 0

    def action_mask(self, state):
        del state
        return np.array([True, False], dtype=np.bool_)


class _DominantIllegalLogitModel(torch.nn.Module):
    def forward(self, observations, roles, action_mask):
        del roles, action_mask
        logits = torch.tensor(
            [[0.0, 1.0e30]], dtype=observations.dtype, device=observations.device
        )
        values = torch.tensor(
            [0.25], dtype=observations.dtype, device=observations.device
        )
        return logits, values


class _NanLegalLogitModel(torch.nn.Module):
    def forward(self, observations, roles, action_mask):
        del roles, action_mask
        logits = torch.tensor(
            [[float("nan"), 0.0]],
            dtype=observations.dtype,
            device=observations.device,
        )
        values = torch.zeros(
            (1,), dtype=observations.dtype, device=observations.device
        )
        return logits, values


class _TwoActionModel(torch.nn.Module):
    def forward(self, observations, roles, action_mask):
        del roles, action_mask
        batch_size = observations.shape[0]
        logits = torch.zeros(
            (batch_size, 2), dtype=observations.dtype, device=observations.device
        )
        values = torch.zeros(
            (batch_size,), dtype=observations.dtype, device=observations.device
        )
        return logits, values


class _TwoPlyGame:
    action_size = 2
    max_plies = 2

    def initial_state(self, seat_roles=(0, 1)):
        del seat_roles
        return 0

    def is_terminal(self, state: int):
        return state >= 2

    def legal_actions(self, state: int):
        return [] if self.is_terminal(state) else [0]

    def current_player(self, state: int):
        return state % 2

    def player_role(self, state: int, player: int):
        del state
        return player

    def observation_tensor(self, state: int, player: int):
        return np.full((1, 1, 1), state + player, dtype=np.float32)

    def action_mask(self, state: int):
        return np.array([not self.is_terminal(state), False], dtype=np.bool_)

    def apply_action(self, state: int, action: int):
        assert action == 0
        return state + 1

    def result(self, state: int):
        assert state == 2
        return RoleResult(winner=1, reason="scripted", plies=2)


class _InvalidPolicyAgent:
    def __init__(self, evaluator, simulations: int, seed: int | None = None):
        del evaluator, simulations, seed

    def policy(self, game, state, player: int):
        del state, player
        return np.array([0.5, 0.5], dtype=np.float64)[: game.action_size]


def _train_buffer_with_policy(policy, action_mask=None):
    if action_mask is None:
        action_mask = np.array([True, True], dtype=np.bool_)
    buffer = ReplayBuffer(capacity=1, seed=1)
    buffer.add(
        TrainingExample(
            observation=np.zeros((1, 1, 1), dtype=np.float32),
            role=0,
            action_mask=np.asarray(action_mask, dtype=np.bool_),
            policy=np.asarray(policy, dtype=np.float32),
            value=0.0,
        )
    )
    return buffer


def _tiny_policy_value_model():
    return PolicyValueNet((1, 1, 1), action_size=2, num_roles=1)


def test_policy_value_net_shared_and_role_heads_shapes():
    game = BreakerBuilder()
    obs = torch.zeros((2, 6, 5, 5), dtype=torch.float32)
    roles = torch.tensor([0, 1], dtype=torch.long)
    mask = torch.ones((2, game.action_size), dtype=torch.bool)

    shared = PolicyValueNet((6, 5, 5), game.action_size, num_roles=2, role_heads=False)
    role = PolicyValueNet((6, 5, 5), game.action_size, num_roles=2, role_heads=True)

    for model in (shared, role):
        logits, values = model(obs, roles, mask)
        assert logits.shape == (2, game.action_size)
        assert values.shape == (2,)
        assert torch.isfinite(logits[mask]).all()
        assert torch.all(values <= 1.0)
        assert torch.all(values >= -1.0)


def test_policy_value_net_masks_illegal_actions():
    game = BreakerBuilder()
    obs = torch.zeros((1, 6, 5, 5), dtype=torch.float32)
    roles = torch.tensor([0], dtype=torch.long)
    mask = torch.zeros((1, game.action_size), dtype=torch.bool)
    mask[0, 3] = True
    model = PolicyValueNet((6, 5, 5), game.action_size, num_roles=2, role_heads=False)
    logits, _ = model(obs, roles, mask)
    assert torch.isfinite(logits[0, 3])
    assert logits[0, 4].item() < -1e8


def test_policy_value_net_uses_role_specific_heads_by_role_id():
    model = PolicyValueNet((6, 5, 5), action_size=4, num_roles=2, role_heads=True)
    obs = torch.zeros((2, 6, 5, 5), dtype=torch.float32)
    mask = torch.ones((2, 4), dtype=torch.bool)

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.policy_heads[0].bias.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        model.policy_heads[1].bias.copy_(torch.tensor([5.0, 6.0, 7.0, 8.0]))
        model.value_heads[0].bias.fill_(0.25)
        model.value_heads[1].bias.fill_(-0.5)

    logits, values = model(obs, torch.tensor([0, 1], dtype=torch.long), mask)

    assert torch.allclose(logits[0], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.allclose(logits[1], torch.tensor([5.0, 6.0, 7.0, 8.0]))
    assert torch.allclose(values, torch.tanh(torch.tensor([0.25, -0.5])))

    swapped_logits, swapped_values = model(
        obs, torch.tensor([1, 0], dtype=torch.long), mask
    )

    assert torch.allclose(swapped_logits[0], torch.tensor([5.0, 6.0, 7.0, 8.0]))
    assert torch.allclose(swapped_logits[1], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.allclose(swapped_values, torch.tanh(torch.tensor([-0.5, 0.25])))


def test_policy_value_net_rejects_non_bool_action_mask():
    model = PolicyValueNet((6, 5, 5), action_size=4)
    obs = torch.zeros((1, 6, 5, 5), dtype=torch.float32)
    roles = torch.tensor([0], dtype=torch.long)
    mask = torch.ones((1, 4), dtype=torch.float32)

    with pytest.raises(ValueError, match="action_mask must use bool dtype"):
        model(obs, roles, mask)


def test_policy_value_net_rejects_invalid_role_id():
    model = PolicyValueNet((6, 5, 5), action_size=4)
    obs = torch.zeros((1, 6, 5, 5), dtype=torch.float32)
    roles = torch.tensor([2], dtype=torch.long)
    mask = torch.ones((1, 4), dtype=torch.bool)

    with pytest.raises(ValueError, match=r"roles must be in \[0, 2\)"):
        model(obs, roles, mask)


def test_policy_value_net_rejects_wrong_mask_shape():
    model = PolicyValueNet((6, 5, 5), action_size=4)
    obs = torch.zeros((1, 6, 5, 5), dtype=torch.float32)
    roles = torch.tensor([0], dtype=torch.long)
    mask = torch.ones((1, 3), dtype=torch.bool)

    with pytest.raises(ValueError, match="action_mask must have shape"):
        model(obs, roles, mask)


def test_policy_value_net_rejects_all_false_action_mask_row():
    model = PolicyValueNet((6, 5, 5), action_size=4)
    obs = torch.zeros((2, 6, 5, 5), dtype=torch.float32)
    roles = torch.tensor([0, 1], dtype=torch.long)
    mask = torch.tensor(
        [[True, False, False, False], [False, False, False, False]],
        dtype=torch.bool,
    )

    with pytest.raises(ValueError, match="at least one legal action per row"):
        model(obs, roles, mask)


def test_policy_value_net_uses_dtype_aware_mask_sentinel_on_cpu():
    model = PolicyValueNet((6, 5, 5), action_size=4)
    obs = torch.zeros((1, 6, 5, 5), dtype=torch.float32)
    roles = torch.tensor([0], dtype=torch.long)
    mask = torch.tensor([[True, False, True, True]], dtype=torch.bool)

    logits, _ = model(obs, roles, mask)

    assert logits.dtype == torch.float32
    assert logits[0, 1].item() == torch.finfo(logits.dtype).min


def test_policy_value_net_uses_float64_mask_sentinel_on_cpu():
    model = PolicyValueNet((6, 5, 5), action_size=4).double()
    obs = torch.zeros((1, 6, 5, 5), dtype=torch.float64)
    roles = torch.tensor([0], dtype=torch.long)
    mask = torch.tensor([[True, False, True, True]], dtype=torch.bool)

    logits, _ = model(obs, roles, mask)

    assert logits.dtype == torch.float64
    assert torch.isfinite(logits[0, 0])
    assert logits[0, 1].item() == torch.finfo(torch.float64).min
    assert logits[0, 1] < logits[0, 0]


def test_replay_buffer_samples_examples():
    buffer = ReplayBuffer(capacity=4, seed=1)
    example = TrainingExample(
        observation=np.zeros((6, 5, 5), dtype=np.float32),
        role=0,
        action_mask=np.ones(10, dtype=np.bool_),
        policy=np.ones(10, dtype=np.float32) / 10,
        value=1.0,
    )
    for _ in range(5):
        buffer.add(example)
    batch = buffer.sample(3)
    assert len(buffer) == 4
    assert len(batch) == 3


def test_replay_buffer_sampling_is_deterministic_for_seed():
    examples = [
        TrainingExample(
            observation=np.full((1,), index, dtype=np.float32),
            role=index % 2,
            action_mask=np.ones(2, dtype=np.bool_),
            policy=np.ones(2, dtype=np.float32) / 2,
            value=float(index),
        )
        for index in range(5)
    ]
    left = ReplayBuffer(capacity=5, seed=7)
    right = ReplayBuffer(capacity=5, seed=7)
    for example in examples:
        left.add(example)
        right.add(example)

    left_values = [example.value for example in left.sample(3)]
    right_values = [example.value for example in right.sample(3)]

    assert left_values == right_values


def test_neural_evaluator_returns_legal_prior_and_value():
    game = BreakerBuilder(max_plies=8)
    state = game.initial_state()
    model = PolicyValueNet((6, 5, 5), game.action_size, num_roles=2, role_heads=True)
    evaluator = NeuralEvaluator(model, device="cpu")
    prior, value = evaluator.evaluate(game, state, player=0)
    assert prior.shape == (game.action_size,)
    assert np.isclose(prior.sum(), 1.0)
    assert set(np.flatnonzero(prior)).issubset(set(game.legal_actions(state)))
    assert -1.0 <= value <= 1.0


def test_neural_evaluator_rejects_terminal_state_with_legal_actions():
    game = _TinyEvaluatorGame(terminal=True)
    evaluator = NeuralEvaluator(_DominantIllegalLogitModel(), device="cpu")

    with pytest.raises(ValueError, match="terminal"):
        evaluator.evaluate(game, state=object(), player=0)


def test_neural_evaluator_masks_logits_before_softmax():
    game = _TinyEvaluatorGame()
    evaluator = NeuralEvaluator(_DominantIllegalLogitModel(), device="cpu")

    prior, value = evaluator.evaluate(game, state=object(), player=0)

    assert np.isfinite(prior).all()
    assert prior[0] == pytest.approx(1.0)
    assert prior[1] == pytest.approx(0.0)
    assert prior.sum() == pytest.approx(1.0)
    assert value == pytest.approx(0.25)


def test_neural_evaluator_restores_training_mode():
    game = _TinyEvaluatorGame()
    model = _DominantIllegalLogitModel()
    model.train()
    evaluator = NeuralEvaluator(model, device="cpu")

    evaluator.evaluate(game, state=object(), player=0)

    assert model.training is True


def test_neural_evaluator_rejects_invalid_policy_logits():
    game = _TinyEvaluatorGame()
    evaluator = NeuralEvaluator(_NanLegalLogitModel(), device="cpu")

    with pytest.raises(ValueError, match="finite"):
        evaluator.evaluate(game, state=object(), player=0)


def test_generate_selfplay_game_produces_training_examples():
    game = BreakerBuilder(max_plies=8)
    model = PolicyValueNet((6, 5, 5), game.action_size, num_roles=2, role_heads=True)
    examples, outcome = generate_selfplay_game(
        game=game,
        model=model,
        device="cpu",
        simulations=4,
        seed=123,
    )
    assert len(examples) > 0
    assert outcome["plies"] == len(examples)
    assert all(ex.observation.shape == (6, 5, 5) for ex in examples)
    assert all(ex.action_mask.dtype == np.bool_ for ex in examples)
    assert all(ex.policy.shape == (game.action_size,) for ex in examples)
    assert all(-1.0 <= ex.value <= 1.0 for ex in examples)


def test_generate_selfplay_game_values_use_acting_player_perspective():
    examples, outcome = generate_selfplay_game(
        game=_TwoPlyGame(),
        model=_TwoActionModel(),
        device="cpu",
        simulations=1,
        seed=123,
    )

    assert outcome == {"winner": 1, "reason": "scripted", "plies": 2}
    assert [example.role for example in examples] == [0, 1]
    assert [example.value for example in examples] == [-1.0, 1.0]


def test_generate_selfplay_game_rejects_illegal_policy_mass(monkeypatch):
    monkeypatch.setattr(selfplay_module, "MCTSAgent", _InvalidPolicyAgent)

    with pytest.raises(ValueError, match="illegal"):
        generate_selfplay_game(
            game=_TwoPlyGame(),
            model=_TwoActionModel(),
            device="cpu",
            simulations=1,
            seed=123,
        )


def test_train_steps_updates_model_and_returns_metrics():
    game = BreakerBuilder(max_plies=8)
    model = PolicyValueNet((6, 5, 5), game.action_size, num_roles=2, role_heads=True)
    buffer = ReplayBuffer(capacity=16, seed=3)
    examples, _ = generate_selfplay_game(
        game=game,
        model=model,
        device="cpu",
        simulations=2,
        seed=4,
    )
    for example in examples:
        buffer.add(example)

    before = [parameter.detach().clone() for parameter in model.parameters()]

    metrics = train_steps(model, buffer, batch_size=2, steps=2, lr=1e-3, device="cpu")

    assert metrics["steps"] == 2
    assert np.isfinite(metrics["policy_loss"])
    assert np.isfinite(metrics["value_loss"])
    assert model.training is True
    assert any(
        not torch.allclose(old, new)
        for old, new in zip(before, model.parameters(), strict=True)
    )


@pytest.mark.parametrize(
    ("policy", "action_mask", "message"),
    [
        ([0.0, 0.0], [True, True], "positive probability mass"),
        ([0.5, 0.5], [True, False], "illegal actions"),
        ([1.0 - 1e-6, 1e-6], [True, False], "illegal actions"),
        ([-0.1, 1.1], [True, True], "negative probabilities"),
        ([-1e-9, 1.0 + 1e-9], [True, True], "negative probabilities"),
        ([float("nan"), 1.0], [True, True], "finite values"),
        ([0.4, 0.4], [True, True], "normalized"),
    ],
)
def test_train_steps_rejects_invalid_target_policies(policy, action_mask, message):
    with pytest.raises(ValueError, match=message):
        train_steps(
            _tiny_policy_value_model(),
            _train_buffer_with_policy(policy, action_mask),
            batch_size=1,
            steps=1,
            lr=1e-3,
            device="cpu",
        )


def test_evaluate_model_vs_random_returns_role_summary():
    game = BreakerBuilder(max_plies=8)
    model = PolicyValueNet((6, 5, 5), game.action_size, num_roles=2, role_heads=True)
    summary = evaluate_model_vs_random(
        game=game,
        model=model,
        device="cpu",
        games=4,
        simulations=2,
        seed=99,
    )
    assert summary["games"] == 4
    assert "role_win_rates" in summary
    assert "model_win_rate" in summary
    assert len(summary["outcomes"]) == 4
    assert (
        summary["model_win_rate"] + summary["random_win_rate"] + summary["draw_rate"]
        == pytest.approx(1.0, abs=0.002)
    )
    assert {outcome["model_player"] for outcome in summary["outcomes"]} == {0, 1}
    assert {outcome["model_role"] for outcome in summary["outcomes"]} == {0, 1}
    assert model.training is True


def test_evaluate_model_vs_random_preserves_eval_mode():
    game = BreakerBuilder(max_plies=8)
    model = PolicyValueNet((6, 5, 5), game.action_size, num_roles=2, role_heads=True)
    model.eval()

    evaluate_model_vs_random(
        game=game,
        model=model,
        device="cpu",
        games=2,
        simulations=2,
        seed=101,
    )

    assert model.training is False


def test_smoke_experiment_configs_are_valid_json():
    config_dir = Path("research/asymbench/experiments/configs")
    for name in ("micro_tafl_smoke.json", "breaker_builder_smoke.json"):
        data = json.loads((config_dir / name).read_text())
        assert data["iterations"] >= 1
        assert data["selfplay_games_per_iteration"] >= 1
        assert data["mcts_simulations"] >= 1
        assert data["model_variants"] == ["shared_heads", "role_heads"]


def test_role_head_runner_writes_smoke_outputs_and_schema(tmp_path):
    config = {
        "game": "breaker_builder",
        "device": "cuda",
        "seeds": [7],
        "model_variants": ["shared_heads", "role_heads"],
        "iterations": 1,
        "selfplay_games_per_iteration": 2,
        "train_steps_per_iteration": 1,
        "batch_size": 2,
        "replay_capacity": 64,
        "mcts_simulations": 1,
        "eval_games": 2,
        "eval_simulations": 1,
        "learning_rate": 0.001,
        "output_root": str(tmp_path / "runs"),
    }
    config_path = tmp_path / "tiny_role_heads.json"
    config_path.write_text(json.dumps(config))

    run_dir = run_experiment(config_path, device_override="cpu")

    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "role_summary.json").is_file()
    assert (run_dir / "config.json").is_file()

    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2
    required_fields = {
        "variant_seed",
        "trial_seed",
        "model_init_seed",
        "eval_seed",
        "selfplay_game_seeds",
        "selfplay_seat_role_counts",
        "train_total_loss",
        "eval_random_win_rate",
        "eval_model_role_win_rates",
        "eval_termination_reasons",
        "eval_games",
        "eval_simulations",
        "device_requested",
        "device_used",
    }
    for row in rows:
        assert required_fields.issubset(row)
        assert row["device_requested"] == "cpu"
        assert row["device_used"] == "cpu"
        assert row["eval_games"] == 2
        assert row["eval_simulations"] == 1
        assert row["trial_seed"] == 7
        assert row["variant_seed"] == row["model_init_seed"]
        assert row["selfplay_seat_role_counts"] == {"0-1": 1, "1-0": 1}

    by_variant = {row["variant"]: row for row in rows}
    assert (
        by_variant["shared_heads"]["eval_seed"]
        == by_variant["role_heads"]["eval_seed"]
    )
    assert (
        by_variant["shared_heads"]["selfplay_game_seeds"]
        == by_variant["role_heads"]["selfplay_game_seeds"]
    )
    assert (
        by_variant["shared_heads"]["selfplay_seat_role_counts"]
        == by_variant["role_heads"]["selfplay_seat_role_counts"]
    )
    assert (
        by_variant["shared_heads"]["model_init_seed"]
        != by_variant["role_heads"]["model_init_seed"]
    )

    summary = json.loads((run_dir / "role_summary.json").read_text())
    assert summary["device_requested"] == "cpu"
    assert summary["device_used"] == "cpu"
    for variant in ("shared_heads", "role_heads"):
        assert "final_eval_random_win_rate_mean" in summary["by_variant"][variant]
        assert "final_total_loss_mean" in summary["by_variant"][variant]
        assert "final_model_role_win_rates_mean" in summary["by_variant"][variant]
        assert "final_termination_reasons" in summary["by_variant"][variant]

    shared_checkpoint_path = run_dir / "shared_heads" / "seed_7" / "final_checkpoint.pt"
    role_checkpoint_path = run_dir / "role_heads" / "seed_7" / "final_checkpoint.pt"
    assert shared_checkpoint_path.is_file()
    assert role_checkpoint_path.is_file()

    shared_checkpoint = torch.load(shared_checkpoint_path, map_location="cpu")
    role_checkpoint = torch.load(role_checkpoint_path, map_location="cpu")
    assert shared_checkpoint["role_heads"] is False
    assert role_checkpoint["role_heads"] is True
    assert (
        shared_checkpoint["variant_seed"]
        == by_variant["shared_heads"]["variant_seed"]
    )
    assert (
        role_checkpoint["variant_seed"]
        == by_variant["role_heads"]["variant_seed"]
    )
    assert shared_checkpoint["variant_seed"] != role_checkpoint["variant_seed"]
    assert shared_checkpoint["trial_seed"] == role_checkpoint["trial_seed"] == 7
    assert shared_checkpoint["model_init_seed"] == shared_checkpoint["variant_seed"]
    assert role_checkpoint["model_init_seed"] == role_checkpoint["variant_seed"]
    assert all(
        tensor.device.type == "cpu"
        for tensor in shared_checkpoint["model_state_dict"].values()
    )
    assert all(
        tensor.device.type == "cpu"
        for tensor in role_checkpoint["model_state_dict"].values()
    )


def test_role_head_runner_variant_seed_does_not_depend_on_variant_order(tmp_path):
    base_config = {
        "game": "breaker_builder",
        "device": "cpu",
        "seeds": [11],
        "iterations": 1,
        "selfplay_games_per_iteration": 1,
        "train_steps_per_iteration": 1,
        "batch_size": 2,
        "replay_capacity": 64,
        "mcts_simulations": 1,
        "eval_games": 1,
        "eval_simulations": 1,
        "learning_rate": 0.001,
    }
    config_paths = []
    for name, variants in (
        ("forward", ["shared_heads", "role_heads"]),
        ("reversed", ["role_heads", "shared_heads"]),
    ):
        config = {
            **base_config,
            "model_variants": variants,
            "output_root": str(tmp_path / name),
        }
        config_path = tmp_path / f"{name}.json"
        config_path.write_text(json.dumps(config))
        config_paths.append(config_path)

    checkpoints_by_run = []
    rows_by_run = []
    for config_path in config_paths:
        run_dir = run_experiment(config_path, device_override="cpu")
        rows = [
            json.loads(line)
            for line in (run_dir / "metrics.jsonl").read_text().splitlines()
        ]
        rows_by_run.append({row["variant"]: row for row in rows})
        checkpoints_by_run.append(
            {
                variant: torch.load(
                    run_dir / variant / "seed_11" / "final_checkpoint.pt",
                    map_location="cpu",
                )
                for variant in ("shared_heads", "role_heads")
            }
        )

    for variant in ("shared_heads", "role_heads"):
        assert (
            rows_by_run[0][variant]["variant_seed"]
            == rows_by_run[1][variant]["variant_seed"]
        )
        assert (
            checkpoints_by_run[0][variant]["variant_seed"]
            == checkpoints_by_run[1][variant]["variant_seed"]
        )

        forward_state = checkpoints_by_run[0][variant]["model_state_dict"]
        reversed_state = checkpoints_by_run[1][variant]["model_state_dict"]
        assert forward_state.keys() == reversed_state.keys()
        for key in forward_state:
            assert torch.equal(forward_state[key], reversed_state[key]), (
                f"{variant} checkpoint tensor changed across variant order: {key}"
            )

    assert (
        rows_by_run[0]["shared_heads"]["variant_seed"]
        != rows_by_run[0]["role_heads"]["variant_seed"]
    )


def test_role_head_runner_accepts_generated_game_source(tmp_path):
    generator = EscapeCaptureGenerator()
    spec = generator.generate(
        seed=77,
        constraints=GenerationConstraints(
            board_sizes=((5, 5),),
            max_plies_range=(20, 20),
        ),
    )
    spec_path = tmp_path / "generated_spec.json"
    spec_path.write_text(json.dumps(spec.to_dict()))

    config = {
        "game_source": {"type": "generated_spec", "path": str(spec_path)},
        "device": "cpu",
        "seeds": [1],
        "model_variants": ["shared_heads", "role_heads"],
        "iterations": 1,
        "selfplay_games_per_iteration": 1,
        "train_steps_per_iteration": 1,
        "batch_size": 2,
        "replay_capacity": 64,
        "mcts_simulations": 1,
        "eval_games": 2,
        "eval_simulations": 1,
        "learning_rate": 0.001,
        "output_root": str(tmp_path / "runs"),
    }
    config_path = tmp_path / "generated_runner_config.json"
    config_path.write_text(json.dumps(config))

    run_dir = run_experiment(config_path, device_override="cpu")

    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
    ]
    assert {row["variant"] for row in rows} == {"shared_heads", "role_heads"}
    assert all(row["game"] == spec.name for row in rows)
    assert all(row["generated_family"] == "escape_capture" for row in rows)
    assert all(row["generated_name"] == spec.name for row in rows)
    assert all(row["generated_seed"] == 77 for row in rows)
    assert all(row["generated_spec_path"] == str(spec_path) for row in rows)

    written_config = json.loads((run_dir / "config.json").read_text())
    assert written_config["game_source"]["type"] == "generated_spec"

    summary = json.loads((run_dir / "role_summary.json").read_text())
    assert summary["generated_family"] == "escape_capture"
    assert summary["generated_name"] == spec.name
    assert summary["generated_seed"] == 77
    assert summary["generated_spec_path"] == str(spec_path)
