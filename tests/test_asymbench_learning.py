import numpy as np
import pytest

torch = pytest.importorskip("torch")

from research.asymbench.games.breaker_builder import BreakerBuilder
from research.asymbench.learning.model import PolicyValueNet
from research.asymbench.learning.replay import ReplayBuffer, TrainingExample
from research.asymbench.learning.selfplay import NeuralEvaluator, generate_selfplay_game


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
