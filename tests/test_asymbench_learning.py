import numpy as np
import pytest

torch = pytest.importorskip("torch")

from research.asymbench.games.breaker_builder import BreakerBuilder
from research.asymbench.learning.model import PolicyValueNet
from research.asymbench.learning.replay import ReplayBuffer, TrainingExample


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
