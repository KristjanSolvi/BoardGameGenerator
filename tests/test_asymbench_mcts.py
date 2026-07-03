import numpy as np

from research.asymbench.games.breaker_builder import BreakerBuilder
from research.asymbench.search.mcts import MCTSAgent, UniformEvaluator


def test_uniform_mcts_returns_legal_policy():
    game = BreakerBuilder(max_plies=12)
    state = game.initial_state()
    agent = MCTSAgent(evaluator=UniformEvaluator(), simulations=8, seed=5)
    policy = agent.policy(game, state, player=game.current_player(state))
    legal = set(game.legal_actions(state))
    assert policy.shape == (game.action_size,)
    assert np.isclose(policy.sum(), 1.0)
    assert {i for i, p in enumerate(policy) if p > 0.0}.issubset(legal)


def test_mcts_choose_returns_legal_action():
    game = BreakerBuilder(max_plies=12)
    state = game.initial_state()
    agent = MCTSAgent(evaluator=UniformEvaluator(), simulations=8, seed=7)
    action = agent.choose(game, state, player=game.current_player(state))
    assert action in game.legal_actions(state)


def test_mcts_is_reproducible_with_fixed_seed():
    game = BreakerBuilder(max_plies=12)
    state = game.initial_state()
    a = MCTSAgent(evaluator=UniformEvaluator(), simulations=8, seed=9).policy(
        game, state, 0
    )
    b = MCTSAgent(evaluator=UniformEvaluator(), simulations=8, seed=9).policy(
        game, state, 0
    )
    assert np.allclose(a, b)
