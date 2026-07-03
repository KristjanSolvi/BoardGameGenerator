import numpy as np
import pytest

from research.asymbench.games.base import RoleResult
from research.asymbench.games.breaker_builder import BreakerBuilder
from research.asymbench.search.mcts import MCTSAgent, UniformEvaluator


class FixedEvaluator:
    def __init__(self, prior, value=0.0):
        self.prior = prior
        self.value = value

    def evaluate(self, game, state, player):
        del game, state, player
        return np.asarray(self.prior, dtype=np.float64), self.value


class TinyImmediateResultGame:
    action_size = 3

    def initial_state(self):
        return "root"

    def current_player(self, state):
        return 0 if state == "root" else 1

    def legal_actions(self, state):
        if state == "root":
            return [0, 1, 2]
        return []

    def apply_action(self, state, action):
        assert state == "root"
        return action

    def is_terminal(self, state):
        return state != "root"

    def result(self, state):
        winners = {0: 0, 1: 1, 2: None}
        return RoleResult(winner=winners[state], reason="scripted", plies=1)


class TerminalWithLegalActionsGame:
    action_size = 1

    def current_player(self, state):
        del state
        return 0

    def legal_actions(self, state):
        del state
        return [0]

    def is_terminal(self, state):
        del state
        return True

    def result(self, state):
        del state
        return RoleResult(winner=None, reason="scripted", plies=0)


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


def test_mcts_rejects_zero_simulations():
    with pytest.raises(ValueError, match="simulations must be positive"):
        MCTSAgent(simulations=0)


def test_mcts_choose_rejects_no_legal_actions():
    game = BreakerBuilder(max_plies=1)
    state = game.initial_state()
    state = game.apply_action(state, game.legal_actions(state)[0])
    agent = MCTSAgent(evaluator=UniformEvaluator(), simulations=1, seed=1)

    with pytest.raises(ValueError, match="no legal actions"):
        agent.choose(game, state, player=game.current_player(state))


def test_mcts_choose_rejects_zero_policy_mass():
    game = TerminalWithLegalActionsGame()
    agent = MCTSAgent(evaluator=UniformEvaluator(), simulations=1, seed=1)

    with pytest.raises(ValueError, match="probability mass"):
        agent.choose(game, "terminal", player=game.current_player("terminal"))


def test_mcts_rejects_policy_for_non_current_player():
    game = BreakerBuilder(max_plies=12)
    state = game.initial_state()
    agent = MCTSAgent(evaluator=UniformEvaluator(), simulations=1, seed=1)

    with pytest.raises(ValueError, match="current player"):
        agent.policy(game, state, player=1)


def test_mcts_rejects_bad_evaluator_prior_shape():
    game = BreakerBuilder(max_plies=12)
    state = game.initial_state()
    agent = MCTSAgent(evaluator=FixedEvaluator([1.0, 0.0]), simulations=1, seed=1)

    with pytest.raises(ValueError, match="prior shape"):
        agent.policy(game, state, player=game.current_player(state))


@pytest.mark.parametrize("prior_value", [0.0, -1.0])
def test_mcts_falls_back_to_uniform_legal_priors(prior_value):
    game = BreakerBuilder(max_plies=12)
    state = game.initial_state()
    prior = np.full(game.action_size, prior_value, dtype=np.float64)
    agent = MCTSAgent(evaluator=FixedEvaluator(prior), simulations=1, seed=1)

    policy = agent.policy(game, state, player=game.current_player(state))

    legal = game.legal_actions(state)
    assert np.isclose(policy.sum(), 1.0)
    assert np.allclose(policy[legal], 1.0 / len(legal))
    assert {i for i, p in enumerate(policy) if p > 0.0} == set(legal)


def test_mcts_rejects_non_finite_evaluator_prior():
    game = BreakerBuilder(max_plies=12)
    state = game.initial_state()
    prior = np.zeros(game.action_size, dtype=np.float64)
    prior[game.legal_actions(state)[0]] = np.nan
    agent = MCTSAgent(evaluator=FixedEvaluator(prior), simulations=1, seed=1)

    with pytest.raises(ValueError, match="finite"):
        agent.policy(game, state, player=game.current_player(state))


def test_mcts_rejects_non_finite_evaluator_value():
    game = BreakerBuilder(max_plies=12)
    state = game.initial_state()
    agent = MCTSAgent(
        evaluator=FixedEvaluator(np.zeros(game.action_size), value=np.inf),
        simulations=1,
        seed=1,
    )

    with pytest.raises(ValueError, match="finite"):
        agent.policy(game, state, player=game.current_player(state))


def test_mcts_backup_sign_prefers_immediate_win():
    game = TinyImmediateResultGame()
    state = game.initial_state()
    agent = MCTSAgent(evaluator=UniformEvaluator(), simulations=20, seed=2)

    action = agent.choose(game, state, player=game.current_player(state))

    assert action == 0
