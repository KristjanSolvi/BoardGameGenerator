# AsymBench Role-Head AlphaZero-Lite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a GPU-local AlphaZero-lite research harness that tests whether role-specific policy/value heads learn asymmetric board games better than shared heads.

**Architecture:** Keep research code isolated under `research/asymbench/` so normal `gamegen` remains dependency-light. Implement two hand-written asymmetric games behind a shared integer-action interface, add role-aware random/MCTS baselines, then add PyTorch model, replay, self-play, training, evaluation, and analysis scripts.

**Tech Stack:** Python standard library, NumPy, PyTorch, pytest, optional matplotlib for plots. Existing `gamegen` code remains unchanged except for adding ignored research outputs and optional research dependency files.

---

## File Structure

Create:

- `requirements-research.txt`: optional research dependencies.
- `research/__init__.py`: package marker.
- `research/asymbench/__init__.py`: research package marker and public exports.
- `research/asymbench/games/__init__.py`: game registry exports.
- `research/asymbench/games/base.py`: shared `AsymGame` protocol, immutable state helpers, result helpers, action-mask helpers.
- `research/asymbench/games/micro_tafl.py`: small tafl-like attacker/defender game.
- `research/asymbench/games/breaker_builder.py`: builder/breaker game with asymmetric action types.
- `research/asymbench/baselines.py`: random agent, rollout runner, role-aware evaluation summaries.
- `research/asymbench/search/__init__.py`: MCTS exports.
- `research/asymbench/search/mcts.py`: PUCT-lite MCTS with uniform or neural evaluator.
- `research/asymbench/learning/__init__.py`: learning exports.
- `research/asymbench/learning/model.py`: PyTorch shared-head and role-head policy/value network.
- `research/asymbench/learning/replay.py`: replay buffer and training example dataclass.
- `research/asymbench/learning/selfplay.py`: MCTS self-play data generation.
- `research/asymbench/learning/train.py`: training loop and checkpoint helpers.
- `research/asymbench/learning/evaluate.py`: checkpoint and baseline evaluation.
- `research/asymbench/experiments/__init__.py`: experiment package marker.
- `research/asymbench/experiments/run_role_heads.py`: CLI experiment runner.
- `research/asymbench/experiments/configs/micro_tafl_smoke.json`: small smoke config.
- `research/asymbench/experiments/configs/breaker_builder_smoke.json`: small smoke config.
- `research/asymbench/analysis/__init__.py`: analysis package marker.
- `research/asymbench/analysis/summarize.py`: JSONL metrics summarizer.
- `tests/test_asymbench_games.py`: game API and mechanics tests.
- `tests/test_asymbench_baselines.py`: random/role-aware baseline tests.
- `tests/test_asymbench_mcts.py`: MCTS legality and determinism tests.
- `tests/test_asymbench_learning.py`: model/replay/self-play/training smoke tests.

Modify:

- `.gitignore`: add `research_runs/`, `*.pt`, and `*.pth`.

Do not modify:

- `gamegen/schema.py`
- `gamegen/validator.py`
- `gamegen/playtest.py`
- LLM prompts

Those belong to the later asymmetric generator milestone, after the RL proof point exists.

---

### Task 1: Research Package Skeleton And Optional Dependencies

**Files:**
- Create: `requirements-research.txt`
- Create: `research/__init__.py`
- Create: `research/asymbench/__init__.py`
- Create: `research/asymbench/games/__init__.py`
- Create: `research/asymbench/search/__init__.py`
- Create: `research/asymbench/learning/__init__.py`
- Create: `research/asymbench/experiments/__init__.py`
- Create: `research/asymbench/analysis/__init__.py`
- Modify: `.gitignore`
- Test: existing `pytest` suite

- [ ] **Step 1: Write package markers and dependency file**

Create `requirements-research.txt`:

```text
numpy
torch
matplotlib
```

Create `research/__init__.py`:

```python
"""Research-only code. Not required for normal gamegen operation."""
```

Create `research/asymbench/__init__.py`:

```python
"""AsymBench research harness for asymmetric board-game learning."""
```

Create `research/asymbench/games/__init__.py`:

```python
"""Reference asymmetric games used by the AsymBench research harness."""
```

Create `research/asymbench/search/__init__.py`:

```python
"""Search agents and MCTS utilities for AsymBench."""
```

Create `research/asymbench/learning/__init__.py`:

```python
"""PyTorch learning utilities for AsymBench."""
```

Create `research/asymbench/experiments/__init__.py`:

```python
"""Runnable experiment entry points for AsymBench."""
```

Create `research/asymbench/analysis/__init__.py`:

```python
"""Analysis utilities for AsymBench experiment outputs."""
```

- [ ] **Step 2: Ignore research outputs and checkpoints**

Append these lines to `.gitignore`, preserving existing content:

```gitignore

# AsymBench research outputs
research_runs/
*.pt
*.pth
```

- [ ] **Step 3: Run baseline tests**

Run:

```powershell
$env:PYTHONPATH='.'
pytest -q
```

Expected: all existing tests pass, currently `27 passed`.

- [ ] **Step 4: Commit skeleton**

Run:

```powershell
git add requirements-research.txt research .gitignore
git commit -m "Add AsymBench research package skeleton"
```

---

### Task 2: Shared Asymmetric Game Interface

**Files:**
- Create: `research/asymbench/games/base.py`
- Test: `tests/test_asymbench_games.py`

- [ ] **Step 1: Write failing interface tests**

Create the first version of `tests/test_asymbench_games.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
$env:PYTHONPATH='.'
pytest -q tests/test_asymbench_games.py
```

Expected: import failure because `research.asymbench.games.base` does not exist.

- [ ] **Step 3: Implement `games/base.py`**

Create `research/asymbench/games/base.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class IllegalActionError(ValueError):
    """Raised when an action id is not legal in a state."""


@dataclass(frozen=True)
class RoleResult:
    winner: int | None
    reason: str
    plies: int

    def value_for_player(self, player: int) -> float:
        if self.winner is None:
            return 0.0
        return 1.0 if self.winner == player else -1.0


class AsymGame(Protocol):
    name: str
    roles: tuple[str, str]
    board_shape: tuple[int, int]
    action_size: int

    def initial_state(self, seat_roles: tuple[int, int] = (0, 1)): ...
    def current_player(self, state) -> int: ...
    def player_role(self, state, player: int) -> int: ...
    def legal_actions(self, state) -> list[int]: ...
    def apply_action(self, state, action: int): ...
    def is_terminal(self, state) -> bool: ...
    def result(self, state) -> RoleResult: ...
    def observation_tensor(self, state, player: int) -> np.ndarray: ...
    def action_mask(self, state) -> np.ndarray: ...
    def render(self, state) -> str: ...


def make_action_mask(action_size: int, legal_actions: list[int]) -> np.ndarray:
    mask = np.zeros(action_size, dtype=np.bool_)
    for action in legal_actions:
        if action < 0 or action >= action_size:
            raise ValueError(
                f"legal action {action} outside action space 0..{action_size - 1}"
            )
        mask[action] = True
    return mask
```

- [ ] **Step 4: Verify interface tests pass**

Run:

```powershell
$env:PYTHONPATH='.'
pytest -q tests/test_asymbench_games.py
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit interface**

Run:

```powershell
git add research/asymbench/games/base.py tests/test_asymbench_games.py
git commit -m "Add AsymBench game interface"
```

---

### Task 3: MicroTafl Reference Game

**Files:**
- Create: `research/asymbench/games/micro_tafl.py`
- Modify: `research/asymbench/games/__init__.py`
- Modify: `tests/test_asymbench_games.py`

- [ ] **Step 1: Add failing MicroTafl tests**

Append to `tests/test_asymbench_games.py`:

```python
from research.asymbench.games.base import IllegalActionError
from research.asymbench.games.micro_tafl import MicroTafl


def test_micro_tafl_initial_state_roles_and_legal_moves():
    game = MicroTafl()
    state = game.initial_state()
    assert game.current_player(state) == 0
    assert game.player_role(state, 0) == MicroTafl.ATTACKER
    assert game.player_role(state, 1) == MicroTafl.DEFENDER
    assert not game.is_terminal(state)
    assert len(game.legal_actions(state)) > 0
    assert game.action_mask(state).sum() == len(game.legal_actions(state))


def test_micro_tafl_seat_role_swap_makes_player_zero_defender():
    game = MicroTafl()
    state = game.initial_state(seat_roles=(MicroTafl.DEFENDER, MicroTafl.ATTACKER))
    assert game.current_player(state) == 0
    assert game.player_role(state, 0) == MicroTafl.DEFENDER
    assert game.player_role(state, 1) == MicroTafl.ATTACKER


def test_micro_tafl_observation_shape_and_action_mask():
    game = MicroTafl()
    state = game.initial_state()
    obs = game.observation_tensor(state, player=0)
    assert obs.shape == (7, 5, 5)
    assert obs.dtype == np.float32
    assert game.action_mask(state).shape == (game.action_size,)


def test_micro_tafl_known_escape_sequence_reaches_defender_win():
    game = MicroTafl()
    state = game.initial_state(seat_roles=(MicroTafl.DEFENDER, MicroTafl.ATTACKER))
    moves = [
        game.encode_slide("c3", "c2"),
        game.encode_slide("a3", "a2"),
        game.encode_slide("c2", "b2"),
        game.encode_slide("a2", "a1"),
        game.encode_slide("b2", "a2"),
        game.encode_slide("e3", "e2"),
        game.encode_slide("a2", "a1"),
    ]
    for action in moves:
        assert action in game.legal_actions(state), game.render(state)
        state = game.apply_action(state, action)
    assert game.is_terminal(state)
    assert game.result(state).winner == 0
    assert game.result(state).reason == "king_escape"


def test_micro_tafl_illegal_action_rejected():
    game = MicroTafl()
    state = game.initial_state()
    illegal = game.encode_slide("c3", "c4")
    assert illegal not in game.legal_actions(state)
    try:
        game.apply_action(state, illegal)
    except IllegalActionError:
        pass
    else:
        raise AssertionError("expected IllegalActionError")
```

- [ ] **Step 2: Run tests to verify MicroTafl import fails**

Run:

```powershell
$env:PYTHONPATH='.'
pytest -q tests/test_asymbench_games.py
```

Expected: import failure for `micro_tafl`.

- [ ] **Step 3: Implement MicroTafl mechanics**

Create `research/asymbench/games/micro_tafl.py` with these exact mechanics:

- Board: 5x5, cells `a1` through `e5`.
- Roles: `ATTACKER = 0`, `DEFENDER = 1`.
- Action encoding: `from_index * 25 + to_index`, so `action_size = 625`.
- Initial role pieces:
  - attackers: `a3`, `b1`, `b5`, `c1`, `c5`, `d1`, `d5`, `e3`
  - defenders: guards at `b3`, `c2`, `c4`, `d3`
  - king: `c3`
- Any piece slides orthogonally through empty cells.
- Defender wins if the king reaches any corner.
- Attackers win if the king is captured by custodial sandwich.
- Non-king pieces are captured when the moved piece sandwiches an adjacent enemy between itself and an allied piece, the king, or a hostile corner.
- The king is captured if after an attacker move it has attackers or hostile corners on two opposite orthogonal sides.
- Draw at `max_plies`, default 80.
- If the current player has no legal action, the opponent wins.

State representation:

```python
@dataclass(frozen=True)
class MicroTaflState:
    board: tuple[int, ...]  # 0 empty, 1 attacker, 2 defender guard, 3 king
    to_move: int
    seat_roles: tuple[int, int]
    plies: int
```

Observation planes, shape `(7, 5, 5)`:

1. attackers
2. defender guards
3. king
4. own pieces
5. enemy pieces
6. current-player-to-move plane
7. acting-player role id plane, filled with 0 for attacker and 1 for defender

- [ ] **Step 4: Export MicroTafl**

Modify `research/asymbench/games/__init__.py`:

```python
"""Reference asymmetric games used by the AsymBench research harness."""

from research.asymbench.games.micro_tafl import MicroTafl

__all__ = ["MicroTafl"]
```

- [ ] **Step 5: Verify MicroTafl tests pass**

Run:

```powershell
$env:PYTHONPATH='.'
pytest -q tests/test_asymbench_games.py -k micro_tafl
```

Expected: all MicroTafl tests pass.

- [ ] **Step 6: Run all game tests**

Run:

```powershell
$env:PYTHONPATH='.'
pytest -q tests/test_asymbench_games.py
```

Expected: all game tests pass.

- [ ] **Step 7: Commit MicroTafl**

Run:

```powershell
git add research/asymbench/games tests/test_asymbench_games.py
git commit -m "Add MicroTafl asymmetric reference game"
```

---

### Task 4: BreakerBuilder Reference Game

**Files:**
- Create: `research/asymbench/games/breaker_builder.py`
- Modify: `research/asymbench/games/__init__.py`
- Modify: `tests/test_asymbench_games.py`

- [ ] **Step 1: Add failing BreakerBuilder tests**

Append to `tests/test_asymbench_games.py`:

```python
from research.asymbench.games.breaker_builder import BreakerBuilder


def test_breaker_builder_initial_state_roles_and_actions():
    game = BreakerBuilder()
    state = game.initial_state()
    assert game.current_player(state) == 0
    assert game.player_role(state, 0) == BreakerBuilder.BUILDER
    assert game.player_role(state, 1) == BreakerBuilder.BREAKER
    assert len(game.legal_actions(state)) == 23
    assert game.action_mask(state).sum() == 23


def test_breaker_builder_observation_shape():
    game = BreakerBuilder()
    state = game.initial_state()
    obs = game.observation_tensor(state, player=0)
    assert obs.shape == (6, 5, 5)
    assert obs.dtype == np.float32


def test_breaker_builder_known_connection_win():
    game = BreakerBuilder()
    state = game.initial_state()
    moves = [
        game.encode_place("a3"),
        game.encode_move("b2", "b1"),
        game.encode_place("b3"),
        game.encode_move("d4", "d5"),
        game.encode_place("c3"),
        game.encode_move("b1", "a1"),
        game.encode_place("d3"),
        game.encode_move("d5", "e5"),
        game.encode_place("e3"),
    ]
    for action in moves:
        assert action in game.legal_actions(state), game.render(state)
        state = game.apply_action(state, action)
    assert game.is_terminal(state)
    assert game.result(state).winner == 0
    assert game.result(state).reason == "builder_connection"


def test_breaker_builder_breaker_can_remove_adjacent_marker():
    game = BreakerBuilder()
    state = game.initial_state()
    state = game.apply_action(state, game.encode_place("b3"))
    remove = game.encode_remove("b2", "b3")
    assert remove in game.legal_actions(state)
    state = game.apply_action(state, remove)
    assert "b3" not in game.builder_cells(state)


def test_breaker_builder_illegal_builder_place_on_blocker_rejected():
    game = BreakerBuilder()
    state = game.initial_state()
    action = game.encode_place("b2")
    assert action not in game.legal_actions(state)
    try:
        game.apply_action(state, action)
    except IllegalActionError:
        pass
    else:
        raise AssertionError("expected IllegalActionError")
```

- [ ] **Step 2: Run tests to verify BreakerBuilder import fails**

Run:

```powershell
$env:PYTHONPATH='.'
pytest -q tests/test_asymbench_games.py -k breaker_builder
```

Expected: import failure for `breaker_builder`.

- [ ] **Step 3: Implement BreakerBuilder mechanics**

Create `research/asymbench/games/breaker_builder.py` with these exact mechanics:

- Board: 5x5, cells `a1` through `e5`.
- Roles: `BUILDER = 0`, `BREAKER = 1`.
- Initial blockers: `b2`, `d4`.
- Builder action: place one marker on any empty cell.
- Breaker action:
  - move one blocker one orthogonal step to an empty cell; or
  - remove one orthogonally adjacent builder marker.
- Builder wins immediately if builder markers form an orthogonally connected path from west edge to east edge.
- Breaker wins if builder has not connected by `max_plies`, default 30.
- If current player has no legal action, opponent wins.

Action encoding:

```text
0..24       place at target cell
25..649     move blocker from cell to cell: 25 + from_index * 25 + to_index
650..1274   remove marker from adjacent cell: 650 + from_index * 25 + target_index
```

State representation:

```python
@dataclass(frozen=True)
class BreakerBuilderState:
    board: tuple[int, ...]  # 0 empty, 1 builder marker, 2 breaker blocker
    to_move: int
    seat_roles: tuple[int, int]
    plies: int
```

Observation planes, shape `(6, 5, 5)`:

1. builder markers
2. breaker blockers
3. own pieces
4. enemy pieces
5. current-player-to-move plane
6. acting-player role id plane, filled with 0 for builder and 1 for breaker

- [ ] **Step 4: Export BreakerBuilder**

Modify `research/asymbench/games/__init__.py`:

```python
"""Reference asymmetric games used by the AsymBench research harness."""

from research.asymbench.games.breaker_builder import BreakerBuilder
from research.asymbench.games.micro_tafl import MicroTafl

__all__ = ["BreakerBuilder", "MicroTafl"]
```

- [ ] **Step 5: Verify BreakerBuilder tests pass**

Run:

```powershell
$env:PYTHONPATH='.'
pytest -q tests/test_asymbench_games.py -k breaker_builder
```

Expected: all BreakerBuilder tests pass.

- [ ] **Step 6: Run all game tests**

Run:

```powershell
$env:PYTHONPATH='.'
pytest -q tests/test_asymbench_games.py
```

Expected: all game tests pass.

- [ ] **Step 7: Commit BreakerBuilder**

Run:

```powershell
git add research/asymbench/games tests/test_asymbench_games.py
git commit -m "Add BreakerBuilder asymmetric reference game"
```

---

### Task 5: Role-Aware Random Baselines

**Files:**
- Create: `research/asymbench/baselines.py`
- Create: `tests/test_asymbench_baselines.py`

- [ ] **Step 1: Write failing baseline tests**

Create `tests/test_asymbench_baselines.py`:

```python
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
        agent_factories={0: lambda seed: RandomAgent(seed), 1: lambda seed: RandomAgent(seed)},
        games=8,
        seed=11,
    )
    second = evaluate_matchup(
        game,
        agent_factories={0: lambda seed: RandomAgent(seed), 1: lambda seed: RandomAgent(seed)},
        games=8,
        seed=11,
    )
    assert first == second
    assert first["games"] == 8
    assert "role_win_rates" in first
    assert "first_player_win_rate" in first
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
$env:PYTHONPATH='.'
pytest -q tests/test_asymbench_baselines.py
```

Expected: import failure for `research.asymbench.baselines`.

- [ ] **Step 3: Implement baselines**

Create `research/asymbench/baselines.py` with:

```python
from __future__ import annotations

import random
from collections import Counter
from typing import Callable


class RandomAgent:
    name = "random"

    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def choose(self, game, state, player: int) -> int:
        return self.rng.choice(game.legal_actions(state))


def play_game(game, agents: dict[int, object], seed: int, seat_roles=(0, 1)) -> dict:
    del seed
    state = game.initial_state(seat_roles=seat_roles)
    while not game.is_terminal(state):
        player = game.current_player(state)
        action = agents[player].choose(game, state, player)
        state = game.apply_action(state, action)
    result = game.result(state)
    winner_role = None if result.winner is None else game.player_role(state, result.winner)
    return {
        "winner": result.winner,
        "winner_role": winner_role,
        "reason": result.reason,
        "plies": result.plies,
        "seat_roles": list(seat_roles),
    }


def evaluate_matchup(
    game,
    agent_factories: dict[int, Callable[[int], object]],
    games: int,
    seed: int,
    seat_roles_list=((0, 1), (1, 0)),
) -> dict:
    outcomes = []
    for i in range(games):
        seat_roles = seat_roles_list[i % len(seat_roles_list)]
        agents = {
            0: agent_factories[0](seed ^ (i * 7919) ^ 0xA0),
            1: agent_factories[1](seed ^ (i * 7919) ^ 0xB1),
        }
        outcomes.append(play_game(game, agents, seed=seed + i, seat_roles=seat_roles))

    winner_counts = Counter(g["winner"] for g in outcomes)
    role_counts = Counter(g["winner_role"] for g in outcomes)
    reason_counts = Counter(g["reason"] for g in outcomes)
    return {
        "games": games,
        "p0_win_rate": round(winner_counts[0] / games, 3),
        "p1_win_rate": round(winner_counts[1] / games, 3),
        "draw_rate": round(winner_counts[None] / games, 3),
        "first_player_win_rate": round(winner_counts[0] / games, 3),
        "role_win_rates": {
            str(role): round(role_counts[role] / games, 3)
            for role in range(len(game.roles))
        },
        "termination_reasons": dict(reason_counts),
        "avg_plies": round(sum(g["plies"] for g in outcomes) / games, 2),
        "outcomes": outcomes,
    }
```

- [ ] **Step 4: Verify baseline tests pass**

Run:

```powershell
$env:PYTHONPATH='.'
pytest -q tests/test_asymbench_baselines.py
```

Expected: all baseline tests pass.

- [ ] **Step 5: Run game and baseline tests**

Run:

```powershell
$env:PYTHONPATH='.'
pytest -q tests/test_asymbench_games.py tests/test_asymbench_baselines.py
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit baselines**

Run:

```powershell
git add research/asymbench/baselines.py tests/test_asymbench_baselines.py
git commit -m "Add role-aware random baselines"
```

---

### Task 6: PUCT-Lite MCTS

**Files:**
- Create: `research/asymbench/search/mcts.py`
- Modify: `research/asymbench/search/__init__.py`
- Create: `tests/test_asymbench_mcts.py`

- [ ] **Step 1: Write failing MCTS tests**

Create `tests/test_asymbench_mcts.py`:

```python
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
    a = MCTSAgent(evaluator=UniformEvaluator(), simulations=8, seed=9).policy(game, state, 0)
    b = MCTSAgent(evaluator=UniformEvaluator(), simulations=8, seed=9).policy(game, state, 0)
    assert np.allclose(a, b)
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
$env:PYTHONPATH='.'
pytest -q tests/test_asymbench_mcts.py
```

Expected: import failure for `research.asymbench.search.mcts`.

- [ ] **Step 3: Implement MCTS**

Create `research/asymbench/search/mcts.py` with:

- `UniformEvaluator.evaluate(game, state, player)` returning:
  - prior: uniform over legal actions;
  - value: `0.0`.
- `MCTSAgent.policy(game, state, player)` returning action probabilities from root visit counts.
- `MCTSAgent.choose(game, state, player)` sampling or argmaxing from policy. Use argmax for deterministic tests.
- PUCT score:

```text
Q + c_puct * prior * sqrt(parent_visits + 1) / (1 + child_visits)
```

- Backup values alternating sign because value is from current player perspective.
- Terminal leaf value from requested player's perspective:

```python
value = game.result(state).value_for_player(player_to_evaluate)
```

- [ ] **Step 4: Export MCTS**

Modify `research/asymbench/search/__init__.py`:

```python
"""Search agents and MCTS utilities for AsymBench."""

from research.asymbench.search.mcts import MCTSAgent, UniformEvaluator

__all__ = ["MCTSAgent", "UniformEvaluator"]
```

- [ ] **Step 5: Verify MCTS tests pass**

Run:

```powershell
$env:PYTHONPATH='.'
pytest -q tests/test_asymbench_mcts.py
```

Expected: all MCTS tests pass.

- [ ] **Step 6: Run all research tests so far**

Run:

```powershell
$env:PYTHONPATH='.'
pytest -q tests/test_asymbench_games.py tests/test_asymbench_baselines.py tests/test_asymbench_mcts.py
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit MCTS**

Run:

```powershell
git add research/asymbench/search tests/test_asymbench_mcts.py
git commit -m "Add PUCT-lite MCTS for AsymBench"
```

---

### Task 7: PyTorch Model And Replay Buffer

**Files:**
- Create: `research/asymbench/learning/model.py`
- Create: `research/asymbench/learning/replay.py`
- Modify: `research/asymbench/learning/__init__.py`
- Create: `tests/test_asymbench_learning.py`

- [ ] **Step 1: Write failing model and replay tests**

Create `tests/test_asymbench_learning.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
$env:PYTHONPATH='.'
pytest -q tests/test_asymbench_learning.py
```

Expected: import failure for learning modules, unless PyTorch is missing. If PyTorch is missing, install research dependencies first with `pip install -r requirements-research.txt`.

- [ ] **Step 3: Implement model**

Create `research/asymbench/learning/model.py` with:

- `PolicyValueNet(input_shape, action_size, num_roles=2, role_heads=False, channels=64)`
- small CNN trunk:
  - Conv2d input channels -> channels, kernel 3, padding 1
  - ReLU
  - Conv2d channels -> channels, kernel 3, padding 1
  - ReLU
  - flatten
  - Linear to 128
  - ReLU
- shared or role-specific policy/value heads.
- Mask illegal actions by filling logits with `-1e9`.
- `tanh` value output.

- [ ] **Step 4: Implement replay buffer**

Create `research/asymbench/learning/replay.py` with:

```python
@dataclass(frozen=True)
class TrainingExample:
    observation: np.ndarray
    role: int
    action_mask: np.ndarray
    policy: np.ndarray
    value: float
```

and `ReplayBuffer(capacity, seed)` using `collections.deque(maxlen=capacity)` plus deterministic `random.Random(seed).sample`.

- [ ] **Step 5: Export learning utilities**

Modify `research/asymbench/learning/__init__.py`:

```python
"""PyTorch learning utilities for AsymBench."""

from research.asymbench.learning.model import PolicyValueNet
from research.asymbench.learning.replay import ReplayBuffer, TrainingExample

__all__ = ["PolicyValueNet", "ReplayBuffer", "TrainingExample"]
```

- [ ] **Step 6: Verify learning model tests pass**

Run:

```powershell
$env:PYTHONPATH='.'
pytest -q tests/test_asymbench_learning.py
```

Expected: all current learning tests pass.

- [ ] **Step 7: Commit model and replay**

Run:

```powershell
git add research/asymbench/learning tests/test_asymbench_learning.py
git commit -m "Add AsymBench policy-value model and replay buffer"
```

---

### Task 8: Neural Evaluator And MCTS Self-Play Examples

**Files:**
- Create: `research/asymbench/learning/selfplay.py`
- Modify: `research/asymbench/learning/__init__.py`
- Modify: `tests/test_asymbench_learning.py`

- [ ] **Step 1: Add failing self-play tests**

Append to `tests/test_asymbench_learning.py`:

```python
from research.asymbench.learning.selfplay import NeuralEvaluator, generate_selfplay_game


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
    assert all(ex.policy.shape == (game.action_size,) for ex in examples)
    assert all(-1.0 <= ex.value <= 1.0 for ex in examples)
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
$env:PYTHONPATH='.'
pytest -q tests/test_asymbench_learning.py -k "neural_evaluator or selfplay"
```

Expected: import failure for `selfplay`.

- [ ] **Step 3: Implement neural evaluator**

Create `research/asymbench/learning/selfplay.py` with `NeuralEvaluator`:

- accepts model and device;
- converts `game.observation_tensor(state, player)` to batch tensor;
- converts role to tensor using `game.player_role(state, player)`;
- converts mask to tensor using `game.action_mask(state)`;
- applies model under `torch.no_grad()`;
- softmaxes logits over legal actions;
- returns NumPy prior and scalar value.

- [ ] **Step 4: Implement `generate_selfplay_game`**

Use `MCTSAgent(NeuralEvaluator(...), simulations, seed)` at each ply.

For each ply store:

- observation for acting player;
- role id for acting player;
- action mask;
- MCTS policy;
- acting player.

At terminal state:

- get `RoleResult`;
- convert each pending record into `TrainingExample` with `value=result.value_for_player(acting_player)`.

Return:

```python
examples, {
    "winner": result.winner,
    "reason": result.reason,
    "plies": result.plies,
}
```

- [ ] **Step 5: Export self-play utilities**

Modify `research/asymbench/learning/__init__.py`:

```python
"""PyTorch learning utilities for AsymBench."""

from research.asymbench.learning.model import PolicyValueNet
from research.asymbench.learning.replay import ReplayBuffer, TrainingExample
from research.asymbench.learning.selfplay import NeuralEvaluator, generate_selfplay_game

__all__ = [
    "NeuralEvaluator",
    "PolicyValueNet",
    "ReplayBuffer",
    "TrainingExample",
    "generate_selfplay_game",
]
```

- [ ] **Step 6: Verify self-play tests pass**

Run:

```powershell
$env:PYTHONPATH='.'
pytest -q tests/test_asymbench_learning.py -k "neural_evaluator or selfplay"
```

Expected: self-play tests pass.

- [ ] **Step 7: Commit self-play**

Run:

```powershell
git add research/asymbench/learning tests/test_asymbench_learning.py
git commit -m "Add neural MCTS self-play generation"
```

---

### Task 9: Training And Evaluation Loops

**Files:**
- Create: `research/asymbench/learning/train.py`
- Create: `research/asymbench/learning/evaluate.py`
- Modify: `research/asymbench/learning/__init__.py`
- Modify: `tests/test_asymbench_learning.py`

- [ ] **Step 1: Add failing training tests**

Append to `tests/test_asymbench_learning.py`:

```python
from research.asymbench.learning.train import train_steps
from research.asymbench.learning.evaluate import evaluate_model_vs_random


def test_train_steps_updates_model_and_returns_metrics():
    game = BreakerBuilder(max_plies=8)
    model = PolicyValueNet((6, 5, 5), game.action_size, num_roles=2, role_heads=True)
    buffer = ReplayBuffer(capacity=16, seed=3)
    examples, _ = generate_selfplay_game(game, model, device="cpu", simulations=2, seed=4)
    for example in examples:
        buffer.add(example)
    metrics = train_steps(model, buffer, batch_size=2, steps=2, lr=1e-3, device="cpu")
    assert metrics["steps"] == 2
    assert np.isfinite(metrics["policy_loss"])
    assert np.isfinite(metrics["value_loss"])


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
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
$env:PYTHONPATH='.'
pytest -q tests/test_asymbench_learning.py -k "train_steps or evaluate_model"
```

Expected: import failure for `train` and `evaluate`.

- [ ] **Step 3: Implement `train_steps`**

Create `research/asymbench/learning/train.py`:

- sample batches from `ReplayBuffer`;
- stack observations, roles, action masks, policies, values;
- compute masked policy cross-entropy:

```python
log_probs = torch.log_softmax(logits, dim=1)
policy_loss = -(target_policy * log_probs).sum(dim=1).mean()
```

- compute MSE value loss;
- optimize with Adam;
- return average losses.

- [ ] **Step 4: Implement model evaluation**

Create `research/asymbench/learning/evaluate.py`:

- `ModelAgent` wraps `MCTSAgent(NeuralEvaluator(model))`.
- `evaluate_model_vs_random` alternates:
  - model as player 0, random as player 1;
  - random as player 0, model as player 1;
  - seat roles `(0, 1)` and `(1, 0)` across games.
- return role-aware summary including model win rate.

- [ ] **Step 5: Export training/evaluation**

Modify `research/asymbench/learning/__init__.py` to also export:

```python
from research.asymbench.learning.evaluate import evaluate_model_vs_random
from research.asymbench.learning.train import train_steps
```

- [ ] **Step 6: Verify learning tests pass**

Run:

```powershell
$env:PYTHONPATH='.'
pytest -q tests/test_asymbench_learning.py
```

Expected: all learning tests pass.

- [ ] **Step 7: Commit training/evaluation**

Run:

```powershell
git add research/asymbench/learning tests/test_asymbench_learning.py
git commit -m "Add AsymBench training and evaluation loops"
```

---

### Task 10: Role-Head Experiment Runner And Smoke Configs

**Files:**
- Create: `research/asymbench/experiments/run_role_heads.py`
- Create: `research/asymbench/experiments/configs/micro_tafl_smoke.json`
- Create: `research/asymbench/experiments/configs/breaker_builder_smoke.json`
- Test: `tests/test_asymbench_learning.py`

- [ ] **Step 1: Add failing config test**

Append to `tests/test_asymbench_learning.py`:

```python
import json
from pathlib import Path


def test_smoke_experiment_configs_are_valid_json():
    config_dir = Path("research/asymbench/experiments/configs")
    for name in ("micro_tafl_smoke.json", "breaker_builder_smoke.json"):
        data = json.loads((config_dir / name).read_text())
        assert data["iterations"] >= 1
        assert data["selfplay_games_per_iteration"] >= 1
        assert data["mcts_simulations"] >= 1
        assert data["model_variants"] == ["shared_heads", "role_heads"]
```

- [ ] **Step 2: Run test to verify config files are missing**

Run:

```powershell
$env:PYTHONPATH='.'
pytest -q tests/test_asymbench_learning.py -k smoke_experiment_configs
```

Expected: file-not-found failure.

- [ ] **Step 3: Add smoke configs**

Create `research/asymbench/experiments/configs/micro_tafl_smoke.json`:

```json
{
  "game": "micro_tafl",
  "device": "cuda",
  "seeds": [1],
  "model_variants": ["shared_heads", "role_heads"],
  "iterations": 2,
  "selfplay_games_per_iteration": 2,
  "train_steps_per_iteration": 4,
  "batch_size": 8,
  "replay_capacity": 512,
  "mcts_simulations": 8,
  "eval_games": 4,
  "eval_simulations": 4,
  "learning_rate": 0.001,
  "output_root": "research_runs/asymbench"
}
```

Create `research/asymbench/experiments/configs/breaker_builder_smoke.json`:

```json
{
  "game": "breaker_builder",
  "device": "cuda",
  "seeds": [1],
  "model_variants": ["shared_heads", "role_heads"],
  "iterations": 2,
  "selfplay_games_per_iteration": 2,
  "train_steps_per_iteration": 4,
  "batch_size": 8,
  "replay_capacity": 512,
  "mcts_simulations": 8,
  "eval_games": 4,
  "eval_simulations": 4,
  "learning_rate": 0.001,
  "output_root": "research_runs/asymbench"
}
```

- [ ] **Step 4: Implement experiment runner**

Create `research/asymbench/experiments/run_role_heads.py`:

- CLI:

```powershell
python -m research.asymbench.experiments.run_role_heads --config research/asymbench/experiments/configs/breaker_builder_smoke.json
```

- load config JSON;
- map game names to classes;
- if config device is `cuda` but `torch.cuda.is_available()` is false, use CPU and record `"device_requested": "cuda", "device_used": "cpu"`;
- for each seed and model variant:
  - create model;
  - create replay buffer;
  - generate self-play games;
  - train for configured steps;
  - evaluate vs random;
  - append one JSON object per iteration to `metrics.jsonl`;
  - save final checkpoint as `.pt`;
- write `config.json` and `role_summary.json`.

Metric row schema:

```json
{
  "iteration": 1,
  "game": "breaker_builder",
  "variant": "role_heads",
  "seed": 1,
  "device_used": "cuda",
  "selfplay_games_total": 2,
  "policy_loss": 0.0,
  "value_loss": 0.0,
  "eval_model_win_rate": 0.0,
  "eval_role_win_rates": {"0": 0.0, "1": 0.0},
  "eval_draw_rate": 0.0,
  "eval_avg_plies": 0.0
}
```

- [ ] **Step 5: Verify config test passes**

Run:

```powershell
$env:PYTHONPATH='.'
pytest -q tests/test_asymbench_learning.py -k smoke_experiment_configs
```

Expected: config test passes.

- [ ] **Step 6: Run CPU smoke experiment**

Temporarily create a PowerShell-local CPU override by editing the command argument is not supported, so copy the config to a temp file outside git or add a CLI `--device cpu` override in `run_role_heads.py`.

Run:

```powershell
$env:PYTHONPATH='.'
python -m research.asymbench.experiments.run_role_heads --config research/asymbench/experiments/configs/breaker_builder_smoke.json --device cpu
```

Expected:

- command exits 0;
- writes a run directory under `research_runs/asymbench/`;
- writes `metrics.jsonl`;
- writes `role_summary.json`.

- [ ] **Step 7: Commit experiment runner**

Run:

```powershell
git add research/asymbench/experiments tests/test_asymbench_learning.py
git commit -m "Add AsymBench role-head experiment runner"
```

---

### Task 11: Analysis Summarizer

**Files:**
- Create: `research/asymbench/analysis/summarize.py`
- Create: `tests/test_asymbench_analysis.py`

- [ ] **Step 1: Write failing summarizer test**

Create `tests/test_asymbench_analysis.py`:

```python
import json
from pathlib import Path

from research.asymbench.analysis.summarize import summarize_metrics


def test_summarize_metrics_groups_by_variant(tmp_path: Path):
    metrics = tmp_path / "metrics.jsonl"
    rows = [
        {"variant": "shared_heads", "eval_model_win_rate": 0.25, "eval_avg_plies": 8},
        {"variant": "shared_heads", "eval_model_win_rate": 0.75, "eval_avg_plies": 10},
        {"variant": "role_heads", "eval_model_win_rate": 1.0, "eval_avg_plies": 7},
    ]
    metrics.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    summary = summarize_metrics(metrics)
    assert summary["shared_heads"]["mean_model_win_rate"] == 0.5
    assert summary["shared_heads"]["mean_avg_plies"] == 9.0
    assert summary["role_heads"]["mean_model_win_rate"] == 1.0
```

- [ ] **Step 2: Run test to verify failure**

Run:

```powershell
$env:PYTHONPATH='.'
pytest -q tests/test_asymbench_analysis.py
```

Expected: import failure for `analysis.summarize`.

- [ ] **Step 3: Implement summarizer**

Create `research/asymbench/analysis/summarize.py`:

- `summarize_metrics(path: Path) -> dict`
- group rows by `variant`;
- compute:
  - `rows`;
  - `mean_model_win_rate`;
  - `mean_avg_plies`;
  - `last_model_win_rate`;
- CLI:

```powershell
python -m research.asymbench.analysis.summarize research_runs/asymbench/<run>/metrics.jsonl
```

- print JSON summary to stdout.

- [ ] **Step 4: Verify summarizer test passes**

Run:

```powershell
$env:PYTHONPATH='.'
pytest -q tests/test_asymbench_analysis.py
```

Expected: summarizer test passes.

- [ ] **Step 5: Commit analysis**

Run:

```powershell
git add research/asymbench/analysis tests/test_asymbench_analysis.py
git commit -m "Add AsymBench metrics summarizer"
```

---

### Task 12: Full Verification And GPU Smoke Run

**Files:**
- Modify only if failures reveal defects in previous files.

- [ ] **Step 1: Verify all existing and research tests**

Run:

```powershell
$env:PYTHONPATH='.'
pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Verify PyTorch CUDA availability**

Run:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Expected on target machine: CUDA available and device name includes `4080`. If CUDA is unavailable, record the exact output in the final report and run CPU smoke only.

- [ ] **Step 3: Run MicroTafl GPU smoke**

Run:

```powershell
$env:PYTHONPATH='.'
python -m research.asymbench.experiments.run_role_heads --config research/asymbench/experiments/configs/micro_tafl_smoke.json
```

Expected: command exits 0 and writes metrics/checkpoints.

- [ ] **Step 4: Run BreakerBuilder GPU smoke**

Run:

```powershell
$env:PYTHONPATH='.'
python -m research.asymbench.experiments.run_role_heads --config research/asymbench/experiments/configs/breaker_builder_smoke.json
```

Expected: command exits 0 and writes metrics/checkpoints.

- [ ] **Step 5: Summarize both smoke runs**

Run the summarizer on each `metrics.jsonl` written by the two smoke commands:

```powershell
$env:PYTHONPATH='.'
python -m research.asymbench.analysis.summarize <path-to-metrics.jsonl>
```

Expected: JSON summary includes both `shared_heads` and `role_heads`.

- [ ] **Step 6: Commit fixes if any were required**

If Step 1-5 required code changes, commit them:

```powershell
git add research tests requirements-research.txt .gitignore
git commit -m "Verify AsymBench AlphaZero-lite smoke experiments"
```

If no code changes were required, do not create an empty commit.

---

## Final Verification Checklist

Before reporting completion:

- [ ] `git status -sb` shows only expected files or a clean tree.
- [ ] `$env:PYTHONPATH='.'; pytest -q` passes.
- [ ] At least one CPU smoke run has completed.
- [ ] GPU availability has been checked.
- [ ] If CUDA is available, both smoke configs have completed on GPU.
- [ ] `research_runs/` is ignored and not staged.
- [ ] Metrics JSONL and summary JSON prove both variants ran.
- [ ] Final report states whether this is only infrastructure or whether any early role-head signal was observed.

## Self-Review

Spec coverage:

- Two reference games: Tasks 3 and 4.
- Role-aware random/MCTS evaluation: Tasks 5 and 6.
- Shared-head and role-head neural agents: Task 7.
- GPU-backed self-play training: Tasks 8, 9, 10, and 12.
- Structured metrics: Tasks 10 and 11.
- Existing `gamegen` tests preserved: Tasks 1 and 12.

Type consistency:

- `RoleResult` is used by games, self-play, and baselines.
- Game action APIs consistently use integer action ids.
- `PolicyValueNet` receives `(obs, roles, mask)` in every task.
- `TrainingExample` fields match the training loop inputs.

Research controls:

- LLM game generation is excluded from this milestone.
- Role/head comparison uses paired variants under equal config.
- Smoke configs are tiny and are not treated as final evidence.
- Failed or weak results remain useful because they test feasibility.
