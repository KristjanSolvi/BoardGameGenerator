# AsymBench Role-Head AlphaZero-Lite Design

Date: 2026-07-03
Branch: `research/asymbench`

## Purpose

This branch tests the core research claim behind AsymBench before we extend the LLM game generator:

> Asymmetric board games create role-specific learning and evaluation effects that are hidden by symmetric game benchmarks.

The milestone is a tractable GPU-local experiment using hand-written asymmetric reference games and an AlphaZero-style learner. The decisive comparison is between:

- a shared-head model that uses one policy head and one value head for both roles;
- a role-head model that shares board features but uses separate policy and value heads per role.

If role-specific heads learn faster, avoid role interference, or produce more stable role-balanced play across several small asymmetric games, that supports a publishable AsymBench direction. It also gives the later LLM generator a concrete target: generate games that expose measurable role-conditioned reasoning and learning behavior.

## Non-Goals

This milestone will not:

- generate new asymmetric games with the LLM pipeline;
- add hidden information, stochasticity, negotiation, or social deduction;
- claim to solve game balancing better than RuleSmith;
- claim to simulate subjective human feedback better than MeepleLM;
- train a large AlphaZero system;
- require PyTorch for normal `gamegen` usage.

The first milestone is an experimental proof point, not the final benchmark.

## Research Questions

### RQ1: Role-Specific Learnability

Do role-specific policy/value heads improve self-play learning on asymmetric games compared with shared heads under equal compute?

Primary metrics:

- final evaluation win rate against a fixed opponent pool;
- area under the learning curve;
- value loss by role;
- policy entropy by role;
- time to reach a target evaluation score.

### RQ2: Role Interference

Does training on one role degrade or destabilize performance on the other role?

Primary metrics:

- attacker/defender evaluation curves;
- drop in role-specific win rate after updates dominated by the other role;
- variance of value predictions by role;
- checkpoint-vs-checkpoint role win matrix.

### RQ3: Balance Is Evaluator-Dependent

Do random, MCTS, and learned agents estimate role strength differently on the same games?

Primary metrics:

- random role win rate;
- MCTS role win rate;
- trained-agent role win rate;
- first-player advantage separated from role advantage where the game supports role/seat swaps.

### RQ4: Benchmark Feasibility

Can small asymmetric games be implemented, validated, trained, and analyzed reproducibly on local hardware?

Primary metrics:

- deterministic test pass rate;
- self-play throughput;
- training runtime on an RTX 4080;
- stability across seeds.

## Experimental Scope

### Game Class

First-generation games are:

- two-player;
- deterministic;
- perfect information;
- alternating turn;
- finite horizon through terminal rules or repetition/move caps;
- small-board, grid-based;
- discrete legal action lists;
- asymmetric in setup, pieces, objectives, or action space.

The target board size is 5x5 to 7x7. The target average game length is under 150 plies. The target legal branching factor is roughly 5-40.

### Reference Games

Implement at least two hand-written games before any LLM generation work.

#### Game 1: MicroTafl

Purpose: attacker-vs-defender asymmetry with unequal pieces and distinct objectives.

Sketch:

- 5x5 or 7x7 board.
- Defenders have a king plus guards.
- Attackers have more pieces.
- Defender wins by moving the king to an escape square.
- Attacker wins by capturing the king.
- All moves and captures are deterministic.

Why it matters:

- Closely connects to the Tablut AlphaZero result.
- Tests whether the dual-head result appears in a smaller, controlled setting.

#### Game 2: BreakerBuilder

Purpose: different action types and objectives, not just unequal piece counts.

Sketch:

- Builder places or advances markers to connect target regions.
- Breaker moves disruptors that freeze, remove, or block builder markers.
- Builder wins by completing a connection or structure.
- Breaker wins by preventing completion until a bounded turn limit or by reaching a disruption threshold.

Why it matters:

- Tests asymmetric action spaces.
- Less tied to a known historical game than tafl.

#### Optional Game 3: RaidHarvest

Purpose: asymmetric incentives and tempo.

Sketch:

- Harvester scores by collecting or transporting resources.
- Raider scores by intercepting, blocking, or capturing enough resources.
- Both roles use compact deterministic actions.

Why it matters:

- Adds an economy-like asymmetry without becoming CivMini-scale.

## Architecture

The research code lives under `research/asymbench/` and stays separate from production `gamegen`.

Proposed layout:

```text
requirements-research.txt
research/
  asymbench/
    __init__.py
    games/
      __init__.py
      base.py
      micro_tafl.py
      breaker_builder.py
    search/
      __init__.py
      mcts.py
    learning/
      __init__.py
      model.py
      replay.py
      selfplay.py
      train.py
      evaluate.py
    experiments/
      run_role_heads.py
      configs/
        micro_tafl_smoke.json
        breaker_builder_smoke.json
    analysis/
      summarize.py
tests/
  test_asymbench_games.py
  test_asymbench_mcts.py
  test_asymbench_learning.py
```

### Optional Dependencies

Normal generator usage must not require PyTorch.

Research dependencies go in `requirements-research.txt`, likely:

```text
torch
numpy
matplotlib
```

`matplotlib` is optional and only needed for plots. All core results must also be emitted as JSON or CSV so the experiment remains scriptable.

## Game Interface

Each reference game implements a small protocol independent from generated engine classes.

Required methods:

```python
class AsymGame:
    name: str
    roles: tuple[str, str]
    board_shape: tuple[int, int]
    action_size: int

    def initial_state(self, seat_roles: tuple[int, int] = (0, 1)) -> State: ...
    def current_player(self, state: State) -> int: ...
    def player_role(self, state: State, player: int) -> int: ...
    def legal_actions(self, state: State) -> list[int]: ...
    def apply_action(self, state: State, action: int) -> State: ...
    def is_terminal(self, state: State) -> bool: ...
    def result(self, state: State) -> dict: ...
    def observation_tensor(self, state: State, player: int) -> np.ndarray: ...
    def action_mask(self, state: State) -> np.ndarray: ...
    def render(self, state: State) -> str: ...
```

Design decisions:

- Actions are encoded as integer ids for neural training.
- Legal action masks prevent illegal policy targets.
- States are immutable and hashable where practical.
- Observations include role id and side-to-move information.
- Seat-role mapping is explicit so role advantage and first-player advantage can be separated.

## AlphaZero-Lite Learner

### Model

Use a small PyTorch network:

- board observation input;
- shared trunk;
- either shared policy/value heads or role-specific policy/value heads;
- legal-action masking at inference and loss time.

Two model variants:

```text
shared_heads:
  trunk -> policy_head
        -> value_head

role_heads:
  trunk -> policy_head_role_0
        -> policy_head_role_1
        -> value_head_role_0
        -> value_head_role_1
```

Keep the trunk small enough that experiments can run repeatedly:

- tiny CNN for grid games, or MLP if observations are compact;
- configurable hidden channels;
- no distributed training;
- no mixed precision until correctness is stable.

### MCTS

Use a compact PUCT-style MCTS:

- network priors over legal actions;
- network value from current player's perspective;
- fixed simulation budget per move;
- Dirichlet noise optional for self-play root exploration;
- temperature schedule optional and simple.

For smoke tests, MCTS can run with a tiny simulation count. For experiments, use a configurable count.

### Self-Play

Self-play loop:

1. Initialize game.
2. At each state, run MCTS.
3. Store observation, legal mask, role id, MCTS policy target, and current player.
4. Sample action from MCTS visit distribution.
5. At terminal state, assign value targets from each stored player's perspective.
6. Add examples to replay buffer.
7. Train on mini-batches.

### Evaluation

Evaluate checkpoints against:

- random;
- MCTS with uniform priors or untrained network;
- earlier checkpoints;
- the opposite model variant under equal simulation budget.

The key comparison must be paired:

- same game;
- same seed set;
- same number of self-play games;
- same MCTS simulations;
- same network size except for heads;
- same optimizer and learning schedule.

## Results Artifacts

Every run writes a timestamped directory under `research_runs/asymbench/`.

Artifacts:

```text
config.json
metrics.jsonl
eval_matrix.json
role_summary.json
checkpoints/
plots/
```

Metrics in `metrics.jsonl`:

- iteration;
- game;
- model variant;
- seed;
- self-play games generated;
- train policy loss;
- train value loss;
- entropy by role;
- evaluation win rate by role;
- draw rate;
- average game length;
- MCTS simulations per move;
- wall-clock time.

`research_runs/` must be ignored by git.

## Testing Strategy

### Unit Tests

Games:

- initial state is legal;
- both roles have legal actions;
- known short sequences produce expected board states;
- each terminal condition can be reached;
- illegal actions are rejected;
- action masks match legal actions;
- observation tensors have stable shape;
- role/seat mapping is correct.

MCTS:

- returns a valid policy over legal moves;
- never selects illegal actions;
- deterministic with fixed seed where expected;
- handles terminal states.

Learning:

- model forward pass shape checks;
- shared-head and role-head variants both run;
- loss decreases on a tiny synthetic batch or at least produces finite gradients;
- replay buffer stores and samples valid examples;
- one self-play smoke game can be generated.

### Baseline Verification

Existing `gamegen` tests must continue to pass with `PYTHONPATH=.`.

Research tests should be runnable separately:

```powershell
$env:PYTHONPATH='.'
pytest -q tests/test_asymbench_*.py
```

## Research Hygiene

To keep the work journal-worthy:

- do not tune the claim after looking at only one successful game;
- record failed games and failed runs;
- keep seed lists in configs;
- report confidence intervals or at least seed variance;
- compare against symmetric controls later;
- keep LLM generation out of the first RL experiment;
- define rejection criteria before generating future games;
- preserve raw metrics, not just plots.

## Acceptance Criteria For This Milestone

The milestone is complete when:

1. Two asymmetric reference games are implemented and unit-tested.
2. Role-aware random/MCTS evaluation works for both games.
3. Shared-head and role-head neural agents both train without crashing.
4. At least one GPU-backed smoke experiment runs end-to-end.
5. Results are written as structured metrics.
6. A short analysis script summarizes role-head versus shared-head performance.
7. Existing `gamegen` tests still pass.

The milestone is strong enough to continue toward a paper when:

1. The role-head model shows a consistent advantage on at least one asymmetric game.
2. The effect is not explained only by one broken role or one degenerate ruleset.
3. MCTS/random baselines show the games are playable and nontrivial.
4. Results are stable enough across multiple seeds to justify deeper experiments.

## Open Questions

1. How many seeds are enough for the first internal report? Suggested: 3 seeds for smoke, 10 for a serious pilot.
2. Should we use pure MCTS baselines before neural self-play? Suggested: yes, to detect broken games early.
3. Should generated games be introduced before or after the first RL curves? Suggested: after, so generator changes target measurable phenomena.
4. Should human playtesting be part of this branch? Suggested: not yet; keep the branch focused on tractable RL and role-aware metrics.

