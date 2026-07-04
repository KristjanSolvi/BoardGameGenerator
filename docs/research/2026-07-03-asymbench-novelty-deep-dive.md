# AsymBench Novelty Deep Dive: Generated Asymmetric Families as Evaluator-Disagreement Benchmarks

Date: 2026-07-03
Branch: `research/asymbench`

## Executive Takeaway

The strongest novelty direction is not "generate asymmetric board games" and not "role heads beat shared heads." Both claims are too broad and already have nearby prior art.

The defensible research claim is:

> AsymBench can generate executable families of asymmetric games and use role-aware evaluator disagreement to select benchmark instances where random agents, MCTS, RL/self-play variants, LLM agents, and eventually humans disagree about role strength, seat advantage, learnability, and strategic difficulty.

This is meaningfully different from text-only virtual playtesting, fixed-game LLM benchmarks, single-game asymmetric balancing, and broad executable game generation. The empirical result from this pass is that family choice matters immediately: `connection_disruption` behaves like a hidden role-collapse stress test under planning, while `escape_capture` is more draw- and seat-sensitive.

## Related Work Boundary

The nearby literature is strong, so the novelty claim has to be narrow and explicit.

| Work | What it already does | Boundary for AsymBench |
| --- | --- | --- |
| [RuleSmith](https://arxiv.org/html/2602.06232v1) | Balances an executable asymmetric game, CivMini, with multi-agent LLM self-play and Bayesian optimization over rule parameters. | It tunes one hand-designed parameterized game. It does not generate benchmark families or study role/seat/evaluator disagreement across generated instances. |
| [MeepleLM](https://aclanthology.org/2026.acl-long.850/) | Trains a virtual playtester from rulebooks, reviews, MDA reasoning, and player personas to produce subjective board-game critique. | It does not execute game mechanics or measure strategic role strength. It is complementary as an experience evaluator. |
| [AutoBG](https://arxiv.org/html/2606.01976v1) | Provides end-to-end board-game design assistance: ideation, rulebook generation, verifier-gated revision, and persona feedback. | It is mostly rulebook/workflow oriented. AsymBench can be the mechanical evaluation layer for generated or drafted rules. |
| [GAVEL](https://arxiv.org/abs/2407.09388) | Generates executable board games in Ludii using LLM mutation plus evolutionary quality-diversity search. It already uses balance, decisiveness, completion, coverage, and MCTS-vs-random strategic-depth metrics. | We cannot claim executable generated board games as novel. Our distinction is role-conditioned asymmetric benchmark selection and evaluator disagreement, not broad game generation. |
| [Board/Game Reasoning Arena](https://arxiv.org/html/2508.03368v1) and [GameBench](https://arxiv.org/html/2406.06613v2) | Evaluate LLM strategic reasoning on fixed games, with random/human/RL/LLM baselines. | These are fixed-game agent benchmarks. AsymBench generates controlled asymmetric game families and measures disagreement over generated instances. |
| [PCG Benchmark](https://arxiv.org/abs/2503.21474) and [Procgen](https://proceedings.mlr.press/v119/cobbe20a.html) | Establish procedural generation as a benchmark method for content generation and RL generalization. | They motivate generated environments, but not role-conditioned board-game families or role/seat disentanglement. |
| [Automated Game Balancing of Asymmetric Video Games](https://sander.landofsand.com/publications/bakkes_cig2016_paper_74.pdf) | Uses Monte Carlo simulation to identify and adjust imbalance in asymmetric video-game configurations. | This is balancing/optimization for existing designs, not generated asymmetric rule-family benchmarking with multiple evaluator classes. |
| [Level the Level](https://arxiv.org/html/2503.24099v1) | Uses RL to balance tile levels for asymmetric player archetypes. | It optimizes level layouts for archetype balance, not executable rule-family generation or cross-evaluator role-collapse discovery. |
| [AlphaZero on Tablut](https://arxiv.org/html/2604.05476v1) | Shows AlphaZero can transfer to a highly asymmetric historical game with separate role heads, while exposing instability and role forgetting. | Role-specific AlphaZero heads are not novel. Our role-head runner should be framed as one evaluator probe inside a generated benchmark. |

Unsafe claims to avoid:

- "First executable board-game generator."
- "First LLM asymmetric game-balancing system."
- "First virtual board-game playtester."
- "Role-specific AlphaZero heads are novel."
- "Role heads are generally better."
- "MCTS filtering is new for generated games."

## What We Built In This Branch

This branch now contains a small but coherent first-generation AsymBench pipeline:

- two generated deterministic perfect-information grid families;
- serializable generated specs and runtime compilation;
- random validation with structural checks;
- shallow MCTS-vs-random evaluation;
- equal-strength MCTS-vs-MCTS evaluation with seat-role swaps;
- shared-head versus role-head AlphaZero-lite self-play experiments;
- a reusable disagreement metric module in `research/asymbench/analysis/disagreement.py`.

The current families are intentionally narrow:

- `escape_capture`: attacker/defender pursuit with key escape versus key capture.
- `connection_disruption`: builder/breaker path construction versus disruption/prevention.

This narrowness is a feature for the first paper direction. "Asymmetric games" is too broad to evaluate as one space; grid pursuit and grid connection already require different validity bands.

## Metric Definitions

For each game `g` and evaluator `e`, define an outcome vector:

```text
O_e(g) = (P(role0 win), P(role1 win), P(draw or max-ply draw))
```

The main benchmark metrics are:

```text
EvaluatorDisagreement(g) =
  max_{a,b} 0.5 * ||O_a(g) - O_b(g)||_1
```

```text
RoleBias(g) = |P_MM(role0 win) - P_MM(role1 win)|
SeatBias(g) = |2 * P_MM(first-player win) - 1|
RoleSeatSeparation(g) = RoleBias(g) - SeatBias(g)
```

`MM` means equal-strength MCTS-vs-MCTS with seat-role swaps.

```text
HiddenRoleCollapse(g) =
  1[min(P_random(role0), P_random(role1)) >= 0.25] * RoleBias_MM(g)
```

This flags games that look approximately balanced under random rollouts but collapse to one role under planning.

```text
RoleInversion(g) =
  1[(P_random(role0)-0.5)(P_MM(role0)-0.5) < 0]
  * |P_random(role0) - P_MM(role0)|
```

This catches games where evaluator skill reverses the apparent favored role.

```text
ArchitectureDelta(g) =
  WinRate(role_heads) - WinRate(shared_heads)
```

This should be treated as architecture sensitivity, not as a universal role-head superiority claim.

## Experiment Scope

Generated validation:

- `250` accepted `escape_capture` games.
- `250` accepted `connection_disruption` games.
- `32` random rollouts per accepted game.
- Seeds ranged from `5000` upward.

Evaluator layers:

- `36` selected candidates, `18` per family.
- MCTS-vs-random: `16` games per candidate, `32` MCTS simulations.
- MCTS-vs-MCTS: `24` games per candidate, `24` simulations for both seats.
- Broad role-head sweep: `36` candidates, `3` seeds, `3` iterations, CUDA.
- Selected deeper role-head sweep: `12` high-disagreement candidates, `4` seeds, `5` iterations, CUDA.
- Stability role-head probe: `6` contentious candidates, `3` new seeds, `7` iterations, CUDA.

## Validation Results

| Family | Accepted | Invalid Attempts | Role 0 Mean | Role 1 Mean | Draw/Max-Ply Mean | Avg Random Plies | Initial Branching |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `escape_capture` | 250 | 3 | 0.4065 | 0.2281 | 0.3654 | 52.756 | 10.992 |
| `connection_disruption` | 250 | 1 | 0.8729 | 0.1271 | 0.1271 | 51.348 | 33.916 |

Interpretation:

- `escape_capture` is much more mixed under random play: attacker wins, defender wins, and many horizon draws all occur.
- `connection_disruption` is builder-favored overall under random play, but some seeds look balanced by random rollout and later collapse under equal MCTS.
- Rejection rate is currently low, so the generators are producing compilable/playable artifacts; the problem is strategic validity, not syntax.

## Equal-Strength MCTS Findings

MCTS-vs-random showed that shallow planning beats random in both families, but the equal-strength MCTS-vs-MCTS layer is the key novelty evidence.

| Family | Candidates | Role 0 Win Mean | Role 1 Win Mean | Draw Mean | First-Player Win Mean | Avg Plies |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `connection_disruption` | 18 | 0.9977 | 0.0023 | 0.0000 | 0.4977 | 16.653 |
| `escape_capture` | 18 | 0.4028 | 0.3750 | 0.2222 | 0.4074 | 38.019 |

The important result is the separation between role and seat:

- `connection_disruption` becomes almost pure role-0/builder wins under equal MCTS, while first-player win rate remains almost exactly balanced.
- That means the issue is not seat order. It is role structure.
- `escape_capture` is less role-collapsed but has more seat and draw sensitivity.

## Evaluator-Disagreement Results

Using the reusable disagreement metrics:

| Family | Candidates | Mean Evaluator Disagreement | High-Disagreement Count | Hidden-Collapse Count | Mean Role Bias | Mean Seat Bias | Role-Driven Count | Seat-Confounded Count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `connection_disruption` | 18 | 0.3380 | 3 | 10 | 0.9954 | 0.0046 | 18 | 0 |
| `escape_capture` | 18 | 0.2633 | 1 | 1 | 0.2361 | 0.3333 | 1 | 11 |

Strong examples:

| Seed | Family | Random Outcome | Equal-MCTS Outcome | Why It Matters |
| --- | --- | --- | --- | --- |
| `connection_disruption_5215` | Connection/Disruption | `(0.031, 0.969, 0.000)` | `(1.000, 0.000, 0.000)` | Role inversion under skill: random says breaker, MCTS says builder. |
| `connection_disruption_5047` | Connection/Disruption | `(0.531, 0.469, 0.000)` | `(1.000, 0.000, 0.000)` | Hidden role collapse from random-balanced to builder-forced. |
| `connection_disruption_5064` | Connection/Disruption | `(0.531, 0.469, 0.000)` | `(1.000, 0.000, 0.000)` | Same collapse on a 6x6 seed. |
| `connection_disruption_5081` | Connection/Disruption | `(0.469, 0.531, 0.000)` | `(1.000, 0.000, 0.000)` | Random slightly favors breaker; planning flips to builder. |
| `connection_disruption_5203` | Connection/Disruption | `(0.469, 0.531, 0.000)` | `(1.000, 0.000, 0.000)` | Another clean role-not-seat collapse. |
| `escape_capture_5160` | Escape/Capture | `(0.500, 0.469, 0.031)` | `(1.000, 0.000, 0.000)` | Escape/Capture can also produce hidden collapse, but less often. |

This is the clearest publishable phenomenon so far:

> Random rollout balance is not enough for generated asymmetric games. Some generated games look fair until a stronger evaluator reveals that one role has a reliable planning advantage, and seat swaps show that this is role-driven rather than first-move-driven.

## Parameter Diagnostics

The parameter diagnostics also support family-specific validity bands.

`escape_capture`:

- 5x5 games had mean max-ply rate `0.116`.
- 6x6 games had mean max-ply rate `0.359`.
- 7x7 games had mean max-ply rate `0.611`.
- More guards increased max-ply pressure: 2 guards `0.260`, 3 guards `0.381`, 4 guards `0.479`.
- More attackers reduced draw pressure but increased role imbalance.

Implication: 7x7 Escape/Capture should be treated as horizon-stress unless exits, attacker density, or max-plies are tuned. Good main-band candidates should probably be 5x5/6x6 with nonzero but not dominant draw rates.

`connection_disruption`:

- Random balance gap decreased as board size increased: 5x5 `0.961`, 6x6 `0.841`, 7x7 `0.689`.
- Max-ply pressure increased with board size: 5x5 `0.019`, 6x6 `0.125`, 7x7 `0.240`.
- Equal-strength MCTS still produced almost total builder wins in the selected candidates.

Implication: making the board larger can make random play look fairer, but does not fix the planning-level builder advantage. The breaker likely needs stronger disruption mechanics, not only larger boards.

## Role-Head Findings

The broad 36-game CUDA sweep does not support a simple "role heads are better" claim.

| Family | Completed Games | Mean Architecture Delta | SD | Role Heads Better | Shared Heads Better | Ties |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `connection_disruption` | 18 | -0.1605 | 0.1727 | 4 | 14 | 0 |
| `escape_capture` | 18 | -0.0030 | 0.1071 | 8 | 7 | 3 |

Interpretation:

- Connection/Disruption favored shared heads under the broad short schedule.
- Escape/Capture was mixed and near neutral.
- Role heads remain useful as a diagnostic probe because some generated games are architecture-sensitive, but we should not present them as a universally superior architecture.

Broad architecture-sensitive seeds:

- Role-head positive: `escape_capture_5049` `+0.222`, `connection_disruption_5215` `+0.167`, `escape_capture_5109` `+0.167`.
- Shared-head positive: `connection_disruption_5116` `-0.444`, `connection_disruption_5200` `-0.389`, `connection_disruption_5064` `-0.334`, `connection_disruption_5156` `-0.334`.

The selected deeper 12-game sweep used different training seeds and a heavier schedule: `4` seeds, `5` iterations, `8` self-play games per iteration, `12` training steps per iteration, `12` MCTS simulations, and `8` evaluation games.

| Family | Completed Games | Mean Architecture Delta | SD | Role Heads Better | Shared Heads Better | Ties |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `connection_disruption` | 6 | 0.0521 | 0.1107 | 2 | 3 | 1 |
| `escape_capture` | 6 | 0.0104 | 0.1660 | 3 | 2 | 1 |

Selected-deep architecture-sensitive seeds:

- Role-head positive: `connection_disruption_5047` `+0.250`, `connection_disruption_5064` `+0.156`, `escape_capture_5014` `+0.156`, `escape_capture_5200` `+0.156`, `escape_capture_5049` `+0.125`.
- Shared-head positive: `escape_capture_5138` `-0.312`, `escape_capture_5160` `-0.062`, `connection_disruption_5081` `-0.031`, `connection_disruption_5203` `-0.031`, `connection_disruption_5215` `-0.031`.

Several deltas changed sign between the broad short sweep and the selected deeper sweep:

| Seed | Broad Delta | Selected-Deep Delta | Interpretation |
| --- | ---: | ---: | --- |
| `escape_capture_5200` | -0.222 | 0.156 | Schedule/seed-sensitive, not a stable shared-head advantage. |
| `connection_disruption_5064` | -0.334 | 0.156 | Strong sign flip; needs a stability probe before paper claims. |
| `connection_disruption_5047` | -0.278 | 0.250 | Strong sign flip on a hidden-collapse seed. |
| `escape_capture_5138` | 0.056 | -0.312 | Stronger run favored shared heads. |
| `connection_disruption_5215` | 0.167 | -0.031 | Earlier role-head advantage did not persist. |

Interpretation:

- Architecture sensitivity is real enough to measure, but not yet stable enough to be a headline result.
- The stable paper claim should be evaluator disagreement and role/seat separation.
- Role-head experiments should be presented as a diagnostic axis and a future ablation, especially for hidden-collapse seeds.

The final six-seed stability probe tested the most contentious cases with new training seeds and a heavier schedule: `3` seeds, `7` iterations, `10` self-play games per iteration, `18` training steps per iteration, `16` MCTS simulations, and `12` evaluation games.

| Family | Completed Games | Mean Architecture Delta | SD | Role Heads Better | Shared Heads Better | Ties |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `connection_disruption` | 3 | -0.0556 | 0.1771 | 1 | 2 | 0 |
| `escape_capture` | 3 | -0.0276 | 0.0988 | 1 | 2 | 0 |

Per-seed stability comparison:

| Seed | Broad Delta | Selected-Deep Delta | Stability Delta | What Survived |
| --- | ---: | ---: | ---: | --- |
| `connection_disruption_5047` | -0.278 | 0.250 | 0.194 | Role-head positive persisted from selected-deep but contradicted broad. |
| `connection_disruption_5064` | -0.334 | 0.156 | -0.167 | Stability returned toward broad shared-head advantage. |
| `connection_disruption_5215` | 0.167 | -0.031 | -0.194 | Broad role-head advantage did not persist. |
| `escape_capture_5049` | 0.222 | 0.125 | -0.083 | Earlier role-head positives did not persist. |
| `escape_capture_5160` | -0.055 | -0.062 | 0.111 | Stability flipped mildly positive for role heads. |
| `escape_capture_5200` | -0.222 | 0.156 | -0.111 | Stability returned toward broad shared-head advantage. |

Stability interpretation:

- Architecture sensitivity is measurable but not stable enough for the main novelty claim.
- The strongest stable claim remains evaluator disagreement, especially hidden role collapse and role-vs-seat separation.
- The role-head ablation should be retained as an analysis axis because some seeds remain architecture-sensitive, but the current data argues against presenting role-heads as a superior evaluator.
- `connection_disruption_5047` is the most interesting architecture-sensitive exception: it stayed role-head positive in the selected-deep and stability probes despite being shared-head positive in the broad short sweep.

## What Is Novel Enough To Pursue

The strongest paper direction is:

> Generated asymmetric game families as controlled probes for evaluator disagreement and role-conditioned learning.

The contribution would not be a new game-playing algorithm. It would be a benchmark methodology:

1. Generate executable asymmetric games inside defined families.
2. Validate them under random play, MCTS-vs-random, equal-strength MCTS-vs-MCTS, role-aware RL, LLM agents, and human play.
3. Select benchmark seeds by structured disagreement, not only by aggregate balance.
4. Report role advantage, seat advantage, draw/horizon pressure, architecture sensitivity, and human/LLM divergence separately.

That creates a publishable distinction from existing work:

- RuleSmith optimizes one asymmetric game; AsymBench samples a family distribution.
- GAVEL generates broad playable games; AsymBench focuses on role-conditioned asymmetric benchmark strata.
- MeepleLM/AutoBG predict or assist subjective design feedback; AsymBench executes mechanics and measures strategic outcomes.
- Tablut adapts AlphaZero to one asymmetric game; AsymBench uses role-aware RL as one evaluator over generated games.

## Proposed Benchmark Strata

A serious paper should not rank all games with one global score. It should select strata:

| Stratum | Rule |
| --- | --- |
| High evaluator disagreement | `EvaluatorDisagreement >= 0.50` |
| Hidden role collapse | Random-balanced, then `RoleBias_MM >= 0.90` |
| Role inversion | `RoleInversion >= 0.40` |
| Clean controls | `EvaluatorDisagreement < 0.15`, low role bias, low seat bias |
| Seat-confound stress | `SeatBias >= 0.25` and `SeatBias >= RoleBias` |
| Horizon/draw stress | high draw or max-ply rate under random or MCTS |
| Architecture-sensitive | `|ArchitectureDelta| >= 0.20` broad sweep or `>= 0.125` deeper sweep |

This lets us publish a benchmark that is intentionally diagnostic rather than pretending every generated game is a balanced game design.

## Family Validity Bands

Suggested first validity bands:

`escape_capture` main band:

- `min(role0_random, role1_random) >= 0.10`
- `0.05 <= draw_random <= 0.70`
- `30 <= avg_random_plies <= 80`
- `7 <= initial_branching <= 15`
- include high-disagreement stress seeds separately from clean controls

`connection_disruption` random-screen band:

- `min(role0_random, role1_random) >= 0.10`
- `35 <= avg_random_plies <= 75`
- `24 <= initial_branching <= 48`

`connection_disruption` fair-agent band:

- `RoleBias_MM <= 0.60`
- `SeatBias <= 0.15`

Most current selected Connection/Disruption seeds fail the fair-agent band. That is not a failure of the project; it is a useful result showing that this generator family needs stronger breaker mechanics before it can supply balanced main-band games.

## Immediate Research Plan

1. Tune Connection/Disruption mechanics.
   Add stronger breaker pressure: longer disruption range, more blockers, protected-cell variation that helps breaker, delayed builder placement, or breaker tempo changes. Re-test with equal MCTS.

2. Preserve hidden-collapse seeds as stress tests.
   Keep `connection_disruption_5047`, `5064`, `5081`, `5203`, and `escape_capture_5160` as named diagnostic seeds even if we later tune the generator.

3. Add MCTS validation to generation acceptance.
   Random validation should be only the first screen. A generated game should not enter the main benchmark band without equal-strength MCTS role/seat metrics.

4. Expand family definitions before adding non-grid games.
   The next family should be another deterministic perfect-information grid family only if it adds a distinct asymmetry type. Auctions, hidden information, negotiation, and social deduction should wait until the family-specific pipeline is stronger.

5. Add LLM and human layers only after the mechanical strata are stable.
   LLM/human evaluation becomes more meaningful when each tested seed has a known mechanical profile: hidden collapse, role inversion, clean control, seat confound, or horizon stress.

6. Report null and mixed role-head results honestly.
   The architecture-sensitivity result is useful precisely because it is not a universal win for role heads.

## Current Limitations

- The families are grid-only, deterministic, perfect-information, two-player games.
- The current RL schedules are still shallow relative to full AlphaZero-style training.
- Equal MCTS uses uniform evaluation, so it is a planning probe, not a solved-game oracle.
- Connection/Disruption currently overproduces builder-favored games under equal MCTS.
- No LLM-agent or human-agent comparison has been run yet.
- No confidence intervals or formal statistical model has been added yet.

## Bottom Line

The project is novel if it becomes an asymmetric-game benchmark built around generated families and evaluator disagreement. It is not novel if it is framed as a generic game generator, a generic LLM playtester, or an AlphaZero role-head paper.

The current empirical evidence already supports the family-based direction: two superficially similar grid-game families produce sharply different failure modes. That is exactly why defining families first was the right move.
