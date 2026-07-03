# AsymBench Generated Grid Families: Empirical Sweep

Date: 2026-07-03
Branch: `research/asymbench`

## Purpose

This note records the first empirical pass over the generated asymmetric grid-game
pipeline. The aim was not to prove a final result, but to check whether the new
family generators produce analyzable games and whether the role-head AlphaZero-lite
runner can expose useful differences across generated asymmetric roles.

The work follows Hafsteinn's direction: keep the first experiments tractable, focus
on what existing LLM game-balancing and virtual-playtester papers do not directly
answer, and test whether generated asymmetric games can be evaluated by role-aware
RL agents under seat swaps and paired randomness.

## Experiment Setup

Generated games:

- Command: `python -m research.asymbench.experiments.generate_grid_games --family all --count 50 --seed 1000 --random-games 8 --output research_runs/asymbench/empirical_sweep_20260703/generated`
- Accepted games: 50 `escape_capture`, 50 `connection_disruption`.
- Validation rollouts: 8 random games per accepted spec.
- Raw outputs: `research_runs/asymbench/empirical_sweep_20260703/`

Role-head sample:

- Selected 4 games per family from the validation sweep.
- Selection covered balanced random roles, high max-ply pressure, low max-ply pressure, short random games, and long random games.
- Device: CUDA on RTX 4080.
- Config per selected game:
  - seeds: `[1, 2]`
  - variants: `shared_heads`, `role_heads`
  - iterations: `2`
  - self-play games per iteration: `4`
  - train steps per iteration: `4`
  - MCTS simulations: `4`
  - eval games: `4`
  - eval simulations: `2`

## Validation Sweep Summary

Each family had 400 random validation rollouts total.

| Family | Accepted | Board Sizes | Terminal Reasons | Role 0 Mean Win | Role 1 Mean Win | Max-Ply Rate | Avg Random Plies | Initial Branching |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Escape/Capture | 50 | 5x5: 17, 6x6: 19, 7x7: 14 | key_capture: 167, key_escape: 96, max_plies: 137 | 0.4175 | 0.2400 | 0.3425 | 51.640 | 10.640 |
| Connection/Disruption | 50 | 5x5: 19, 6x6: 21, 7x7: 10 | builder_connection: 367, max_plies: 33 | 0.9175 | 0.0825 | 0.0825 | 49.395 | 31.700 |

Interpretation:

- Escape/Capture is substantially more varied under random play. It has attacker wins, defender escapes, and many horizon draws.
- Connection/Disruption is strongly builder-biased under random play in this generator range. The breaker can win by horizon, but the random baseline rarely creates enough disruption.
- This is useful, not a failure. It gives us a concrete generator-tuning target: increase destructive pressure in Connection/Disruption and reduce max-ply draw pressure in some Escape/Capture instances.

Role labels:

- Escape/Capture role 0 is attacker, role 1 is defender. `max_plies` is a draw.
- Connection/Disruption role 0 is builder, role 1 is breaker. `max_plies` is a breaker win.

## Role-Head Sample Summary

The following table reports final-iteration means across two training seeds per selected game.

| Family | Game | Selection | Validation Role 0 / Role 1 / Draw | Shared Model Win | Role-Head Model Win | Delta | Shared Avg Plies | Role-Head Avg Plies |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Connection/Disruption | `connection_disruption_7x7_seed_1002` | long_random_games | 1.000 / 0.000 / 0.000 | 0.375 | 0.250 | -0.125 | 79.375 | 80.125 |
| Connection/Disruption | `connection_disruption_6x6_seed_1004` | balanced_random_roles | 0.500 / 0.500 / 0.000 | 0.625 | 0.125 | -0.500 | 41.750 | 47.625 |
| Connection/Disruption | `connection_disruption_6x6_seed_1006` | high_max_ply_pressure | 0.250 / 0.750 / 0.000 | 0.500 | 0.625 | 0.125 | 37.875 | 41.625 |
| Connection/Disruption | `connection_disruption_5x5_seed_1047` | low_max_ply_pressure, short_random_games | 1.000 / 0.000 / 0.000 | 0.500 | 0.500 | 0.000 | 51.750 | 45.750 |
| Escape/Capture | `escape_capture_6x6_seed_1004` | high_max_ply_pressure | 0.125 / 0.000 / 0.875 | 0.000 | 0.000 | 0.000 | 53.000 | 53.000 |
| Escape/Capture | `escape_capture_5x5_seed_1009` | short_random_games | 0.500 / 0.375 / 0.125 | 0.250 | 0.000 | -0.250 | 17.625 | 32.375 |
| Escape/Capture | `escape_capture_6x6_seed_1012` | balanced_random_roles | 0.125 / 0.125 / 0.750 | 0.000 | 0.250 | 0.250 | 56.375 | 52.875 |
| Escape/Capture | `escape_capture_5x5_seed_1024` | low_max_ply_pressure | 0.875 / 0.125 / 0.000 | 0.250 | 0.125 | -0.125 | 17.750 | 38.625 |

Family-level role-head deltas:

| Family | Sampled Games | Shared Model Win | Role-Head Model Win | Role-Head Minus Shared | Shared Draw Rate | Role-Head Draw Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Escape/Capture | 4 | 0.1250 | 0.0938 | -0.0312 | 0.5312 | 0.6250 |
| Connection/Disruption | 4 | 0.5000 | 0.3750 | -0.1250 | 0.0000 | 0.0000 |

## What This Suggests

The strongest immediate result is not "role heads are better." In this tiny
sample they are not consistently better. The stronger result is that the pipeline
now exposes measurable family-specific asymmetries:

- random baselines identify role dominance and horizon pressure;
- generated metadata follows the game into RL metrics and summaries;
- paired seeds make shared-head and role-head comparisons directly auditable;
- the same evaluator can run both generated families without a universal DSL.

That supports the core novelty direction:

> AsymBench can generate controlled asymmetric game families and evaluate them with role-aware RL, measuring role dominance, learnability, horizon pressure, and architecture sensitivity under paired randomness.

This differs from RuleSmith-style balancing and MeepleLM-style virtual playtesting because the generated game itself is not the only artifact. The benchmark also stores role-conditioned learning behavior for each generated asymmetric instance.

## Research Implications

1. Family definitions are necessary.
   The two families behave differently enough that "asymmetric game generation" is too broad as one state/action representation. Grid escape games and connection-disruption games need separate validity and balance criteria.

2. Generator validation must be empirical, not just syntactic.
   Connection/Disruption can pass structural checks while remaining builder-favored under random play. Escape/Capture can pass legality while becoming max-ply heavy. The validation report is therefore part of the benchmark, not just a filter.

3. Role-aware RL is a useful evaluator even when the headline effect is null.
   A null or mixed role-head advantage is still informative because it localizes which generated instances produce architecture sensitivity.

4. The next novelty target should be generator tuning by evaluator disagreement.
   Interesting games are those where random, shallow MCTS, shared-head, role-head, LLM, and human agents disagree about balance or learnability.

## Concrete Next Steps

1. Add shallow-MCTS validation reports.
   Random rollouts are good for first screening, but they overstate builder strength in Connection/Disruption. A shallow MCTS validator would give a second balance lens.

2. Tune Connection/Disruption generator pressure.
   Increase blocker count or blocker mobility, add removal opportunities, or vary protected terrain so breaker wins are not mostly horizon wins.

3. Reduce Escape/Capture horizon draws.
   Adjust max plies, exit placement, guard count, or attacker density to produce more decisive key escape/capture outcomes.

4. Run a larger role-head sweep on selected nontrivial games.
   Use the validation sweep to select games with:
   - random role win rates between 0.25 and 0.75;
   - max-ply rate below 0.5;
   - nontrivial branching factor;
   - both terminal modes observed.

5. Add an evaluator-disagreement score.
   A generated game should be ranked higher when agents disagree in structured ways, for example random says builder-favored while role-head RL finds breaker counterplay.

6. Preserve selected generated examples as benchmark seeds.
   The report identifies candidate seeds worth tracking:
   - `connection_disruption_6x6_seed_1004`
   - `connection_disruption_6x6_seed_1006`
   - `escape_capture_5x5_seed_1009`
   - `escape_capture_6x6_seed_1012`

## Limitations

- The role-head sweep is intentionally small and should not be treated as a final learning result.
- Evaluation used tiny MCTS and short training schedules.
- The generator CLI writes accepted artifacts only, so rejection-rate analysis is limited to outer seed gaps and accepted validation reports.
- No LLM or human-agent comparison has been run yet.
- The current families are grid-based, deterministic, perfect-information games only.

## Reproducibility Notes

Verification during this pass:

- Full tests: `203 passed`.
- CUDA: PyTorch `2.11.0+cu128`, RTX 4080 available.
- Generated sweep: 50 accepted specs per family.
- Role-head sample: 8 generated games, 2 training seeds each.
- Metrics audit: final rows contained paired-seed fields and generated metadata.

Important artifact paths:

- Generated specs and validation reports: `research_runs/asymbench/empirical_sweep_20260703/generated/`
- Aggregated validation tables: `research_runs/asymbench/empirical_sweep_20260703/analysis/`
- Role-head run configs: `research_runs/asymbench/empirical_sweep_20260703/role_head_configs/`
- Role-head run outputs: `research_runs/asymbench/empirical_sweep_20260703/role_head_runs/`

