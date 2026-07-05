# Connection/Disruption Breaker-Capability Sweep

Date: 2026-07-05
Branch: `research/asymbench`

## Purpose

The previous matched profile sweep showed that adding more blockers did not fix
`connection_disruption`: equal-strength MCTS still found builder wins in every
sampled game.

This pass tested stronger breaker interventions:

1. `ranged_breaker`: denser blockers plus range-2 removal.
2. `line_breaker`: denser blockers plus unobstructed orthogonal line removal.
3. `wall_breaker`: denser blockers, line removal, and protected chokepoint
   terrain in the center column.

The research question was:

> Does breaker capability alone solve builder collapse, or does the family also
> need topology that slows and focuses the builder objective?

## Implementation

`connection_disruption` now supports three additional committed profiles beyond
`stress` and `fair_agent`:

| Profile | Breaker Action Change | Topology Change |
| --- | --- | --- |
| `ranged_breaker` | Adds `range2_remove`. | None. |
| `line_breaker` | Adds `line_remove` along clear ranks/files. | None. |
| `wall_breaker` | Uses `line_remove`. | Adds protected central-column terrain with one or two gaps. |

`line_remove` is blocked by intervening pieces or protected terrain. Existing
`stress` and `fair_agent` specs keep the original adjacent-removal action set.

## Sweep Command

Each profile was generated with:

```bash
python -m research.asymbench.experiments.generate_grid_games \
  --family connection_disruption \
  --connection-profile <profile> \
  --count 30 \
  --seed 8000 \
  --random-games 16 \
  --mcts-games 12 \
  --mcts-simulations 16 \
  --output research_runs/asymbench/profile_split_20260704/matched_sweep/<profile>
```

The `stress` and `fair_agent` rows are from the previous matched sweep. The
`ranged_breaker`, `line_breaker`, and `wall_breaker` rows were added in this pass.

## Aggregate Results

| Metric | `stress` | `fair_agent` | `ranged_breaker` | `line_breaker` | `wall_breaker` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Accepted games | 30 | 30 | 30 | 30 | 30 |
| Mean blockers | 2.5333 | 5.2333 | 5.0333 | 5.3667 | 5.1333 |
| Mean protected cells | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3.1333 |
| Random role 0 win | 0.9250 | 0.8188 | 0.6771 | 0.7605 | 0.6063 |
| Random role 1 win | 0.0750 | 0.1812 | 0.3229 | 0.2395 | 0.3937 |
| Random role bias | 0.9001 | 0.7959 | 0.5709 | 0.7043 | 0.5876 |
| Random max-ply rate | 0.0750 | 0.1812 | 0.3229 | 0.2396 | 0.3937 |
| MCTS role 0 win | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.8862 |
| MCTS role 1 win | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1138 |
| MCTS role bias | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.7834 |
| MCTS first-player win | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5028 |
| MCTS seat bias | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0831 |
| MCTS max-ply rate | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1139 |
| Main-band candidates | 0 | 0 | 0 | 0 | 1 |
| Hidden-collapse candidates | 4 | 6 | 15 | 8 | 5 |
| Role-inversion candidates | 2 | 4 | 8 | 5 | 8 |

The provisional main-band rule was:

```text
MCTS role bias <= 0.60 and MCTS seat bias <= 0.15
```

The stricter clean-main rule also required:

```text
MCTS max-ply rate <= 0.50
```

`wall_breaker` produced one clean-main candidate under the 12-game, 16-simulation
MCTS screen:

| Seed | Board | Blockers | Protected | Random Outcome | MCTS Outcome | MCTS Bias | Seat Bias | MCTS Max-Ply |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: |
| `8003` | 5x5 | 6 | 2 | `(0.688, 0.312)` | `(0.667, 0.333)` | 0.334 | 0.000 | 0.333 |

## Higher-Simulation Probe

The eight best `wall_breaker` candidates were re-evaluated with:

```text
mcts_games = 12
mcts_simulations = 64
```

Key results:

| Seed | 16-sim MCTS | 64-sim MCTS | Interpretation |
| --- | --- | --- | --- |
| `8003` | `(0.667, 0.333)`, bias `0.334`, seat `0.000` | `(0.500, 0.500)`, bias `0.000`, seat `0.000` | Best candidate; survives stronger MCTS. |
| `8011` | `(0.417, 0.583)`, bias `0.166`, seat `0.500` | `(0.583, 0.417)`, bias `0.166`, seat `0.166` | Role balance persists, seat confound improves. |
| `8019` | `(0.667, 0.333)`, bias `0.334`, seat `0.334` | `(0.583, 0.417)`, bias `0.166`, seat `0.166` | Improves under higher simulations but still seat-sensitive. |
| `8014` | `(0.583, 0.417)`, bias `0.166`, seat `0.166` | `(1.000, 0.000)`, bias `1.000`, seat `0.000` | Collapses back to builder under stronger MCTS. |

This makes `connection_disruption_wall_breaker_5x5_seed_8003` the first strong
candidate for a clean control in this family.

## Interpretation

The important result is not that `wall_breaker` solves the family. It does not.
Mean MCTS bias is still high at `0.7834`.

The important result is that topology changes created a new regime:

- local range and line-removal changes alone did not affect equal-MCTS collapse;
- adding protected chokepoints reduced MCTS role bias for some seeds;
- the best 5x5 wall seed remained balanced under higher MCTS simulations;
- several larger-board wall seeds still collapsed, so filtering remains necessary.

This supports a stronger benchmark story:

> AsymBench should generate families with explicit mechanic/topology strata, then
> select seeds by evaluator disagreement and high-simulation role/seat filters.

Random balance alone is still misleading. For example, `ranged_breaker` had the
best random role bias (`0.5709`) before wall topology, yet MCTS still found pure
builder wins.

## Research Consequence

The next serious `connection_disruption` direction is not more local breaker
power. It is topology-aware generation plus evaluator-based acceptance.

Recommended candidate strata:

| Stratum | Candidate Source |
| --- | --- |
| Hidden-collapse stress | `stress`, `fair_agent`, `ranged_breaker`, `line_breaker` |
| Random-balanced but MCTS-collapsed | `ranged_breaker` seeds such as `8010`, `8027` |
| Clean control | `wall_breaker` seed `8003` |
| Seat-confound stress | `wall_breaker` candidates with low role bias but high seat bias |
| Horizon-pressure stress | wall candidates with high MCTS max-ply rate |

## Strata Selection Helper

This pass also added `research.asymbench.analysis.strata`, which classifies
generated validation directories into reusable benchmark strata:

- `clean_control`
- `hidden_collapse`
- `role_inversion`
- `seat_confound`
- `horizon_stress`
- `role_collapse`

On the official `wall_breaker` output, the helper ranks
`connection_disruption_wall_breaker_5x5_seed_8003` as the top clean-control
candidate, matching the manual analysis above.

The helper is now also executable as a repeatable selection/export workflow:

```bash
python -m research.asymbench.analysis.strata \
  --input research_runs/asymbench/profile_split_20260704/matched_sweep/wall_breaker \
  --input research_runs/asymbench/profile_split_20260704/wall_breaker_large_20260705 \
  --limit-per-stratum 10 \
  --output research_runs/asymbench/profile_split_20260704/strata_selection_20260705/wall_combined_selection.json \
  --role-config-template research/asymbench/experiments/configs/breaker_builder_smoke.json \
  --role-config-output research_runs/asymbench/profile_split_20260704/strata_selection_20260705/wall_combined_role_head_configs
```

The export requires MCTS diagnostics by default, so older random-only validation
reports are not accidentally selected as clean controls. The manifest records
the ranked stratum entries, paths to `spec.json` and `validation.json`, metric
values, thresholds, and generated role-head runner configs.

## Partial Larger Sweep

A larger `wall_breaker` run starting at seed `9000` was interrupted after 63
accepted games. That partial sweep found no additional clean-control candidates,
but it did provide useful stress strata:

| Stratum | Count Selected | Top Candidate |
| --- | ---: | --- |
| `hidden_collapse` | 10 | `connection_disruption_wall_breaker_7x7_seed_9029` |
| `role_collapse` | 10 | `connection_disruption_wall_breaker_5x5_seed_9001` |
| `role_inversion` | 10 | `connection_disruption_wall_breaker_7x7_seed_9020` |
| `seat_confound` | 3 | `connection_disruption_wall_breaker_6x6_seed_9074` |
| `horizon_stress` | 10 | `connection_disruption_wall_breaker_6x6_seed_9004` |

When combined with the earlier 30-game `8000` sweep, the selector produced:

| Stratum | Count Selected | Top Candidate |
| --- | ---: | --- |
| `clean_control` | 1 | `connection_disruption_wall_breaker_5x5_seed_8003` |
| `hidden_collapse` | 10 | `connection_disruption_wall_breaker_7x7_seed_9029` |
| `role_collapse` | 10 | `connection_disruption_wall_breaker_5x5_seed_8000` |
| `role_inversion` | 10 | `connection_disruption_wall_breaker_7x7_seed_8002` |
| `seat_confound` | 5 | `connection_disruption_wall_breaker_6x6_seed_8011` |
| `horizon_stress` | 10 | `connection_disruption_wall_breaker_7x7_seed_8002` |

This is a better benchmark construction story than taking accepted games
uniformly at random: clean controls are rare, stress cases are plentiful, and
both are selected by explicit role/seat/horizon criteria.

## Fresh 9100 Sweep

A second larger `wall_breaker` run started at seed `9100` with target count 120.
The generator attempted its default 100-seed window, accepted 98 games, then
exited nonzero because it did not reach the requested 120 accepted games. The
98 accepted games are still valid for analysis.

This batch produced 6 clean-control candidates under the 12-game, 16-simulation
MCTS screen:

| Rank | Seed | Board | 16-Sim MCTS Bias | Seat Bias | MCTS Max-Ply |
| ---: | ---: | --- | ---: | ---: | ---: |
| 1 | `9124` | 6x6 | 0.000 | 0.000 | 0.500 |
| 2 | `9109` | 6x6 | 0.334 | 0.000 | 0.333 |
| 3 | `9142` | 5x5 | 0.334 | 0.000 | 0.333 |
| 4 | `9180` | 6x6 | 0.334 | 0.000 | 0.333 |
| 5 | `9193` | 6x6 | 0.334 | 0.000 | 0.333 |
| 6 | `9198` | 6x6 | 0.334 | 0.000 | 0.333 |

Those six candidates were re-evaluated with 12 equal-MCTS games at 64
simulations:

| Seed | 64-Sim MCTS Outcome | Bias | Seat Bias | MCTS Max-Ply | Interpretation |
| ---: | --- | ---: | ---: | ---: | --- |
| `9124` | `(0.417, 0.583)` | 0.166 | 0.166 | 0.583 | Role-balanced, but slightly above the original seat/max-ply clean thresholds. |
| `9109` | `(0.667, 0.333)` | 0.334 | 0.000 | 0.333 | Survives as a clean control. |
| `9142` | `(0.667, 0.333)` | 0.334 | 0.334 | 0.333 | Role-balanced but seat-sensitive. |
| `9180` | `(0.667, 0.333)` | 0.334 | 0.334 | 0.333 | Role-balanced but seat-sensitive. |
| `9193` | `(0.583, 0.417)` | 0.166 | 0.166 | 0.417 | Good near-clean candidate. |
| `9198` | `(1.000, 0.000)` | 1.000 | 0.000 | 0.000 | Collapses under stronger MCTS. |

Combining the `8000`, partial `9000`, and `9100` batches gives 7 clean-screen
candidates and fills every stress stratum to the current limit of 10:

| Stratum | Count Selected | Top Candidate |
| --- | ---: | --- |
| `clean_control` | 7 | `connection_disruption_wall_breaker_6x6_seed_9124` |
| `hidden_collapse` | 10 | `connection_disruption_wall_breaker_7x7_seed_9029` |
| `role_collapse` | 10 | `connection_disruption_wall_breaker_5x5_seed_8000` |
| `role_inversion` | 10 | `connection_disruption_wall_breaker_7x7_seed_8002` |
| `seat_confound` | 10 | `connection_disruption_wall_breaker_6x6_seed_9181` |
| `horizon_stress` | 10 | `connection_disruption_wall_breaker_7x7_seed_8002` |

This strengthens the benchmark-selection claim: clean controls are uncommon but
recoverable, while diagnostic stress cases are abundant. The next filter should
probably split `clean_control` into strict clean, near-clean with mild seat
sensitivity, and high-simulation survivors.

## Next Implementation Step

Run a completed larger `wall_breaker` sweep or continue the 9100-series search
to select:

- 10 clean controls,
- 10 hidden-collapse stress games,
- 10 role-inversion games,
- 10 seat-confound games,
- 10 horizon-pressure games.

Those selected seeds would be a much better input set for role-head RL and later
LLM/human comparison than taking random accepted games.
