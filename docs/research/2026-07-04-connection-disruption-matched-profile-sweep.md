# Connection/Disruption Matched Profile Sweep

Date: 2026-07-04
Branch: `research/asymbench`

## Purpose

This sweep tested whether the new `fair_agent` sampling profile moves
`connection_disruption` toward a usable main benchmark band, compared with the
original `stress` profile.

The experiment used matched seeds so the comparison asks a narrow question:

> Does denser breaker setup reduce the planning-level builder collapse, or does
> it only change random-play balance?

## Commands

Stress profile:

```bash
python -m research.asymbench.experiments.generate_grid_games \
  --family connection_disruption \
  --connection-profile stress \
  --count 30 \
  --seed 8000 \
  --random-games 16 \
  --mcts-games 12 \
  --mcts-simulations 16 \
  --output research_runs/asymbench/profile_split_20260704/matched_sweep/stress
```

Fair-agent profile:

```bash
python -m research.asymbench.experiments.generate_grid_games \
  --family connection_disruption \
  --connection-profile fair_agent \
  --count 30 \
  --seed 8000 \
  --random-games 16 \
  --mcts-games 12 \
  --mcts-simulations 16 \
  --output research_runs/asymbench/profile_split_20260704/matched_sweep/fair_agent
```

Both profiles accepted seeds `8000` through `8029`.

## Validation Schema Addition

The validation report now records enough MCTS diagnostics to compute role and
seat effects from saved JSON:

- `mcts_role_win_rates`
- `mcts_first_player_win_rate`
- `average_mcts_plies`
- `mcts_terminal_reasons`

The older fields remain backward-compatible. Reports without MCTS diagnostics
still load with empty or zero-valued MCTS fields.

## Aggregate Results

| Metric | `stress` | `fair_agent` |
| --- | ---: | ---: |
| Accepted games | 30 | 30 |
| Mean blockers | 2.5333 | 5.2333 |
| Mean initial branching | 32.1000 | 28.9667 |
| Random role 0 win rate | 0.9250 | 0.8188 |
| Random role 1 win rate | 0.0750 | 0.1812 |
| Random role bias | 0.9001 | 0.7959 |
| Random max-ply rate | 0.0750 | 0.1812 |
| Average random plies | 50.1060 | 51.1310 |
| MCTS role 0 win rate | 1.0000 | 1.0000 |
| MCTS role 1 win rate | 0.0000 | 0.0000 |
| MCTS role bias | 1.0000 | 1.0000 |
| MCTS first-player win rate | 0.5000 | 0.5000 |
| MCTS seat bias | 0.0000 | 0.0000 |
| MCTS max-ply rate | 0.0000 | 0.0000 |
| Average MCTS plies | 16.4160 | 16.7663 |
| Main-band candidates | 0 | 0 |
| Hidden-collapse candidates | 4 | 6 |
| Role-inversion candidates | 2 | 4 |

The proposed main-band rule for this pass was:

```text
MCTS role bias <= 0.60 and MCTS seat bias <= 0.15
```

No generated game in either profile passed that screen.

## Paired-Seed Result

Across all 30 matched seeds:

| Paired Metric | Value |
| --- | ---: |
| Mean fair-agent minus stress MCTS role-bias delta | 0.0000 |
| Mean fair-agent minus stress MCTS role-0 win delta | 0.0000 |
| Mean fair-agent minus stress MCTS max-ply delta | 0.0000 |
| Fair-agent improved MCTS role bias | 0 / 30 |
| Stress-only main-band seeds | 0 / 30 |
| Fair-agent-only main-band seeds | 0 / 30 |

This is a clean negative result: denser blockers changed random-play outcomes,
but did not change the equal-MCTS conclusion.

## Interpretation

`fair_agent` did what it was designed to do at the random-play level:

- blocker density roughly doubled;
- random builder dominance decreased;
- random breaker/max-ply wins increased;
- hidden-collapse and role-inversion candidates became more common.

But it failed at the main objective:

- equal-strength MCTS still found builder wins in every sampled game;
- first-player win rate stayed exactly balanced at `0.5`;
- MCTS max-ply wins disappeared, so this was not a horizon-only breaker effect.

That means the failure mode is role-structural:

> The builder has a reliable short planning path that blocker density alone does
> not remove.

This is useful for the novelty story. It shows why AsymBench should not rely on
random validation or simple parameter nudges. The benchmark needs evaluator-based
family diagnostics and separate strata for stress tests versus main-band games.

## Consequence For The Project

The `fair_agent` profile should not yet be treated as a fair-game generator. It
is better described as:

> a higher-disruption sampling profile that increases random/MCTS disagreement,
> but still collapses to builder wins under shallow planning.

The original `stress` and new `fair_agent` profiles are therefore both useful as
stress strata, but neither supplies clean `connection_disruption` controls yet.

## Recommended Next Mechanic Intervention

The next change should alter breaker capability, not just setup density.

Candidate interventions:

1. Breaker tempo.
   Let breaker make two disruption actions on its turn, or give breaker the
   first effective action independent of seat order.

2. Ranged disruption.
   Add range-2 or line-of-sight removal so builder paths cannot force a short
   connection before blockers interact.

3. Objective pressure.
   Require builder to connect through a marked relay cell, or require a thicker
   connection. This slows builder without making breaker wins depend only on
   max-ply.

4. Generator-side MCTS rejection.
   Keep `stress` seeds, but reject `fair_agent` candidates from the main band
   unless `MCTS role bias <= 0.60` and `MCTS seat bias <= 0.15`.

The most sensible next implementation is a new experimental breaker-capability
profile, then another matched sweep against these results.
