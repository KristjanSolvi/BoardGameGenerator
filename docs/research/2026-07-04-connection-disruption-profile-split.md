# Connection/Disruption Profile Split

Date: 2026-07-04
Branch: `research/asymbench`

## Purpose

The previous sweeps showed that `connection_disruption` is useful but overloaded:
it produces hidden-collapse stress tests, while the main benchmark also needs
candidate games that have a plausible chance of fair-agent balance under shallow
planning.

This note records the first code-level split between those two uses.

## Implemented Profiles

`ConnectionDisruptionGenerator` now supports two sampling profiles:

| Profile | Use | Current Sampling Change |
| --- | --- | --- |
| `stress` | Preserve the original hidden-collapse regime. | Default behavior; 1 to 4 initial blockers, original generated names. |
| `fair_agent` | Search for main-band candidates that give breaker more early influence. | Denser initial blocker starts; generated names are prefixed with `connection_disruption_fair_agent_`. |

This is not yet a claim that `fair_agent` games are balanced. It is a controlled
generator stratum that should be filtered by MCTS role/seat diagnostics.

## Validation Change

`validate_generated_game` now accepts optional shallow MCTS diagnostics:

```bash
python -m research.asymbench.experiments.generate_grid_games \
  --family connection_disruption \
  --connection-profile fair_agent \
  --count 20 \
  --seed 7000 \
  --random-games 16 \
  --mcts-games 12 \
  --mcts-simulations 16 \
  --output research_runs/asymbench/profile_split_20260704/generated
```

When `--mcts-games` is set, validation JSON fills the existing
`mcts_role_win_rates` field. With the default `--mcts-games 0`, generation speed
and old reports stay unchanged.

## Why This Matters For Novelty

The benchmark should not treat every asymmetric generated game as one pool.
`connection_disruption` now has explicit strata:

- `stress`: keep seeds that reveal hidden role collapse, role inversion, or
  random-vs-planning disagreement.
- `fair_agent`: produce candidates for the main benchmark band, then filter them
  by role bias, seat bias, and evaluator disagreement.

This supports the defensible paper framing:

> AsymBench generates asymmetric game families and selects benchmark instances by
> evaluator disagreement and role/seat separation, not by random-play balance
> alone.

## Next Empirical Check

Run matched sweeps for `stress` and `fair_agent` with the same seeds and MCTS
settings. The immediate question is whether denser breaker starts reduce the
equal-MCTS builder collapse without turning the family into horizon-only breaker
wins.

Useful acceptance targets for `fair_agent` candidates:

| Metric | Target |
| --- | --- |
| MCTS role bias | `<= 0.60` for main-band candidates |
| Seat bias | `<= 0.15` |
| Random max-ply rate | below the stress-only band |
| Evaluator disagreement | keep both high-disagreement and clean-control strata |

If denser blockers are not enough, the next mechanic-level intervention should
be stronger breaker tempo or range, not just larger boards.
