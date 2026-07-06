# Escape-Collapse Deepening Probe Results

Date: 2026-07-06
Branch: `research/asymbench`
Tooling commit: `b07e692`

## Purpose

The previous stability probe found that `collapse::escape_capture` was the
strongest role-head-positive cell:

```text
2 generated games
16 paired training-seed deltas
mean pooled seed delta = +0.0911
seed CI95 = [+0.0001, +0.1773]
```

That was the best candidate for a robust architecture-sensitive asymmetric game
cell, but it was still based on only two generated games. This deepening probe
tests whether the positive role-head effect survives when the cell is expanded
to four generated games while avoiding role-inversion and seat-sensitive
overlap.

## Selection

The committed manifests contained nine `collapse::escape_capture` candidates.
Several overlapped with `role_inversion` or `seat_sensitive`, so this probe used
the four cleanest collapse-focused seeds:

```text
5160: hidden_collapse + role_collapse, verified_hidden_collapse
5115: role_collapse, high_sim_collapsed
5121: role_collapse, high_sim_collapsed
5051: role_collapse
```

The goal was to test the cell itself, not an entangled collapse/inversion or
collapse/seat confound.

## Setup

Template:

```text
device = cuda
seeds = [101, 202, 303, 404, 505, 606, 707, 808]
iterations = 6
selfplay_games_per_iteration = 8
train_steps_per_iteration = 16
mcts_simulations = 12
eval_games = 32
eval_simulations = 12
batch_size = 32
replay_capacity = 4096
```

Pilot generation command:

```bash
python -m research.asymbench.experiments.prepare_manifest_pilot \
  --manifest docs/research/2026-07-05-connection-disruption-benchmark-manifest.json \
  --manifest docs/research/2026-07-05-escape-capture-benchmark-manifest.json \
  --output-root research_runs/asymbench/role_head_escape_collapse_deep_20260706 \
  --template research_runs/asymbench/role_head_escape_collapse_deep_20260706/deep_template.json \
  --per-bucket-per-family 4 \
  --cell collapse::escape_capture \
  --candidate-seed 5160 \
  --candidate-seed 5115 \
  --candidate-seed 5121 \
  --candidate-seed 5051
```

Aggregation command:

```bash
python -m research.asymbench.analysis.summarize_manifest_pilot \
  research_runs/asymbench/role_head_escape_collapse_deep_20260706/pilot_manifest.json \
  --output research_runs/asymbench/role_head_escape_collapse_deep_20260706/deep_summary_final.json
```

The CUDA run completed all four configs:

```text
completed = 4 / 4
elapsed = 11413.5 seconds
elapsed_hours = 3.17
```

## Cell Result

`architecture_delta` means:

```text
role_heads mean model win - shared_heads mean model win
```

| Cell | Entries | Paired Seeds | Shared Win | Role-Head Win | Entry Delta | Pooled Seed Delta | Seed CI95 | Positive Seeds | Negative Seeds | Zero Seeds | Seed Sign Stability |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `collapse::escape_capture` | 4 | 32 | `0.4200` | `0.4777` | `+0.0577` | `+0.0577` | `[-0.0048, +0.1242]` | 18 | 8 | 6 | `0.5625` |

Draw rates:

```text
shared_heads draw rate = 0.2265
role_heads draw rate = 0.1973
```

## Entry Results

| Seed | Strata | Labels | Shared Win | Role-Head Win | Delta | Seed CI95 | Positive Seeds | Negative Seeds | Zero Seeds | Seed Sign Stability |
| ---: | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `5160` | `hidden_collapse`, `role_collapse` | `high_sim_collapsed`, `verified_hidden_collapse` | `0.3671` | `0.4531` | `+0.0860` | `[-0.0313, +0.2148]` | 5 | 1 | 2 | `0.6250` |
| `5115` | `role_collapse` | `high_sim_collapsed` | `0.3634` | `0.4220` | `+0.0586` | `[-0.0508, +0.1758]` | 6 | 1 | 1 | `0.7500` |
| `5121` | `role_collapse` | `high_sim_collapsed` | `0.4845` | `0.4806` | `-0.0039` | `[-0.0351, +0.0196]` | 3 | 2 | 3 | `0.3750` |
| `5051` | `role_collapse` | none | `0.4649` | `0.5549` | `+0.0900` | `[-0.1016, +0.2774]` | 4 | 4 | 0 | `0.5000` |

## Interpretation

The role-head advantage remains positive on average, but the stronger claim did
not survive cleanly after widening from two to four generated games.

Important points:

1. The expanded cell remains role-head-positive in mean win rate:

   ```text
   +0.0577 pooled seed delta
   ```

2. The pooled seed-level interval now crosses zero:

   ```text
   [-0.0048, +0.1242]
   ```

   That makes the result suggestive rather than decisive.

3. Three of four generated games are positive by entry mean, but seed signs are
   not highly stable:

   ```text
   18 positive seed deltas
   8 negative seed deltas
   6 zero seed deltas
   seed sign stability = 0.5625
   ```

4. The newly added seed `5121` is essentially neutral, while seed `5051` is
   positive on average but high variance. That implies the original two-entry
   signal was not simply false, but it was too narrow.

5. Role heads reduce draw rate slightly in this cell:

   ```text
   shared draw = 0.2265
   role-head draw = 0.1973
   ```

   This may be part of the architecture effect: role-conditioned heads may be
   better at converting certain escape-capture collapse positions into decisive
   play, but the effect is not uniform across generated games.

## Research Consequence

This is a stronger scientific result than a clean positive headline would have
been.

The evidence now supports this claim:

> AsymBench can identify asymmetric game strata where architecture effects are
> directionally reproducible but heterogeneous across generated instances.

That matters because the benchmark is not merely ranking agents. It is exposing
when a result depends on:

- game family,
- asymmetric stratum,
- generated instance,
- training seed,
- and draw pressure.

For a high-esteem paper, this pushes the story toward **stratified evaluator
stress testing**, not a simple claim that role-head AlphaZero-lite is better.

## Next Step

The next best step is to add a paper-facing analysis artifact:

```text
input = stability and deepening summary JSON files
output = markdown/csv tables for:
  - cell-level deltas
  - bootstrap intervals
  - sign stability
  - draw-rate shifts
  - entry heterogeneity
```

After that, the first LLM-agent experiment should use a deliberately small game
set:

```text
positive exemplar: escape_capture seed 5160 or 5115
neutral exemplar: escape_capture seed 5121
high-variance exemplar: escape_capture seed 5051
control exemplar: clean connection_disruption seed 8003 or 9109
```

That would let us test whether LLM agents show the same stratum/instance
sensitivity as AlphaZero-lite, rather than jumping directly to broad LLM
evaluation.
