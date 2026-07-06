# Role-Head Stability Probe Results

Date: 2026-07-06
Branch: `research/asymbench`
Tooling commit: `f8ecda4`

## Purpose

The targeted scale-up suggested that role-head/shared-head differences were
cell-dependent, but it still averaged over only five training seeds per config.
This probe asks whether the most important cells remain stable when each
generated game is rerun with more training seeds and seed-level aggregation.

The central question is:

> Are role-head architecture deltas stable within specific asymmetric game
> strata, or are apparent cell-level effects still training-seed noise?

## Setup

The probe used exact `bucket::family` cells rather than all buckets:

```text
role_inversion::connection_disruption
collapse::connection_disruption
collapse::escape_capture
clean::connection_disruption
```

Each cell selected two generated games from the committed benchmark manifests,
for eight configs total.

Template:

```text
device = cuda
seeds = [101, 202, 303, 404, 505, 606, 707, 808]
iterations = 6
selfplay_games_per_iteration = 8
train_steps_per_iteration = 16
mcts_simulations = 12
eval_games = 24
eval_simulations = 12
batch_size = 32
replay_capacity = 4096
```

Generation command:

```bash
python -m research.asymbench.experiments.prepare_manifest_pilot \
  --manifest docs/research/2026-07-05-connection-disruption-benchmark-manifest.json \
  --manifest docs/research/2026-07-05-escape-capture-benchmark-manifest.json \
  --output-root research_runs/asymbench/role_head_stability_20260706 \
  --template research_runs/asymbench/role_head_stability_20260706/stability_template.json \
  --per-bucket-per-family 2 \
  --cell role_inversion::connection_disruption \
  --cell collapse::connection_disruption \
  --cell collapse::escape_capture \
  --cell clean::connection_disruption
```

Aggregation command:

```bash
python -m research.asymbench.analysis.summarize_manifest_pilot \
  research_runs/asymbench/role_head_stability_20260706/pilot_manifest.json \
  --output research_runs/asymbench/role_head_stability_20260706/stability_summary_final.json
```

The CUDA run completed all eight configs:

```text
completed = 8 / 8
elapsed = 34353 seconds
elapsed_hours = 9.54
```

## Cell Results

`architecture_delta` means:

```text
role_heads mean model win - shared_heads mean model win
```

The seed-level CI pools the 16 paired training-seed deltas in each cell
(`2 generated games * 8 training seeds`).

| Cell | Entries | Shared Win | Role-Head Win | Entry Delta | Pooled Seed Delta | Seed CI95 | Positive Seeds | Negative Seeds | Seed Sign Stability |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| `role_inversion::connection_disruption` | 2 | `0.2578` | `0.3334` | `+0.0756` | `+0.0756` | `[-0.0469, +0.2136]` | 8 | 5 | `0.5000` |
| `collapse::connection_disruption` | 2 | `0.5260` | `0.5130` | `-0.0130` | `-0.0130` | `[-0.1718, +0.1303]` | 9 | 6 | `0.5625` |
| `collapse::escape_capture` | 2 | `0.3568` | `0.4479` | `+0.0911` | `+0.0911` | `[+0.0001, +0.1773]` | 11 | 3 | `0.6875` |
| `clean::connection_disruption` | 2 | `0.6248` | `0.5859` | `-0.0389` | `-0.0389` | `[-0.1689, +0.0940]` | 6 | 9 | `0.5625` |

## Entry Results

| Cell | Seed | Shared Win | Role-Head Win | Delta | Seed CI95 | Positive Seeds | Negative Seeds | Seed Sign Stability |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| `role_inversion::connection_disruption` | `9198` | `0.2188` | `0.3176` | `+0.0989` | `[-0.0730, +0.3024]` | 4 | 3 | `0.5000` |
| `role_inversion::connection_disruption` | `8002` | `0.2969` | `0.3491` | `+0.0523` | `[-0.1250, +0.2294]` | 4 | 2 | `0.5000` |
| `collapse::connection_disruption` | `9263` | `0.4114` | `0.4010` | `-0.0104` | `[-0.1820, +0.1354]` | 4 | 4 | `0.5000` |
| `collapse::connection_disruption` | `8000` | `0.6406` | `0.6250` | `-0.0156` | `[-0.2449, +0.2084]` | 5 | 2 | `0.6250` |
| `collapse::escape_capture` | `5160` | `0.3543` | `0.4636` | `+0.1094` | `[-0.0053, +0.2345]` | 6 | 1 | `0.7500` |
| `collapse::escape_capture` | `5115` | `0.3593` | `0.4321` | `+0.0729` | `[-0.0470, +0.2033]` | 5 | 2 | `0.6250` |
| `clean::connection_disruption` | `8003` | `0.7081` | `0.5781` | `-0.1300` | `[-0.3384, +0.1043]` | 2 | 6 | `0.7500` |
| `clean::connection_disruption` | `9109` | `0.5415` | `0.5938` | `+0.0523` | `[-0.0726, +0.1669]` | 4 | 3 | `0.5000` |

## Interpretation

The stability probe narrows the architecture story.

The previous scale-up suggested:

```text
role_inversion::connection_disruption: role-head positive
collapse::connection_disruption: shared-head positive
collapse::escape_capture: role-head positive
clean::connection_disruption: mixed
```

After seed-level probing:

1. `collapse::escape_capture` is the strongest surviving role-head-positive
   cell. Both generated games are positive, 11 of 16 paired seed deltas are
   positive, and the pooled bootstrap interval is barely above zero.

2. `role_inversion::connection_disruption` remains positive at the entry-mean
   level, but the seed-level interval crosses zero and sign stability is weak.
   It is a promising cell, not yet a robust architecture result.

3. `collapse::connection_disruption` no longer supports a strong shared-head
   advantage. Both entry means are slightly negative for role heads, but pooled
   seed signs are mixed and the interval is wide.

4. `clean::connection_disruption` remains a useful control because the two
   generated games point in opposite directions.

5. Draw pressure is cell-specific. The connection-disruption cells have zero
   draw rate in this probe, while `collapse::escape_capture` remains draw-heavy
   but role-head positive.

## Research Consequence

This makes the novelty claim sharper:

> The benchmark is not just finding asymmetric games. It is separating apparent
> evaluator/architecture effects from seed-stable effects across named
> asymmetric game strata.

The current evidence supports a cautious claim:

- AsymBench can produce tractable asymmetric game cells where architecture
  comparisons are measurable.
- Some apparent architecture effects shrink or become unstable under additional
  training seeds.
- `collapse::escape_capture` is currently the best candidate for a robust
  role-head-positive result.
- `role_inversion::connection_disruption` remains valuable as a high-variance
  stress test.

## Next Step

The next experiment should not broaden immediately. It should deepen the one
cell that survived best:

```text
cell = collapse::escape_capture
entries = 4 to 6 generated games
training seeds = 8
same training budget
eval_games = 32
```

That would test whether the positive role-head effect is a property of the
`collapse::escape_capture` family/stratum interaction rather than just two
selected generated games.
