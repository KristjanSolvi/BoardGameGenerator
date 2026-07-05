# Role-Head Manifest Pilot Results

Date: 2026-07-05
Branch: `research/asymbench`

## Purpose

This pass ran the full 10-config manifest-selected AlphaZero-lite pilot on CUDA.
The goal was to test whether role-head versus shared-head architecture effects
depend on benchmark stratum:

```text
architecture_delta = role_heads mean model win
                   - shared_heads mean model win
```

This is the first cross-family RL result tied directly to the committed
AsymBench manifests.

## Run Setup

The pilot was generated from:

```text
docs/research/2026-07-05-connection-disruption-benchmark-manifest.json
docs/research/2026-07-05-escape-capture-benchmark-manifest.json
```

Command:

```bash
python -m research.asymbench.experiments.prepare_manifest_pilot \
  --manifest docs/research/2026-07-05-connection-disruption-benchmark-manifest.json \
  --manifest docs/research/2026-07-05-escape-capture-benchmark-manifest.json \
  --output-root research_runs/asymbench/role_head_manifest_pilot_20260705 \
  --per-bucket-per-family 1
```

Each selected config used:

```text
device = cuda
seeds = [101, 202]
iterations = 3
selfplay_games_per_iteration = 4
train_steps_per_iteration = 8
mcts_simulations = 8
eval_games = 8
eval_simulations = 8
```

The full 10-config run completed in about 18 minutes on the local CUDA setup.

Aggregate summary:

```bash
python -m research.asymbench.analysis.summarize_manifest_pilot \
  research_runs/asymbench/role_head_manifest_pilot_20260705/pilot_manifest.json \
  --output research_runs/asymbench/role_head_manifest_pilot_20260705/pilot_summary_full.json
```

## Bucket Results

| Bucket | Entries | Shared Win | Role-Head Win | Delta | Shared Draw | Role-Head Draw |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `clean` | 2 | `0.4063` | `0.5000` | `+0.0938` | `0.1563` | `0.0625` |
| `collapse` | 2 | `0.3125` | `0.1563` | `-0.1563` | `0.1875` | `0.1563` |
| `horizon_stress` | 2 | `0.2813` | `0.2188` | `-0.0625` | `0.5000` | `0.4063` |
| `role_inversion` | 2 | `0.5313` | `0.0625` | `-0.4688` | `0.0313` | `0.2813` |
| `seat_sensitive` | 2 | `0.2500` | `0.0625` | `-0.1875` | `0.4688` | `0.4375` |

## Family Results

| Family | Entries | Shared Win | Role-Head Win | Delta | Shared Draw | Role-Head Draw |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `connection_disruption` | 5 | `0.5000` | `0.2125` | `-0.2875` | `0.0000` | `0.0000` |
| `escape_capture` | 5 | `0.2125` | `0.1875` | `-0.0250` | `0.5375` | `0.5375` |

## Entry Results

| Bucket | Family | Seed | Shared Win | Role-Head Win | Delta | Shared Draw | Role-Head Draw |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `clean` | `connection_disruption` | `8003` | `0.5000` | `0.4375` | `-0.0625` | `0.0000` | `0.0000` |
| `clean` | `escape_capture` | `5073` | `0.3125` | `0.5625` | `+0.2500` | `0.3125` | `0.1250` |
| `role_inversion` | `connection_disruption` | `9198` | `0.5625` | `0.0000` | `-0.5625` | `0.0000` | `0.0000` |
| `role_inversion` | `escape_capture` | `5174` | `0.5000` | `0.1250` | `-0.3750` | `0.0625` | `0.5625` |
| `collapse` | `connection_disruption` | `9263` | `0.3750` | `0.1250` | `-0.2500` | `0.0000` | `0.0000` |
| `collapse` | `escape_capture` | `5160` | `0.2500` | `0.1875` | `-0.0625` | `0.3750` | `0.3125` |
| `seat_sensitive` | `connection_disruption` | `9180` | `0.5000` | `0.0625` | `-0.4375` | `0.0000` | `0.0000` |
| `seat_sensitive` | `escape_capture` | `5040` | `0.0000` | `0.0625` | `+0.0625` | `0.9375` | `0.8750` |
| `horizon_stress` | `connection_disruption` | `9004` | `0.5625` | `0.4375` | `-0.1250` | `0.0000` | `0.0000` |
| `horizon_stress` | `escape_capture` | `5091` | `0.0000` | `0.0000` | `0.0000` | `1.0000` | `0.8125` |

## Interpretation

The pilot does not support a simple claim that role heads are better. In this
short schedule, shared heads were stronger on most collapse, role-inversion, and
connection-disruption entries.

The useful signal is more specific:

1. Architecture delta appears stratum-dependent.
   The clean bucket was mildly positive for role heads, while role-inversion and
   seat-sensitive buckets were strongly negative.

2. `connection_disruption` remains a harsh role-collapse family.
   It produced no draw pressure in this pilot, but role heads underperformed
   shared heads by `-0.2875` mean win rate.

3. `escape_capture` remains horizon/seat dominated.
   Draw rates were high for both architectures, especially on seat-sensitive
   and horizon-stress seeds. That supports the need for separate horizon and
   seat strata before using humans or LLM agents.

4. The strongest positive role-head result was the clean `escape_capture` seed
   `5073`, where role heads improved from `0.3125` to `0.5625` model win rate
   and reduced draw rate from `0.3125` to `0.1250`.

5. The strongest negative role-head result was the `connection_disruption`
   role-inversion seed `9198`, where shared heads reached `0.5625` but role
   heads reached `0.0000`.

## Research Consequence

This is useful precisely because it is not a clean positive architecture story.
For a paper, the safer claim is:

> Generated asymmetric benchmark strata expose architecture-specific failure
> modes that are hidden by aggregate win rate alone.

The next experiment should not simply scale every seed equally. It should focus
on discriminating between three explanations:

| Hypothesis | Test |
| --- | --- |
| Role heads need more data to help on collapse/inversion games. | Increase seeds, iterations, and self-play games on `collapse` and `role_inversion`. |
| Shared heads regularize better under sparse short self-play. | Compare equal parameter budgets and longer schedules. |
| Seat/horizon confounds dominate `escape_capture`. | Add explicit seat-conditioned reporting and longer-horizon evaluation. |

Recommended next run:

```text
target buckets = clean, collapse, role_inversion
seeds = 5
iterations = 6
selfplay_games_per_iteration = 8
train_steps_per_iteration = 16
mcts_simulations = 12
eval_games = 16
eval_simulations = 12
```

This would be the first scale-up run worth treating as more than a smoke pilot.
