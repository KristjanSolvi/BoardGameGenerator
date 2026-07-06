# Role-Head Targeted Scale-Up Results

Date: 2026-07-06
Branch: `research/asymbench`

## Purpose

The first 10-config role-head pilot was useful but too short to support an
architecture claim. It suggested that role heads might fail badly on
role-inversion and collapse strata, but that could have been sparse self-play
noise.

This pass ran a heavier targeted scale-up on the three most informative buckets:

```text
clean
role_inversion
collapse
```

The experiment asks:

> Does role-head/shared-head architecture sensitivity remain stratum-dependent
> under a larger AlphaZero-lite budget?

## Setup

The targeted pilot was generated from the two committed benchmark manifests:

```bash
python -m research.asymbench.experiments.prepare_manifest_pilot \
  --manifest docs/research/2026-07-05-connection-disruption-benchmark-manifest.json \
  --manifest docs/research/2026-07-05-escape-capture-benchmark-manifest.json \
  --output-root research_runs/asymbench/role_head_scale_20260705 \
  --template research_runs/asymbench/role_head_scale_20260705/heavy_template.json \
  --per-bucket-per-family 2 \
  --bucket clean \
  --bucket collapse \
  --bucket role_inversion
```

The heavy template used:

```text
device = cuda
seeds = [101, 202, 303, 404, 505]
iterations = 6
selfplay_games_per_iteration = 8
train_steps_per_iteration = 16
mcts_simulations = 12
eval_games = 16
eval_simulations = 12
batch_size = 32
replay_capacity = 4096
```

This produced 12 configs:

```text
3 buckets * 2 families * 2 entries per family = 12 configs
```

The CUDA run completed all 12 configs in about 6.8 hours:

```text
completed = 12 / 12
elapsed = 24526.3 seconds
```

Final aggregation command:

```bash
python -m research.asymbench.analysis.summarize_manifest_pilot \
  research_runs/asymbench/role_head_scale_20260705/pilot_manifest.json \
  --output research_runs/asymbench/role_head_scale_20260705/scale_summary_final.json
```

## Bucket Results

`architecture_delta` means:

```text
role_heads mean model win - shared_heads mean model win
```

| Bucket | Entries | Shared Win | Role-Head Win | Delta | Shared Draw | Role-Head Draw |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `clean` | 4 | `0.4750` | `0.4626` | `-0.0125` | `0.2062` | `0.2094` |
| `collapse` | 4 | `0.4656` | `0.4688` | `+0.0032` | `0.1313` | `0.0936` |
| `role_inversion` | 4 | `0.3281` | `0.4031` | `+0.0750` | `0.1312` | `0.1656` |

## Family Results

| Family | Entries | Shared Win | Role-Head Win | Delta | Shared Draw | Role-Head Draw |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `connection_disruption` | 6 | `0.5082` | `0.4812` | `-0.0270` | `0.0000` | `0.0000` |
| `escape_capture` | 6 | `0.3376` | `0.4084` | `+0.0708` | `0.3125` | `0.3123` |

## Bucket-by-Family Results

| Bucket | Family | Entries | Shared Win | Role-Head Win | Delta | Shared Draw | Role-Head Draw |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `clean` | `connection_disruption` | 2 | `0.6436` | `0.5812` | `-0.0624` | `0.0000` | `0.0000` |
| `clean` | `escape_capture` | 2 | `0.3065` | `0.3439` | `+0.0374` | `0.4125` | `0.4187` |
| `collapse` | `connection_disruption` | 2 | `0.5811` | `0.4750` | `-0.1061` | `0.0000` | `0.0000` |
| `collapse` | `escape_capture` | 2 | `0.3500` | `0.4626` | `+0.1126` | `0.2626` | `0.1872` |
| `role_inversion` | `connection_disruption` | 2 | `0.3000` | `0.3875` | `+0.0875` | `0.0000` | `0.0000` |
| `role_inversion` | `escape_capture` | 2 | `0.3563` | `0.4187` | `+0.0624` | `0.2624` | `0.3311` |

## Entry Results

| Bucket | Family | Seed | Shared Win | Role-Head Win | Delta | Shared Draw | Role-Head Draw |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `clean` | `connection_disruption` | `8003` | `0.8000` | `0.5750` | `-0.2250` | `0.0000` | `0.0000` |
| `clean` | `connection_disruption` | `9109` | `0.4872` | `0.5874` | `+0.1002` | `0.0000` | `0.0000` |
| `clean` | `escape_capture` | `5073` | `0.4628` | `0.5378` | `+0.0750` | `0.1748` | `0.1500` |
| `clean` | `escape_capture` | `5145` | `0.1502` | `0.1500` | `-0.0002` | `0.6502` | `0.6874` |
| `role_inversion` | `connection_disruption` | `9198` | `0.2498` | `0.3998` | `+0.1500` | `0.0000` | `0.0000` |
| `role_inversion` | `connection_disruption` | `8002` | `0.3502` | `0.3752` | `+0.0250` | `0.0000` | `0.0000` |
| `role_inversion` | `escape_capture` | `5174` | `0.3874` | `0.3750` | `-0.0124` | `0.1750` | `0.2498` |
| `role_inversion` | `escape_capture` | `5076` | `0.3252` | `0.4624` | `+0.1372` | `0.3498` | `0.4124` |
| `collapse` | `connection_disruption` | `9263` | `0.4374` | `0.4000` | `-0.0374` | `0.0000` | `0.0000` |
| `collapse` | `connection_disruption` | `8000` | `0.7248` | `0.5500` | `-0.1748` | `0.0000` | `0.0000` |
| `collapse` | `escape_capture` | `5160` | `0.3000` | `0.4878` | `+0.1878` | `0.2876` | `0.1372` |
| `collapse` | `escape_capture` | `5115` | `0.4000` | `0.4374` | `+0.0374` | `0.2376` | `0.2372` |

## Interpretation

The scale-up changes the short-pilot interpretation.

The short pilot suggested:

```text
clean: mildly positive for role heads
role_inversion: strongly negative for role heads
collapse: negative for role heads
```

The heavier run suggests:

```text
clean: neutral
role_inversion: modestly positive for role heads
collapse: neutral overall, but family-dependent
```

Important points:

1. The strong negative role-inversion result did not survive scale-up.
   Role-inversion became the clearest positive bucket at `+0.0750`.

2. The clean bucket no longer supports a role-head advantage.
   It averaged `-0.0125`, effectively neutral at this sample size.

3. Collapse is not one phenomenon across families.
   `connection_disruption` collapse favored shared heads by `-0.1061`, while
   `escape_capture` collapse favored role heads by `+0.1126`.

4. Family effects remain visible.
   `connection_disruption` stayed slightly shared-head-favored overall
   (`-0.0270`), while `escape_capture` became role-head-favored (`+0.0708`) but
   with much higher draw pressure.

5. Draw pressure is not just noise.
   `escape_capture` clean and role-inversion entries still had substantial draw
   rates, while `connection_disruption` had zero draw rate in this scale-up.
   That supports treating horizon/decisiveness as a separate axis in the paper.

## Research Consequence

This strengthens the novelty story:

> AsymBench is not a benchmark for proving one architecture is universally
> better. It is a generator-and-selector pipeline for exposing where evaluators
> and learning architectures change behavior across role, family, seat, and
> horizon strata.

The current result is publishable as an early empirical pattern, not as a final
architecture conclusion:

- Short-run architecture deltas can be misleading.
- Scale-up can reverse a stratum-level conclusion.
- Stratum and family labels make those reversals measurable instead of anecdotal.

## Next Step

The next high-value experiment is not another broad sweep. It is a stability
probe on the four most informative cells:

| Cell | Reason |
| --- | --- |
| `role_inversion::connection_disruption` | Positive after scale-up and no draw confound. |
| `collapse::connection_disruption` | Negative after scale-up and no draw confound. |
| `collapse::escape_capture` | Positive after scale-up but with draw reduction. |
| `clean::connection_disruption` | Mixed signs across two clean seeds. |

Recommended stability probe:

```text
entries per cell = 2
training seeds = 8 to 10
iterations = 6
selfplay_games_per_iteration = 8
train_steps_per_iteration = 16
mcts_simulations = 12
eval_games = 24
eval_simulations = 12
```

That run should estimate variance well enough to decide whether the next paper
claim is about role-head architecture, evaluator instability, or family-specific
RL learnability.
