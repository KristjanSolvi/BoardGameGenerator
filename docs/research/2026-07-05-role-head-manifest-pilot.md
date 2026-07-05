# Manifest-Driven Role-Head Pilot

Date: 2026-07-05
Branch: `research/asymbench`

## Purpose

This pass connects the benchmark manifests to the AlphaZero-lite evaluator.
The immediate goal is a tractable RL experiment that follows the project story
we have been building:

> Compare shared-head and role-head self-play on generated asymmetric games
> selected by role, seat, horizon, and evaluator-disagreement strata.

This matters because the two committed manifests now expose different failure
profiles:

| Family | Main Stress Profile |
| --- | --- |
| `connection_disruption` | Role collapse and hidden collapse. |
| `escape_capture` | Seat sensitivity and horizon pressure. |

The pilot layer makes those strata runnable by `run_role_heads.py` without
depending on transient `research_runs` spec paths from earlier sweeps.

## Implementation

Added:

```text
research/asymbench/experiments/prepare_manifest_pilot.py
```

The script reads committed benchmark manifests with embedded specs, selects a
small cross-family pilot set, writes stable local spec files, and emits
`run_role_heads.py` configs.

Default pilot buckets:

| Bucket | Selection Preference |
| --- | --- |
| `clean` | `strict_clean`, then `near_clean`, then `clean_control`. |
| `role_inversion` | `verified_role_inversion`, then `role_inversion`. |
| `collapse` | `verified_hidden_collapse`, then `high_sim_collapsed`, then collapse strata. |
| `seat_sensitive` | `seat_sensitive`, then `seat_confound`. |
| `horizon_stress` | `horizon_stress`, ranked by max-ply pressure. |

The default run template is intentionally a pilot, not a final experiment:

```text
seeds = [101, 202]
iterations = 3
selfplay_games_per_iteration = 4
train_steps_per_iteration = 8
mcts_simulations = 8
eval_games = 8
eval_simulations = 8
device = cuda
```

## Command

The pilot config set was generated with:

```bash
python -m research.asymbench.experiments.prepare_manifest_pilot \
  --manifest docs/research/2026-07-05-connection-disruption-benchmark-manifest.json \
  --manifest docs/research/2026-07-05-escape-capture-benchmark-manifest.json \
  --output-root research_runs/asymbench/role_head_manifest_pilot_20260705 \
  --per-bucket-per-family 1
```

This produced 10 runnable configs: one per bucket per family.

## Selected Pilot Entries

| Bucket | Family | Seed | Labels | Metrics |
| --- | --- | ---: | --- | --- |
| `clean` | `connection_disruption` | `8003` | `strict_clean` | role `0.000`, seat `0.000`, max `0.500` |
| `clean` | `escape_capture` | `5073` | `strict_clean` | role `0.084`, seat `0.166`, max `0.250` |
| `role_inversion` | `connection_disruption` | `9198` | `high_sim_collapsed`, `verified_role_inversion` | role `1.000`, seat `0.000`, inversion `0.688` |
| `role_inversion` | `escape_capture` | `5174` | `high_sim_collapsed`, `verified_role_inversion` | role `0.834`, seat `0.166`, inversion `0.573` |
| `collapse` | `connection_disruption` | `9263` | `high_sim_collapsed` | role `0.834`, seat `0.166`, max `0.083` |
| `collapse` | `escape_capture` | `5160` | `high_sim_collapsed`, `verified_hidden_collapse` | role `1.000`, seat `0.000`, max `0.000` |
| `seat_sensitive` | `connection_disruption` | `9180` | `seat_sensitive` | role `0.334`, seat `0.334`, max `0.333` |
| `seat_sensitive` | `escape_capture` | `5040` | `seat_sensitive` | role `0.000`, seat `1.000`, max `1.000` |
| `horizon_stress` | `connection_disruption` | `9004` | none | role `0.166`, seat `0.166`, max `0.583` |
| `horizon_stress` | `escape_capture` | `5091` | `seat_sensitive` | role `0.000`, seat `1.000`, max `1.000` |

## End-to-End Smoke Result

One config was run on CUDA to verify the pipeline:

```bash
python -m research.asymbench.experiments.run_role_heads \
  --config research_runs/asymbench/role_head_manifest_pilot_20260705/configs/clean/escape_capture/escape_capture_5073_escape_capture_5x5_seed_5073.json \
  --device cuda
```

Run directory:

```text
research_runs/asymbench/role_head_manifest_pilot_20260705/runs/escape_capture_5x5_seed_5073_20260705_185715_508420
```

Final two-seed result:

| Variant | Mean Model Win | Mean Random Win | Draw Rate | Avg Plies |
| --- | ---: | ---: | ---: | ---: |
| `shared_heads` | `0.3125` | `0.3750` | `0.3125` | `52.875` |
| `role_heads` | `0.5625` | `0.3125` | `0.1250` | `33.315` |

This is not evidence for a paper claim yet. It only verifies that the
manifest-selected generated specs can flow through the role-head evaluator and
produce architecture-comparable metrics on GPU.

## Next Experiment

Run all 10 pilot configs and aggregate by bucket:

```text
role_head_delta = final_eval_model_win_rate(role_heads)
                - final_eval_model_win_rate(shared_heads)
```

The first research question should be:

> Is architecture delta stratum-dependent?

Expected useful outcomes:

| Outcome | Interpretation |
| --- | --- |
| Role heads help mainly on collapse/inversion seeds. | Role-specific value/policy heads may help when role objectives are strategically divergent. |
| Role heads do not help on clean controls. | Good; it means the architecture difference is not just a universal capacity advantage. |
| Seat-sensitive seeds remain unstable for both architectures. | We need explicit seat controls or evaluator protocols before human/LLM comparison. |
| Horizon-stress seeds produce high draw/max-ply rates under self-play. | The family needs horizon-aware acceptance thresholds or longer training/evaluation budgets. |

This is the tractable RL bridge to the professor's `asymbench` direction before
adding LLM agents or human playtests.
