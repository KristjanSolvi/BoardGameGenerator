# Escape/Capture Benchmark Manifest

Date: 2026-07-05
Branch: `research/asymbench`

## Purpose

This pass formalizes `escape_capture` as the second benchmark family in
AsymBench. The goal is not just to add more generated games. It is to add a
family with a different evaluator failure profile from `connection_disruption`.

`connection_disruption` is currently strongest as a role-collapse family:
planning often finds builder wins that random play misses. `escape_capture`
instead gives many seat-sensitive and horizon-sensitive pursuit states, plus a
smaller number of clean controls, role collapses, and role inversions.

That gives the project a stronger benchmark story:

> AsymBench should compare agents across generated asymmetric families whose
> role, seat, horizon, and evaluator-disagreement pathologies are explicitly
> selected rather than sampled uniformly.

## Artifact

Committed manifest:

```text
docs/research/2026-07-05-escape-capture-benchmark-manifest.json
```

The manifest embeds each selected `spec.json`, records first-pass MCTS metrics,
and attaches 64-simulation verification metrics for all selected entries.

## Data Source

The source pool was the archived July 3 `escape_capture` sweep:

```text
research_runs/asymbench/deep_sweep_20260703/generated
```

That sweep had 250 valid `escape_capture` specs, but its validation files were
random-only. This pass revalidated the same 250 specs with:

```text
random_games = 32
mcts_games = 12
mcts_simulations = 16
```

The local first-pass output was:

```text
research_runs/asymbench/escape_capture_manifest_20260705/mcts16_pool
```

The selected union of manifest entries was then rechecked with:

```text
random_games = 32
mcts_games = 12
mcts_simulations = 64
```

The local high-simulation report was:

```text
research_runs/asymbench/escape_capture_manifest_20260705/escape_high_sim_mcts64.json
```

## First-Pass Strata

The 250-game pool produced this first-pass label distribution:

| Stratum | Count |
| --- | ---: |
| `clean_control` | 24 |
| `hidden_collapse` | 2 |
| `role_collapse` | 7 |
| `role_inversion` | 37 |
| `seat_confound` | 121 |
| `horizon_stress` | 55 |
| Unlabeled | 72 |

Board-size split:

| Stratum | 5x5 | 6x6 | 7x7 |
| --- | ---: | ---: | ---: |
| `clean_control` | 12 | 9 | 3 |
| `seat_confound` | 18 | 49 | 54 |
| `horizon_stress` | 0 | 18 | 37 |
| `hidden_collapse` | 0 | 2 | 0 |
| `role_collapse` | 5 | 2 | 0 |
| `role_inversion` | 8 | 17 | 12 |

This matches the earlier qualitative diagnosis: larger `escape_capture` boards
create much more horizon pressure, and the family is far more seat-confounded
than `connection_disruption`.

## Verification Labels

The final manifest uses a tighter high-simulation `strict_clean` threshold than
the earlier connection manifest:

```text
strict_role_bias <= 0.25
strict_seat_bias <= 0.20
strict_mcts_max_ply_rate <= 0.50
```

The broader `near_clean` label remains useful for candidates that are not clean
enough for a control set but deserve follow-up.

Across the 39 unique selected entries, the 64-simulation verification labels
were:

| Verification Label | Unique Entries |
| --- | ---: |
| `strict_clean` | 2 |
| `near_clean` | 4 |
| `seat_sensitive` | 27 |
| `high_sim_collapsed` | 5 |
| `verified_hidden_collapse` | 1 |
| `verified_role_inversion` | 2 |

Strong examples:

| Seed | First-Pass Stratum | 64-Sim Label | Why It Matters |
| --- | --- | --- | --- |
| `5073` | `clean_control` | `strict_clean` | Best current clean-control candidate in this family. |
| `5160` | `hidden_collapse`, `role_collapse` | `high_sim_collapsed`, `verified_hidden_collapse` | Random-play balance was misleading; stronger MCTS finds attacker collapse. |
| `5009` | `horizon_stress`, `seat_confound` | `seat_sensitive` | Illustrates the family-specific horizon/seat confound. |
| `5145` | `role_inversion` | `strict_clean` | First-pass inversion that becomes clean under stronger MCTS. |
| `5076` | `role_inversion` | `seat_sensitive`, `verified_role_inversion` | Role-inversion signal persists, but with a seat confound. |
| `5174` | `role_inversion` | `high_sim_collapsed`, `verified_role_inversion` | Skill changes the apparent favored role and then reveals collapse. |

## Interpretation

This gives us a two-family benchmark contrast:

| Family | Current Best Use |
| --- | --- |
| `connection_disruption` | Role-collapse and hidden-collapse stress tests for planning/RL agents. |
| `escape_capture` | Seat-confound, horizon-stress, and pursuit/escape sensitivity tests. |

This matters for novelty because nearby work already covers broad executable
game generation, LLM-assisted board-game design, and single-game balancing.
The more defensible AsymBench claim is:

> We generate asymmetric game families and select benchmark instances by
> evaluator-conditioned role, seat, and horizon diagnostics, then test whether
> MCTS, RL/self-play, LLM agents, and humans fail on the same strata.

`escape_capture` strengthens that claim because it shows the selector is not
only rediscovering one builder-favored pathology. A second family produces a
different distribution of failures, so the benchmark can ask whether an agent is
robust to the type of asymmetry rather than merely good at one game template.

## Next Research Step

The next useful step is to run the same manifest entries through the
AlphaZero-lite shared-head versus role-head evaluator, grouped by stratum:

| Group | Question |
| --- | --- |
| `strict_clean` and `near_clean` | Do architectures agree when role and seat are controlled? |
| `seat_sensitive` | Do role-aware policies confuse first-player advantage with role advantage? |
| `horizon_stress` | Does longer planning reduce max-ply artifacts, or does training learn to exploit them? |
| `high_sim_collapsed` | Does self-play rediscover the MCTS collapse, and how quickly? |
| `verified_role_inversion` | Do agents reverse their apparent role preference as evaluator strength increases? |

For the professor's suggested tractable RL direction, this manifest is the
controlled slice we need before scaling: the same generated specs can be run by
random agents, MCTS, AlphaZero-lite, LLM agents, and eventually humans under
identical role/seat assignments.
