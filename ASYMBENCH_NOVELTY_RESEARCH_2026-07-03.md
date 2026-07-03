# AsymBench Novelty Research Memo

Date: 2026-07-03

Context: Hafsteinn pointed to two very relevant papers:

- [RuleSmith: Multi-Agent LLMs for Automated Game Balancing](https://arxiv.org/html/2602.06232v1)
- [MeepleLM: A Virtual Playtester Simulating Diverse Subjective Experiences](https://aclanthology.org/2026.acl-long.850/) / [arXiv version](https://arxiv.org/abs/2601.07251)

His core suggestion was: find what they did not do, and decide whether "AsymBench" is still interesting if we can make the RL experiments tractable.

This memo is a focused novelty analysis for that question. It assumes the current `gamegen` project already has a symmetric game-generation pipeline with executable engines, validation, random/flat-Monte-Carlo playtesting, critic prompts, novelty review, and archived run logs.

## Executive Takeaway

The most defensible novel direction is not "automated game balancing" and not "virtual board-game playtesting". RuleSmith and MeepleLM cover those spaces well.

The stronger gap is:

> Generate small, executable asymmetric board games as benchmark tasks, then measure how different kinds of agents learn, exploit, and misunderstand the two roles.

That gives us a possible "AsymBench" contribution:

> AsymBench: a generated benchmark suite of mechanically validated asymmetric abstract games, with role-aware balance metrics, RL/search baselines, LLM play, and human or persona-based subjective calibration.

The important twist is that asymmetry becomes the experimental variable. We do not merely ask "is the game balanced?" We ask:

- Which role is stronger?
- Which role is harder to learn?
- Does a shared policy fail compared with role-specific heads?
- Do LLMs make different rule-tracking errors by role?
- Do humans and LLMs disagree about which role is fun, fair, or understandable?
- Does a ruleset balanced for LLM self-play stay balanced for MCTS, RL, and humans?

That is different enough from RuleSmith and MeepleLM to be worth pursuing.

## What RuleSmith Does

RuleSmith introduces a framework for automated balancing of an asymmetric game using LLM self-play plus Bayesian optimization. It instantiates the method on CivMini, a small Civilization-inspired game with two asymmetric factions: Empire and Nomads.

Important details:

- The game is hand-authored and parameterized, not generated from scratch.
- The asymmetric factions have different units, economies, incentives, mobility, and combat profiles.
- The optimizer searches over rule parameters such as initial resources, gather amounts, kill rewards, damage, HP, production costs, and scoring weights.
- LLM agents read rulebooks and structured game states, then output JSON actions.
- The engine validates actions and falls back to safe defaults when the LLM output is invalid.
- The objective is balance: reduce faction win-rate disparity and avoid excessive draws.
- The method uses Bayesian optimization with acquisition-based adaptive sampling to spend more evaluations on promising candidates.
- Their experiments use model pairings such as InternVL3.5-2B and InternVL3.5-8B.
- They report that optimized parameters can reach near 50/50 win rates for the evaluated LLM pairings.

RuleSmith's own limitations are useful for us:

- LLM agents are imperfect proxies for humans and may encode model-specific biases.
- The experiments use one simplified, fully observable environment with relatively small action spaces.
- It does not model partial observability, stochasticity, long-term uncertainty, or rich social interactions.
- It does not provide formal robustness guarantees under distribution shift.
- It balances parameters for the evaluator pair; those parameters may not transfer to other agents or humans.

Source: [RuleSmith paper](https://arxiv.org/html/2602.06232v1), [RuleSmith GitHub](https://github.com/Adonis-galaxy/RuleSmith)

## What RuleSmith Does Not Do

This is the first serious novelty opening.

RuleSmith does not:

- generate multiple new games;
- produce a benchmark suite;
- compare generated asymmetric games against symmetric generated games;
- use asymmetry as a controlled independent variable;
- characterize role-specific learnability;
- test shared-policy versus role-specific-policy RL across many games;
- estimate whether LLM-balanced parameters transfer to search agents, RL agents, or humans;
- provide human playtesting as a validation loop;
- build a contamination-resistant benchmark from fresh generated games;
- separate role strength from role cognitive difficulty;
- use generated game families to study which kinds of asymmetry are hard for which agents.

RuleSmith answers:

> Given a hand-authored asymmetric game, can LLM self-play and Bayesian optimization tune parameters toward balance?

AsymBench could answer:

> Can we generate many asymmetric games whose role structure creates measurable differences in agent learning, planning, and rule understanding?

Those are related but not the same contribution.

## What MeepleLM Does

MeepleLM is a virtual playtester for board-game feedback. It targets subjective player experience, not mechanical self-play strength.

Important details from the paper:

- It focuses on critique grounded in emergent user experience.
- It treats board-game critique as difficult because models must infer latent gameplay dynamics from rules without an explicit engine.
- It also models subjective heterogeneity across player groups.
- The dataset contains 1,727 structurally corrected rulebooks and 150K reviews.
- It augments the data with Mechanics-Dynamics-Aesthetics reasoning.
- It distills player personas and trains a model to simulate persona-specific feedback.
- The authors report that MeepleLM outperforms commercial models in community alignment and critique quality, including a 70% preference rate in user studies.
- The arXiv entry says it is an ACL 2026 Main Conference paper.

Source: [MeepleLM arXiv](https://arxiv.org/abs/2601.07251), [ACL Anthology entry](https://aclanthology.org/2026.acl-long.850/)

## What MeepleLM Does Not Do

This is the second serious novelty opening.

MeepleLM does not:

- mechanically execute the game;
- validate legal move generation;
- produce objective balance or learnability metrics;
- train RL agents;
- test LLMs as actual players under engine constraints;
- separate subjective fun from objective role advantage;
- generate new asymmetric games as benchmark tasks;
- measure whether different personas correspond to different strategic policies;
- compare persona feedback against observed play traces in a generated game.

MeepleLM answers:

> Can we simulate diverse subjective player feedback from rulebooks and reviews?

AsymBench could answer:

> Can executable generated games connect subjective role experience to measurable agent behavior?

This gives us a useful bridge: after mechanical filtering and agent experiments, use a MeepleLM-style persona layer to predict which role feels fair, frustrating, complex, or exciting. Then compare that against human feedback if we can run even a small study.

## Nearby Benchmark Work

The benchmark space is crowded. We need to be precise.

Relevant examples:

- [GameBench](https://arxiv.org/abs/2406.06613) evaluates strategic reasoning in nine game environments and finds that LLMs remain below human performance in many settings.
- [Board Game Arena](https://arxiv.org/abs/2508.03368) wraps OpenSpiel games for evaluating LLMs, random agents, humans, and RL agents.
- [TextArena](https://arxiv.org/abs/2504.11442) provides many text-based games and ratings for LLM agents.
- [DSGBench](https://arxiv.org/abs/2503.06047) evaluates LLM-based agents in complex strategic games with decision tracking.
- Werewolf-style and hidden-role benchmarks test social deduction and role-specific metrics.

What they mostly do not do:

- generate fresh games as benchmark instances;
- treat asymmetry as a controlled generated variable;
- report role-aware RL learnability across generated games;
- compare role advantage, role cognitive load, and role-specific LLM error rates;
- evaluate whether benchmark games remain balanced across random, search, RL, LLM, and human players.

So AsymBench should not claim "first game benchmark for LLM agents". It should claim a narrower benchmark property:

> generated, executable, role-aware asymmetric benchmark games with tractable RL/search baselines.

## Nearby RL Work

The most relevant RL result is [Reproducing AlphaZero on Tablut](https://arxiv.org/abs/2604.05476), because Tablut is an asymmetric historical board game with unequal pieces and different objectives. The authors modify AlphaZero with separate policy and value heads for attacker and defender while sharing a residual trunk. They report training instability, especially catastrophic forgetting between roles, and stabilize learning with board augmentation, a larger replay buffer, and games against past checkpoints.

Why this matters:

- It gives us a concrete RL hypothesis for asymmetric generated games.
- It supports Hafsteinn's "tractable RL experiments" condition.
- It suggests that role-specific heads are not just implementation detail; they are a measurable consequence of asymmetric game structure.

Other useful infrastructure:

- [OpenSpiel](https://arxiv.org/abs/1908.09453) supports many game types and common RL/search algorithms.
- [PettingZoo](https://arxiv.org/abs/2009.14471) provides a multi-agent environment API.
- [Pgx](https://arxiv.org/abs/2303.17503) shows how JAX/vectorized board-game environments can make RL experiments much faster.

The practical point: we do not need to train large AlphaZero systems first. We can design generated games to be small enough for lightweight RL and MCTS to run locally or on modest GPUs.

## Strongest Novelty Candidate

### Candidate A: AsymBench as a Generated Benchmark Suite

Core idea:

> Generate many small asymmetric abstract games, mechanically validate them, and use them as a benchmark for role-conditioned strategic reasoning.

Novelty:

- generated games, not fixed public games;
- asymmetric roles by construction;
- role-aware balance metrics;
- LLM, search, RL, and human/persona comparison;
- contamination resistance through freshly generated held-out games.

Risk:

- If we only generate games and run random/MCTS, it may look like a small extension of existing game-generation work.
- We need a clear evaluation question beyond "the games exist".

Verdict:

Strong if paired with RL or human/LLM evaluation.

### Candidate B: Role-Specific Learnability as the Main Contribution

Core idea:

> Asymmetric games are not only imbalanced in win rate; their roles may have different learning curves, policy representations, and failure modes.

Experiments:

- train shared-head versus role-specific-head agents;
- compare learning curves by role;
- measure role interference and forgetting;
- compare generated symmetric games, mild asymmetric games, and strong asymmetric games;
- correlate asymmetry profile with RL instability.

Novelty:

- directly builds on the Tablut AlphaZero insight but generalizes it to generated game families;
- makes "tractable RL" central;
- produces concrete plots and metrics that are publishable even without a large human study.

Risk:

- Requires implementing an RL pipeline or exporting games to an RL framework.
- Generated games must be simple and standardized enough for training.

Verdict:

Probably the strongest technical angle.

### Candidate C: Cross-Evaluator Balance Transfer

Core idea:

> A game balanced for one evaluator may be imbalanced for another. Test whether parameters balanced by LLM self-play transfer to MCTS, RL, and humans.

This directly targets a RuleSmith limitation.

Experiments:

- balance generated asymmetric games using random/MCTS or LLM self-play;
- evaluate the same rules with RL agents and humans;
- estimate balance transfer matrix:

| Optimized for | Evaluated by random | MCTS | RL | LLM | humans |
|---|---:|---:|---:|---:|---:|
| random | | | | | |
| MCTS | | | | | |
| LLM | | | | | |
| human | | | | | |

Novelty:

- exposes whether LLM playtesters are reliable balancing proxies;
- connects RuleSmith and MeepleLM but does not duplicate either;
- feasible with a small number of games.

Risk:

- Human data is expensive.
- LLM evaluations can be noisy and costly.

Verdict:

Very good as a second contribution, especially if RL experiments work.

### Candidate D: Engine-Grounded Persona Playtesting

Core idea:

> Combine executable traces with subjective persona feedback: personas critique actual role experiences after simulated play, not only a static rulebook.

Novelty:

- MeepleLM infers play experience from rulebooks; we can condition feedback on actual generated play traces, illegal moves, wins/losses, and role-specific state histories.
- Could compare "rulebook-only persona critique" versus "trace-grounded persona critique".

Risk:

- More HCI/user-study oriented.
- Harder to validate without human feedback.

Verdict:

Promising, but should not be the first technical milestone.

## Recommended Thesis

The strongest thesis is:

> Asymmetric generated games create measurable role-conditioned learning and reasoning effects that are hidden by symmetric game benchmarks. We introduce AsymBench, a generator and benchmark protocol for small executable asymmetric games, and evaluate them with role-aware search, RL, LLM, and human/persona metrics.

This gives us three concrete contributions:

1. A generation pipeline for mechanically valid asymmetric games.
2. A role-aware evaluation protocol separating role strength, role difficulty, and agent skill.
3. A tractable RL/search study showing that asymmetry changes learnability and requires role-aware architectures or evaluation.

This is sharper than:

> We generate asymmetric games.

It is also safer than:

> We automate game balancing better than RuleSmith.

## Proposed Research Questions

### RQ1: Can LLM-generated asymmetric games be mechanically validated and filtered into playable benchmark tasks?

Evidence:

- number of generated candidates;
- validation pass rate;
- reasons for rejection;
- accepted game examples;
- rule clarity and novelty review;
- reproducibility from seed and logs.

### RQ2: Can role-aware metrics separate role advantage from agent skill?

Evidence:

- role-swapped tournaments;
- first-player versus role advantage;
- MCTS-vs-random by role;
- MCTS-vs-MCTS by role;
- confidence intervals over games and seeds.

### RQ3: Do asymmetric games create role-specific learnability gaps for RL agents?

Evidence:

- learning curves by role;
- shared-head vs role-specific-head comparison;
- catastrophic forgetting or interference metrics;
- sample efficiency by role;
- generated symmetric control games.

### RQ4: Do LLMs fail differently across asymmetric roles?

Evidence:

- illegal move rate by role;
- objective-confusion rate by role;
- opponent-goal misunderstanding;
- state-tracking errors;
- performance gap against MCTS/RL by role.

### RQ5: Do mechanical balance and subjective role experience diverge?

Evidence:

- persona feedback by role;
- human feedback by role;
- correlation with objective win rates, rule length, branching factor, and role complexity.

## Tractable RL Plan

We should make the RL scope intentionally modest. The point is not to train world-class agents. The point is to get reliable comparative signals across generated games.

### Environment constraints

Limit first-generation AsymBench games to:

- 2 players;
- deterministic;
- perfect information;
- alternating turns;
- small boards, ideally 5x5 to 8x8;
- discrete action lists;
- no stochasticity;
- no hidden information;
- no negotiation;
- max game length under 150 plies;
- branching factor preferably 5-30;
- compact observations that can be encoded as planes or feature vectors.

### Baseline tiers

Tier 0: no learning

- random;
- greedy heuristic if game exposes score or progress;
- flat Monte Carlo;
- shallow minimax or MCTS where feasible.

Tier 1: lightweight RL

- tabular Q-learning only for very small state spaces;
- DQN/PPO-style self-play for feature-vector states;
- shared policy versus role-conditioned policy;
- fixed opponent pools to reduce non-stationarity.

Tier 2: AlphaZero-lite

- small MLP or tiny CNN policy/value model;
- small MCTS budget;
- role-specific policy/value heads;
- replay buffer with past checkpoints;
- compare against shared-head model.

Tier 2 should be optional until Tier 0 and Tier 1 are stable.

### Key RL metrics

- `role_learning_auc`: area under learning curve for each role.
- `role_sample_efficiency_gap`: training steps to reach target win rate by role.
- `role_head_advantage`: role-specific-head performance minus shared-head performance.
- `role_interference`: drop in performance on role A after training role B.
- `checkpoint_exploitability_proxy`: win/loss against a pool of past checkpoints.
- `policy_entropy_by_role`: whether one role collapses faster than the other.
- `balance_at_convergence`: final role win rates between equal-strength trained agents.

The Tablut paper makes `role_head_advantage` and `role_interference` especially credible.

## Game Generation Strategy

Do not start with unconstrained arbitrary asymmetry. Start with controlled archetypes:

1. Pursuer vs Evader
   - one role captures or blocks;
   - the other escapes or scores exits.

2. Builder vs Breaker
   - one role creates structures;
   - the other destroys, freezes, or corrupts them.

3. Monarch vs Swarm
   - one powerful piece versus many weak pieces;
   - different objectives.

4. Economy vs Raid
   - one role accumulates resources;
   - the other wins through disruption or tactical strikes.

5. Territory vs Mobility
   - one role scores area control;
   - the other scores routes, crossings, or exits.

Each generated game should declare:

```json
{
  "asymmetry_profile": {
    "setup": "different",
    "piece_sets": "role_specific",
    "action_spaces": "role_specific",
    "objectives": "role_specific",
    "resources": "none | role_specific",
    "information": "perfect",
    "turn_structure": "alternating",
    "tempo": "equal | asymmetric"
  }
}
```

The first benchmark should avoid hidden information. We can add hidden roles later if the basic asymmetric engine and RL setup works.

## Evaluation Protocol

For each generated game:

1. Validate engine contract.
2. Confirm both roles have legal opening moves.
3. Confirm both roles can win in at least one guided or search-based trajectory.
4. Run random-vs-random by role and seat.
5. Run MC/MCTS-vs-random with the stronger agent assigned to each role.
6. Run equal-strength MC/MCTS-vs-MC/MCTS.
7. Compute role advantage and first-player advantage separately.
8. Train lightweight RL agents on accepted games.
9. Compare shared-head and role-specific-head models.
10. Run LLM agents with structured legal-action output.
11. Optionally run persona feedback and a small human study.

Reject games if:

- one role has no plausible win path;
- equal-strength MCTS gives a role win rate outside a configurable range, e.g. 25-75% for early experiments or 35-65% for polished benchmark games;
- random games rarely terminate;
- average length is too short or too long;
- branching factor is too small for strategy or too large for tractable evaluation;
- role rules are too complex to teach.

## What To Compare Against

We need baselines that make reviewers comfortable.

Compare against:

- the existing symmetric game generator mode;
- hand-written asymmetric reference games, such as a tiny Tablut-like game;
- RuleSmith-style parameter tuning on one hand-authored game, if feasible;
- fixed OpenSpiel or Ludii games where possible;
- LLM-only critique without engine execution;
- random and flat-MC playtest reports already in `gamegen`.

The strongest comparison is:

> symmetric generated games versus asymmetric generated games under the same evaluation harness.

That isolates the value of asymmetry.

## Novelty Claim Ladder

### Conservative claim

> We extend an executable LLM-based board-game generator from symmetric games to asymmetric games and introduce role-aware validation and playtesting metrics.

This is useful but may be too incremental.

### Solid claim

> We introduce a generated asymmetric benchmark suite and show that role-aware metrics reveal agent weaknesses hidden by symmetric game benchmarks.

This is stronger because it makes the benchmark the contribution.

### Strong claim

> We show that generated asymmetric games expose role-specific learnability gaps in RL and LLM agents, including shared-policy interference, role-conditioned rule errors, and evaluator-specific balance failures.

This is the target.

### Very strong claim

> We demonstrate that balance is evaluator-dependent: games balanced for LLM self-play can be measurably imbalanced for MCTS, RL, or humans, so asymmetric game evaluation requires cross-evaluator role-aware validation.

This directly challenges the premise of using LLM agents as playtesters without external calibration.

## Minimal Publishable Study

A realistic first paper could do this:

1. Generate 100 asymmetric candidate games.
2. Filter to 10 accepted games through engine validation and search baselines.
3. Include 5 symmetric generated games as controls.
4. For every game, report:
   - validation pass/fail reasons;
   - role advantage;
   - first-player advantage;
   - branching factor;
   - termination rate;
   - MC/MCTS skill signal.
5. Train lightweight RL agents on 3-5 games:
   - shared-head model;
   - role-specific-head model;
   - maybe fixed-opponent PPO or AlphaZero-lite.
6. Evaluate 3-5 LLMs on the same games:
   - legal move rate;
   - win rate;
   - role objective errors;
   - state tracking errors.
7. Optionally run a small human study:
   - maybe 12-24 participants;
   - each plays two role-swapped games;
   - collect perceived fairness, complexity, fun, and role preference.

This would be enough for a serious workshop or conference submission if executed cleanly.

## What We Should Not Do First

Avoid starting with:

- hidden information;
- negotiation;
- social deduction;
- large 4X-like games;
- arbitrary natural-language rules without a strict schema;
- LLM self-play as the only balance signal;
- a full AlphaZero-scale training project;
- subjective persona critique before the engine/RL layer works.

Those can be later extensions.

## Practical Implementation Direction

The current generator already has useful pieces:

- strict spec schema;
- engine generation;
- validation;
- deterministic playtesting;
- flat-MC baseline;
- critic and novelty checker;
- run logs.

The first implementation target should be an `asymmetric_v1` mode:

- new schema fields: roles, role-specific pieces, role-specific legal actions, role-specific objectives;
- remove mandatory `mirror_state` for this mode;
- add `roles()`, `player_role()`, and `observation()` to the engine contract;
- update playtest reports to be role-aware;
- add hand-written reference asymmetric games before generating LLM games;
- keep hidden information out of v1.

The second target should be a compact RL export:

- JSON observation/action encoding;
- deterministic reset and step API;
- role id in observation;
- optional PettingZoo-style wrapper;
- small vectorized batch runner if we pursue AlphaZero-lite or PPO.

## Recommended Framing For Hafsteinn

Use this framing:

> RuleSmith balances one parameterized asymmetric game using LLM self-play. MeepleLM predicts subjective player critique from rulebooks and personas. Our gap is executable generation: create many small asymmetric games, validate them mechanically, and use them to study role-specific balance, learnability, and LLM/human failure modes. If we can make the RL experiments tractable, AsymBench becomes a benchmark of asymmetric reasoning rather than just another game generator.

Shorter version:

> The novelty is not automatic balancing or virtual feedback. The novelty is generated asymmetric benchmark tasks with role-aware RL/search/LLM/human evaluation.

## Concrete Next Steps

1. Build two or three hand-written asymmetric reference games.
2. Add role-aware playtest metrics to the current harness.
3. Implement the simplest RL-compatible environment wrapper.
4. Run shared-policy versus role-specific-policy experiments on the hand-written games.
5. Only then modify the LLM designer prompt to generate asymmetric games.
6. Generate a pilot batch and measure pass rates.
7. Pick 3-5 accepted games for deeper RL and LLM evaluation.
8. Decide whether a small human or MeepleLM-style persona study is feasible.

## Source List

- RuleSmith paper: https://arxiv.org/html/2602.06232v1
- RuleSmith GitHub: https://github.com/Adonis-galaxy/RuleSmith
- MeepleLM ACL entry: https://aclanthology.org/2026.acl-long.850/
- MeepleLM arXiv: https://arxiv.org/abs/2601.07251
- Reproducing AlphaZero on Tablut: https://arxiv.org/abs/2604.05476
- OpenSpiel: https://arxiv.org/abs/1908.09453
- PettingZoo: https://arxiv.org/abs/2009.14471
- Pgx: https://arxiv.org/abs/2303.17503
- GameBench: https://arxiv.org/abs/2406.06613
- Board Game Arena: https://arxiv.org/abs/2508.03368
- TextArena: https://arxiv.org/abs/2504.11442
- DSGBench: https://arxiv.org/abs/2503.06047
- AutoBG: https://arxiv.org/abs/2606.01976
- GAVEL: https://arxiv.org/abs/2407.09388
- Grammar and Gameplay-aligned RL for Game Description Generation: https://arxiv.org/abs/2503.15783
- Boardwalk: https://arxiv.org/abs/2508.16447
