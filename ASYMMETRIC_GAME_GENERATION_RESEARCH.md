# Asymmetric Game Generation: Novelty Scan and Research Direction

Date: 2026-07-02

This note summarizes the current project position, adjacent research, and a concrete path toward a defensible novel contribution around generated asymmetric games for evaluating LLM agents, classical agents, and humans.

## Short Answer

The broad idea is not new:

- Automatic game generation has existed since at least Metagame and Ludi.
- Ludii and GAVEL already cover broad board-game description, analysis, and generation.
- LLM game benchmarks are now an active area, including GameBench, TextArena, Board Game Arena, DSGBench, GAMEBoT, Mindgames, lmgame-Bench, Orak, and others.
- Asymmetric games and asymmetric-information games are already used in AI and LLM-agent evaluation.

The stronger novelty angle is narrower:

> A reproducible pipeline that generates executable asymmetric abstract board games, validates them mechanically, estimates role balance and role-specific skill signals, and uses fresh generated games as controlled human/LLM benchmark tasks.

That is different from:

- generating symmetric abstract games,
- hand-picking existing asymmetric games,
- balancing one hand-designed asymmetric environment,
- creating a design assistant for human board-game designers,
- benchmarking LLMs on fixed games that may be known or contaminated.

The contribution should be framed as "asymmetry-aware generated benchmark design", not "first automatic board-game generator".

## Current Project Baseline

There are two relevant local checkouts:

- Generator repo: `C:\Users\solvi\Desktop\BoardGameGenerator`
- Generated games repo: `C:\Users\solvi\Desktop\BoardGameGeneratorGames`

The current generator is explicitly symmetric:

- `README.md` says the system invents "two-player symmetric abstract board games".
- `prompts/designer.md` says the first move must be the only asymmetry.
- `prompts/designer_revision.md` requires revisions to preserve symmetry, perfect information, and no chance.
- `prompts/critic.md` rejects revisions that introduce hidden information, chance, or asymmetry.
- `gamegen/schema.py` rejects asymmetric setup.
- `gamegen/engine_interface.py` requires `mirror_state`.
- `gamegen/validator.py` validates color-swap symmetry through mirrored opening moves.
- `gamegen/playtest.py` reports first-player win rate, draw rate, MC-vs-random skill signal, MC-vs-MC first-player rate, and branching factor.

The generated games repo currently contains `filewake/`, a single generated game. Its `spec.json` says both players have the same pieces, movement rule, scoring rule, target count, and no-legal-move loss rule. Its `playtest_report.json` shows:

- random first-player win rate: 0.505
- draw rate: 0.0
- flat-Monte-Carlo vs random win rate: 0.988
- average random game length: 68.67 plies

This is a useful base. The project already has the right ingredients: LLM generation, executable engines, deterministic validation, random and search baselines, replayable logs, mechanical playtesting, and adversarial novelty review. The missing research layer is that all of those components currently assume symmetry.

## Relevant Prior Work

### 1. General game generation is old

Barney Pell's Metagame generated games from a restricted class of symmetric chess-like games and used them to force agents to rely on general principles rather than one fixed rule set. That is close in spirit to generated benchmarks, but it was restricted to a symmetric chess-like class and predates modern LLMs.

Source: [Metagame in Symmetric Chess-Like Games](https://svn.sable.mcgill.ca/sable/courses/COMP763/oldpapers/pell-92-metagame.pdf)

Cameron Browne and Frederic Maire's Ludi system automatically measured and synthesized combinatorial games through self-play and aesthetic predictors. It evolved 1,389 games and identified 19 viable games, including games that later became known outside the research context. This means "automatic abstract board-game generation" is definitely prior art.

Source: [Evolutionary Game Design](https://cambolbro.com/cv/publications/ciaig-browne-maire-19.pdf)

### 2. Ludii is the major existing board-game substrate

Ludii is a general game system for playing, evaluating, and designing many games, including board games, card games, dice games, and mathematical games. It represents games as structured ludemes and includes a large game library.

Source: [Ludii Portal](https://ludii.games/)

Ludii also has an explicit concept taxonomy. Its concept list includes hidden information, stochasticity, asymmetric rules, asymmetric play rules, asymmetric end rules, asymmetric forces, asymmetric setup, and asymmetric piece types.

Source: [Ludii concept search](https://ludii.games/searchConcepts.php)

The General Board Game Concepts paper formalizes these concepts for GGP and discusses uses including game generation, game reconstruction, recommendation, explainability, transfer learning, and benchmark selection.

Source: [General Board Game Concepts](https://arxiv.org/pdf/2107.01078)

Implication for novelty: we should not claim that detecting or representing asymmetric game concepts is new. A stronger claim is to make asymmetry a first-class generation and evaluation target for LLM/human benchmarking.

### 3. GAVEL is the closest modern game-generation neighbor

GAVEL combines Ludii, language models, and quality-diversity search. It trains a code model to mutate/recombine Ludii game descriptions, evaluates candidates through compilation/playability/balance/agency/depth checks, and uses Ludii concept vectors to maintain diversity.

Source: [GAVEL: Generating Games Via Evolution and Language Models](https://arxiv.org/html/2407.09388v2)

GAVEL is a serious prior-art pressure point. It can use Ludii concepts, including asymmetry concepts, and it generates novel board games in a broad DSL.

Possible differentiation:

- GAVEL is primarily about generating new games in Ludii concept space.
- Our contribution can be about generated games as benchmark instruments.
- GAVEL's evaluation is game-quality oriented; ours can be role-balance, role-difficulty, agent-skill, human-calibration, and contamination-resistance oriented.
- GAVEL does not appear to center a typed taxonomy of asymmetry as the core experimental variable.

### 4. LLM generation of formal game descriptions is active

Hu, Zhao, and Liu investigate LLM generation of VGDL game rules and levels. They show that context matters and that LLMs can produce parsable but semantically wrong game logic.

Source: [Generating Games via LLMs: An Investigation with VGDL](https://arxiv.org/html/2404.08706v1)

Tanaka and Simo-Serra propose grammar-based game description generation with LLMs, using explicit GDL grammar to improve game description generation. Follow-up work uses grammar and gameplay-aligned reinforcement learning for game description generation with Ludii GDL.

Sources:

- [Grammar-based Game Description Generation using LLMs](https://arxiv.org/html/2407.17404v1)
- [Grammar and Gameplay-aligned RL for Game Description Generation with LLMs](https://arxiv.org/html/2503.15783v2)

Implication for novelty: formal rule generation by LLM is not enough. The novelty needs to be in what the generated games are for and how they are validated.

### 5. LLM board-game implementation is also being studied

Boardwalk studies whether LLMs can implement digital board games from natural-language rules, using both free-form and API-constrained implementations. Their reported best model produced error-free games in only 55.6% of cases.

Source: [Boardwalk: Towards a Framework for Creating Board Games with LLMs](https://sol.sbc.org.br/index.php/sbgames/article/view/37375)

Implication: the current pipeline's strict engine validation and repair loop is useful. Generated asymmetric games will need even stronger contract tests because asymmetric roles create more failure modes.

### 6. Board-game design assistants exist

AutoBG is an end-to-end board-game design assistant with ideation, rulebook generation, critic-guided revision, and persona feedback. It is aimed at helping designers and uses large datasets of rulebooks and player reviews.

Source: [AutoBG](https://arxiv.org/html/2606.01976v1)

Implication: avoid framing the project as a general-purpose board-game design assistant. Frame it as a benchmark generator with mechanical validity and evaluation science.

### 7. LLM game benchmarks are crowded

GameBench evaluates strategic reasoning across nine game environments chosen to cover abstract strategy, stochasticity, hidden information, language communication, social deduction, and cooperation. It also uses a Bradley-Terry style rating.

Source: [GameBench](https://arxiv.org/html/2406.06613v1)

Board Game Arena uses OpenSpiel to evaluate LLMs, random agents, humans, and RL agents on board and matrix games, with unified game loops, prompt interfaces, backends, and logging.

Sources:

- [Board Game Arena](https://arxiv.org/html/2508.03368v1)
- [OpenSpiel docs](https://openspiel.readthedocs.io/en/latest/intro.html)

TextArena provides many text-based games, online play, and TrueSkill ratings, focusing on agentic behavior and social skills.

Source: [TextArena](https://arxiv.org/abs/2504.11442)

DSGBench evaluates LLM agents in complex strategic games such as StarCraft II, Civilization, Diplomacy, Werewolf, and Stratego, with fine-grained decision tracking.

Source: [DSGBench](https://arxiv.org/html/2503.06047v2)

GAMEBoT decomposes game reasoning into modular subproblems and validates intermediate reasoning, not just final win/loss.

Source: [GAMEBoT](https://arxiv.org/abs/2412.13602)

Mindgames builds on TextArena and evaluates social and strategic reasoning across multi-agent settings, including hidden information, opponent modeling, cooperative inference, and deception.

Source: [Mindgames](https://arxiv.org/abs/2605.29512)

Other recent benchmarks include lmgame-Bench, Orak, PTCG-Bench, Clue-style deductive game environments, and hidden-role evaluations. These emphasize perception, memory, planning, harness design, self-evolution, deduction, and deception.

Sources:

- [lmgame-Bench](https://arxiv.org/abs/2505.15146)
- [Orak](https://arxiv.org/abs/2506.03610)
- [PTCG-Bench](https://arxiv.org/html/2605.29653v1)
- [Text-based Clue evaluation](https://arxiv.org/html/2603.17169v1)
- [Hidden-role LLM evaluation](https://arxiv.org/html/2605.22826v1)

Implication: a benchmark paper must be precise about what current benchmarks do not isolate. "Games are good benchmarks" is no longer a contribution.

### 8. Asymmetric game balancing is directly adjacent

RuleSmith is very close in motivation: it uses LLM-driven self-play and Bayesian optimization to balance asymmetric games. Its demonstration game, CivMini, has asymmetric civilizations with different units, production, incentives, and combat parameters.

Source: [RuleSmith](https://arxiv.org/html/2602.06232v1)

This is the most important recent competitor for the asymmetric angle.

Possible differentiation:

- RuleSmith balances a parameterized asymmetric game.
- We can generate many new asymmetric games, not only tune a hand-authored environment.
- RuleSmith uses LLM agents as simulators for balancing; we can use deterministic/rule-based/search baselines first, then evaluate LLMs and humans separately.
- RuleSmith optimizes fairness; we can optimize benchmark usefulness: balanced enough, role-distinct, skill-separating, teachable, contamination-resistant, and reproducible.

### 9. Rating asymmetric games needs special care

Ben Wise's work on Elo ratings for asymmetric software-agent tournaments argues that asymmetric games and AI tournaments require modifications to standard rating assumptions.

Source: [Elo Ratings for Large Tournaments of Software Agents in Asymmetric Games](https://arxiv.org/abs/2105.00839)

Implication: do not report a single naive Elo or win rate across asymmetric roles. Use role-aware pairing, role-swapped matches, and models that separate agent strength from role advantage.

## What Would Be Novel

### Defensible Core Claim

The cleanest claim is:

> We introduce an asymmetry-aware game-generation pipeline that creates executable, mechanically validated, role-balanced abstract board games for evaluating humans and LLM agents on fresh strategic tasks.

This claim has four parts that must all be true:

1. The games are generated, not hand selected.
2. The games are asymmetric by design, not accidentally biased.
3. The asymmetry is typed, measured, and validated.
4. The outputs are used as evaluation instruments, not just playable curiosities.

### Claims to Avoid

Avoid these claims:

- "First automatic board-game generator." False because of Metagame, Ludi, Ludii, GAVEL, and related systems.
- "First LLM-generated games." False because VGDL, Ludii GDL, GAVEL, AutoBG, and related work exist.
- "First LLM game benchmark." False because the benchmark literature is now large.
- "First asymmetric game work." False because Ludii concepts, asymmetric tournament ratings, social deduction, hidden-role games, RuleSmith, and game-theoretic work exist.
- "Asymmetric means better." Too vague. The project should show which cognitive abilities asymmetry tests.

### Stronger Possible Claim

If the implementation and study are good, a stronger claim is:

> Symmetric generated games mostly test rule comprehension, planning, and tactical search under common rules. Generated asymmetric games additionally test role-conditioned rule learning, opponent modeling across unequal action spaces, strategic transfer between roles, and the ability to distinguish role advantage from agent skill.

That is a real evaluation-science claim and can be tested experimentally.

## Recommended Research Direction

### Build an Asymmetry-First Generator

Extend the generator so every game declares an explicit asymmetry profile:

```json
{
  "asymmetry_profile": {
    "setup": "same | different",
    "pieces": "same | role_specific",
    "actions": "same | role_specific",
    "objectives": "same | role_specific",
    "information": "public | private | hidden_role | asymmetric_observation",
    "resources": "same | role_specific",
    "turn_structure": "alternating | role_tempo | simultaneous",
    "communication": "none | public | private | restricted"
  }
}
```

Start with two-player deterministic perfect-information asymmetric abstracts. Add hidden information only after the engine contract and validators are stable.

Good first asymmetry families:

- Pursuer vs evader: different movement and win conditions.
- Builder vs breaker: one creates structures, the other disrupts or races.
- Attacker vs defender: unequal forces but bounded objective.
- Scorer vs blocker: one accumulates points, one controls tempo or denies access.
- Cartographer vs raider: one controls map expansion, the other controls piece mobility.

Avoid starting with diplomacy, negotiation, or deception. Those are interesting but will make validation and evaluation noisy.

### Replace Symmetry Validation With Role-Equivalence Validation

The current `mirror_state` check should not apply to asymmetric games. Replace it with checks that validate:

- every role has legal moves at the start,
- every role can win under at least one constructive playout or guided search,
- every role has non-trivial choices,
- all actions declared in the spec are reachable or explainably rare,
- terminal conditions are reachable,
- games terminate under random, scripted, and search-based policies,
- no role has a dominant immediate win or forced no-move loss,
- role-swapped seating separates first-player advantage from role advantage.

For asymmetric games, the key object is not a color-swap isomorphism. It is a role assignment and seat assignment matrix:

| Seat | Role A agent | Role B agent | What it estimates |
|---|---|---|---|
| A first | X | Y | role plus tempo |
| B first | Y | X | role plus tempo with swapped agents |
| A first | Y | X | agent/role interaction |
| B first | X | Y | agent/role interaction with swapped agents |

The benchmark should estimate:

- role advantage,
- first-move advantage,
- agent strength,
- role-agent interaction,
- role-specific skill signal.

### Use Role-Aware Metrics

Recommended metrics:

- `role_win_rate_random`: win rate by role under random-vs-random.
- `role_win_rate_search`: win rate by role under MC-vs-MC or MCTS-vs-MCTS.
- `role_advantage_delta`: absolute deviation from 0.5 after controlling for seat.
- `tempo_advantage_delta`: first-mover effect after controlling for role.
- `skill_signal_by_role`: search-vs-random win rate when the skilled agent plays each role.
- `cross_role_transfer`: agent performance after seeing one role versus after seeing both role manuals.
- `illegal_action_rate_by_role`: LLM rule-following errors separated by role.
- `state_tracking_error_by_role`: failures to report public state, private state, or legal actions.
- `rules_load`: approximate length and branching complexity of each role's rule text.
- `strategic_distinctness`: distance between roles' legal-action types, objectives, resources, and evaluation features.
- `teachability`: human time-to-first-legal-move and rules quiz accuracy.
- `human_balance`: role win rate after novice and experienced humans play role-swapped matches.

This gives the project something most LLM benchmarks lack: explicit separation between game quality, role fairness, and model skill.

### Keep Generated Games Fresh

One major benchmark problem is contamination. Fixed games become part of model training data and online strategy corpora. Generated games can help if they are:

- seedable,
- archived,
- anonymized,
- withheld until evaluation,
- simple enough to teach quickly,
- mechanically validated before use,
- generated from a constrained grammar that avoids copying known games.

The benchmark could use:

- a public training split of generated games,
- a hidden evaluation split generated after model cutoff or shortly before evaluation,
- a reproducible seed-release protocol after results are locked,
- novelty reports against known games.

### Compare Humans, LLMs, and Classical Agents

The paper becomes much stronger if it includes humans. Suggested study:

1. Generate 20-50 candidate asymmetric games.
2. Filter mechanically to 5-10 viable games.
3. Run random, flat-MC, MCTS, and maybe simple heuristic agents.
4. Run LLM agents with identical harnesses.
5. Run a small human study with role-swapped pairs.
6. Compare:
   - humans vs LLMs on legal play,
   - humans vs LLMs on role learning,
   - LLMs vs search agents on tactical strength,
   - role balance under different skill levels,
   - how often LLM rankings change between symmetric and asymmetric generated games.

Useful experimental conditions:

- rules only,
- rules plus examples,
- role-specific private manual,
- full manual with opponent role visible,
- short memory vs full history,
- action list shown vs action list hidden,
- one-shot play vs repeated play.

### Add Intermediate Reasoning Checks Without Trusting CoT

Do not rely on free-form chain-of-thought as ground truth. Instead, ask agents for structured intermediate outputs:

```json
{
  "public_state_summary": "...",
  "my_role_objective": "...",
  "opponent_role_objective": "...",
  "legal_action_considered": "...",
  "predicted_opponent_reply": "...",
  "chosen_action": "..."
}
```

Then validate what can be validated:

- Was the legal action actually legal?
- Did the agent correctly identify its own objective?
- Did it correctly identify the opponent's objective?
- Did it maintain state across turns?
- Did it make errors specific to its role's asymmetric rules?

This borrows the good idea from transparent benchmark work like GAMEBoT, but applies it to generated asymmetric games.

## Concrete Differentiation Table

| Existing work | What it already covers | How this project can differ |
|---|---|---|
| Metagame | Generated symmetric chess-like games for general game playing | Modern LLM pipeline, executable audit trail, explicit asymmetric roles, human/LLM evaluation |
| Ludi | Evolutionary abstract game generation and quality predictors | Asymmetry-aware benchmark generation rather than general game design quality |
| Ludii | Broad game DSL, game concepts, asymmetric concept detection | Use asymmetry concepts as generation targets and benchmark variables |
| GAVEL | LLM/evolution generation over Ludii game descriptions | Focus on generated asymmetric games as evaluation instruments, not just game discovery |
| VGDL LLM generation | LLM-generated rules and levels | Abstract board games, stronger validation, role-balance metrics |
| Grammar/RL GDL generation | Grammar-constrained Ludii GDL output | Asymmetry-aware benchmark validity and role metrics |
| AutoBG | Board-game design assistant and rulebook improvement | Research benchmark generator, not designer support |
| Boardwalk | LLM implementation of board games from rules | Generate and validate new games; use implementation reliability as part of pipeline |
| GameBench | Fixed games covering strategic reasoning categories | Fresh generated asymmetric games with role-aware controls |
| Board Game Arena | OpenSpiel wrapper for LLM board-game evaluation | Generated games and role-specific asymmetry rather than fixed OpenSpiel catalogue |
| TextArena/Mindgames | Text/social multi-agent games with ratings | Executable abstract board games with mechanical validation and human calibration |
| DSGBench | Complex fixed strategic games and fine-grained metrics | Lightweight generated tasks isolating asymmetry variables |
| GAMEBoT | Validates intermediate reasoning in games | Apply structured intermediate checks to generated asymmetric roles |
| RuleSmith | Balances an asymmetric parameterized game via LLM self-play and BO | Generate many asymmetric games, use non-LLM mechanical baselines first, evaluate humans/LLMs separately |
| Asymmetric Elo work | Rating/tournament concerns for asymmetric software-agent games | Use role-aware ratings as part of generated-game benchmark design |

## Proposed System Changes

### Phase 1: Two-Player Perfect-Information Asymmetry

Modify:

- `prompts/designer.md`: ask for asymmetric games with typed asymmetry.
- `prompts/designer_revision.md`: preserve declared asymmetry instead of preserving symmetry.
- `prompts/critic.md`: score role distinctness, role balance, teachability, and benchmark usefulness.
- `prompts/rules_engineer.md`: replace `mirror_state` assumptions with role metadata and optional canonicalization.
- `gamegen/schema.py`: add `players`, `roles`, `asymmetry_profile`, `role_rules`, and role-specific win/loss conditions.
- `gamegen/engine_interface.py`: add `roles()`, `player_role(state, player)`, and maybe `observation(state, player)`.
- `gamegen/validator.py`: remove mandatory symmetry check for asymmetric mode; add role viability and reachability checks.
- `gamegen/playtest.py`: record role-aware, seat-aware, and role-swapped metrics.

Do not add hidden information yet. Keep all state public and deterministic until role-aware mechanics are reliable.

### Phase 2: Benchmark Harness

Add:

- LLM player wrapper with structured action output.
- Rule quiz generator for humans and LLMs.
- Illegal move parser and repair policy.
- Role-swapped tournament runner.
- Rating model that separates agent, role, and first-player effects.
- Export format for generated benchmark sets.

The benchmark should log full trajectories:

- state,
- acting player,
- role,
- legal actions,
- selected action,
- parser errors,
- invalid action errors,
- terminal outcome,
- role-specific metrics.

### Phase 3: Hidden Information and Social Asymmetry

Only after Phase 1 and 2 are stable:

- private observations,
- hidden role objectives,
- asymmetric information,
- restricted communication,
- bluffing/deception tasks.

This phase is higher risk because it mixes game generation, imperfect-information state handling, and LLM social reasoning. It is publishable if done well, but it should not be the first milestone.

## Example Generated Game Design Targets

These are not final games, just target archetypes for the generator.

### 1. Runner vs Net

One role moves a small number of fast pieces toward exits. The other places temporary barriers and moves slower sentries. The runner wins by escaping enough pieces; the net wins by immobilizing or delaying until a turn limit.

Benchmark skill: planning under unequal action spaces, opponent modeling, tempo.

### 2. Builder vs Breaker

The builder places links to connect two board regions. The breaker moves a limited number of disruptors that remove or freeze links. Both roles have public information and deterministic moves.

Benchmark skill: strategic abstraction, threat evaluation, long-horizon planning.

### 3. Cartographer vs Raider

The cartographer expands the board by revealing or claiming cells. The raider moves through revealed cells and scores by reaching high-value sites. The roles share the board but value different regions.

Benchmark skill: role-conditioned valuation and adaptive planning.

### 4. Monarch vs Rebels

The monarch has one powerful piece and limited guards. Rebels have many weak pieces and win by surrounding or occupying key cells. The monarch wins by crossing or capturing enough rebels.

Benchmark skill: asymmetric force evaluation and tactical search.

## Evaluation Protocol Sketch

For each generated game:

1. Validate engine contract and deterministic behavior.
2. Run random-vs-random by all role/seat assignments.
3. Run MC-vs-random by all role/seat assignments.
4. Run MC-vs-MC by all role/seat assignments.
5. Reject games where:
   - either role has less than 35% or more than 65% win rate under equal-strength search,
   - random play almost never terminates,
   - branching factor is too low or too high,
   - either role lacks meaningful choices,
   - either role's objective is unreachable in guided tests,
   - rules are too long to teach in a short human study.
6. Run LLM agents on accepted games with structured action output.
7. Run humans on a smaller subset with role swaps.
8. Fit a model such as:

```text
outcome ~ agent_strength + role_advantage + first_player_advantage
        + agent_role_interaction + game_random_effect
```

This is more defensible than raw win rates.

## What a Paper Could Claim

### Minimum Viable Paper

> We extend an LLM-based board-game generator from symmetric games to mechanically validated asymmetric games and introduce role-aware playtesting metrics that separate role advantage from agent skill.

Evidence needed:

- 10-20 accepted generated asymmetric games.
- Mechanical validation.
- Random/search baselines.
- Role-aware metric tables.
- Comparison to symmetric generator outputs.

### Strong Paper

> Generated asymmetric games expose LLM failures not visible in symmetric generated games, especially role-conditioned rule tracking, opponent-objective modeling, and strategy transfer between roles.

Evidence needed:

- Several LLM agents.
- Symmetric vs asymmetric benchmark split.
- Humans or strong classical baselines.
- Error taxonomy by role.
- Statistical model separating role and agent effects.

### Ambitious Paper

> Fresh generated asymmetric games provide a contamination-resistant benchmark family where difficulty and cognitive demands can be controlled through explicit asymmetry profiles.

Evidence needed:

- Public generator.
- Train/dev/test game splits.
- Held-out generated evaluation set.
- Human calibration.
- Repeated generation showing stable control over target asymmetry profiles.

## Main Risks

### Risk: Asymmetric games are harder to balance

Mitigation: accept "balanced enough" bands, use role-swapped evaluation, tune numeric parameters after structural generation, and report uncertainty.

### Risk: LLMs generate invalid or unclear asymmetric rules

Mitigation: typed schemas, role-specific rule sections, stricter examples, validator feedback, and structured rulebooks.

### Risk: LLM self-play is too noisy for game filtering

Mitigation: do not use LLM self-play as the first filter. Use random, scripted, MC/MCTS, and only then LLMs.

### Risk: Role difficulty is confounded with role strength

Mitigation: separate "role power" from "role cognitive load". A role can be balanced but harder for LLMs or novices.

### Risk: Generated games are legal but not fun

Mitigation: for benchmark purposes, fun is secondary to validity, teachability, and skill separation. For human studies, still filter for clarity and length.

### Risk: Prior work may already generate asymmetric games incidentally

Mitigation: claim focus and evaluation method, not first incidental generation.

## Immediate Next Steps

1. Add an `asymmetric` mode to the generator config rather than replacing symmetric mode.
2. Extend the spec schema with roles and asymmetry profile.
3. Create 3-5 hand-written reference asymmetric engines to test validator/playtest logic before asking LLMs to generate them.
4. Replace `mirror_state` validation with role viability, reachability, and role-swapped metrics in asymmetric mode.
5. Add a role-aware playtest report.
6. Update critic and novelty prompts to evaluate role distinctness and benchmark usefulness.
7. Generate a small pilot batch and manually inspect failures.
8. Only then add LLM agent evaluation.

## Recommended Project Positioning

Use wording like:

> Our system generates mechanically validated asymmetric abstract board games as fresh evaluation environments. Unlike prior game-generation systems that optimize broad game quality or novelty, we target role-distinct benchmark tasks and report role-aware measures of balance, skill signal, rule-following, and transfer. Unlike fixed LLM game benchmarks, our tasks can be generated after model training cutoffs and audited from seed to engine to playtest logs.

Avoid wording like:

> We are the first to generate board games with AI.

## Source List

- Pell, "Metagame in Symmetric Chess-Like Games" - https://svn.sable.mcgill.ca/sable/courses/COMP763/oldpapers/pell-92-metagame.pdf
- Browne and Maire, "Evolutionary Game Design" - https://cambolbro.com/cv/publications/ciaig-browne-maire-19.pdf
- Ludii Portal - https://ludii.games/
- Ludii Concepts - https://ludii.games/searchConcepts.php
- Piette et al., "General Board Game Concepts" - https://arxiv.org/pdf/2107.01078
- Todd et al., "GAVEL: Generating Games Via Evolution and Language Models" - https://arxiv.org/html/2407.09388v2
- Hu, Zhao, Liu, "Generating Games via LLMs: An Investigation with VGDL" - https://arxiv.org/html/2404.08706v1
- Tanaka and Simo-Serra, "Grammar-based Game Description Generation using LLMs" - https://arxiv.org/html/2407.17404v1
- "Grammar and Gameplay-aligned RL for Game Description Generation with LLMs" - https://arxiv.org/html/2503.15783v2
- Becker et al., "Boardwalk: Towards a Framework for Creating Board Games with LLMs" - https://sol.sbc.org.br/index.php/sbgames/article/view/37375
- Li et al., "AutoBG" - https://arxiv.org/html/2606.01976v1
- Costarelli et al., "GameBench" - https://arxiv.org/html/2406.06613v1
- Board Game Arena - https://arxiv.org/html/2508.03368v1
- OpenSpiel docs - https://openspiel.readthedocs.io/en/latest/intro.html
- TextArena - https://arxiv.org/abs/2504.11442
- DSGBench - https://arxiv.org/html/2503.06047v2
- GAMEBoT - https://arxiv.org/abs/2412.13602
- Mindgames - https://arxiv.org/abs/2605.29512
- lmgame-Bench - https://arxiv.org/abs/2505.15146
- Orak - https://arxiv.org/abs/2506.03610
- PTCG-Bench - https://arxiv.org/html/2605.29653v1
- Text-based Clue evaluation - https://arxiv.org/html/2603.17169v1
- Hidden-role LLM evaluation - https://arxiv.org/html/2605.22826v1
- RuleSmith - https://arxiv.org/html/2602.06232v1
- Wise, "Elo Ratings for Large Tournaments of Software Agents in Asymmetric Games" - https://arxiv.org/abs/2105.00839
