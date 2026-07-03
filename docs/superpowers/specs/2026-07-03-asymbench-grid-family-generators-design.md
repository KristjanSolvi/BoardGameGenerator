# AsymBench Grid Family Generators Design

Date: 2026-07-03
Branch: `research/asymbench`

## Purpose

This design extends AsymBench from hand-written asymmetric reference games toward generated asymmetric game families.

The selected direction is:

> Build two family-specific generators for deterministic, perfect-information grid games, while keeping a shared evaluator interface and shared grid utilities.

This is the most practical next step because the current AlphaZero-lite harness already supports grid observations, integer actions, legal action masks, seat-role swaps, paired randomness, and role-aware metrics. It lets us generate games that the evaluator can actually learn from before we attempt broader game types such as auctions, hidden information, negotiation, or social deduction.

## Research Motivation

"Asymmetric games" is too broad to be a useful generator target. A grid capture game, an auction game, and a social deduction game need different state encodings, action spaces, legality checks, and agent assumptions. A single universal generator DSL would likely become a large independent project before we can test the research claim.

The first publishable path is narrower:

> Define controlled families of asymmetric grid games, generate many valid instances within those families, and evaluate whether role-aware learning agents expose balance, learnability, and role-interference properties that generic game generation/playtesting methods miss.

The novelty is not that asymmetric games exist. The novelty is the combination of:

- family-controlled asymmetric game generation;
- role-aware RL evaluation under paired randomness and seat swaps;
- metrics that separate role advantage, seat advantage, learnability, and model architecture effects;
- a pipeline that can later compare RL agents, LLM agents, and humans on the same generated asymmetric games.

## Selected Approach

Use two family-specific generators first:

1. `EscapeCaptureGenerator`
2. `ConnectionDisruptionGenerator`

Both produce games that implement the existing `AsymGame` protocol:

- immutable state;
- two roles;
- deterministic transitions;
- perfect information;
- integer action IDs;
- legal action masks;
- tensor observations;
- terminal result with winner/reason/plies;
- explicit seat-role mapping.

They share lower-level grid utilities, but they do not share one universal rule DSL yet.

### Why Not One Shared DSL First

A universal DSL would be attractive for long-term elegance, but it is risky now:

- it would need to cover movement, placement, capture, connection, scoring, resources, and terminal rules before we know which abstractions matter;
- it could produce formally valid but strategically useless games;
- debugging generated games would be harder because failures could come from the DSL, generator, compiler, or evaluator;
- it delays the research loop where generated games produce metrics.

Family-specific generators are easier to validate, easier to test, and enough to make the first research claim credible.

### Why Not Only One Family

One family is too narrow for a serious asymmetric-game benchmark claim. It risks becoming "we generated tafl variants" or "we generated connection variants." Two related but distinct families give breadth while keeping the evaluator tractable:

- Escape/Capture tests unequal pieces and opposing terminal objectives.
- Connection/Disruption tests asymmetric action types and constructive/destructive pressure.

## Scope

The first generator milestone covers only:

- two-player games;
- two asymmetric roles;
- deterministic transitions;
- perfect information;
- alternating turns;
- rectangular grid boards;
- finite horizons through terminal rules or max plies;
- full observability for both agents;
- discrete legal action lists.

The milestone does not cover:

- auctions;
- hidden information;
- stochastic outcomes;
- simultaneous moves;
- negotiation or natural-language actions;
- private utilities;
- more than two players;
- human-facing rulebook generation;
- LLM generation of rules.

Those are later research tracks after the grid benchmark works.

## Family 1: Escape/Capture

### Research Role

Escape/Capture games model asymmetric pursuit:

- one role protects or moves a key piece toward an exit condition;
- the other role tries to capture, immobilize, or block it.

This family generalizes the current `MicroTafl` reference game.

### Generated Parameters

The generator should vary:

- board size, initially 5x5 to 7x7;
- exit squares, usually corners or edge cells;
- key-piece start location;
- defender guard count and start pattern;
- attacker count and start pattern;
- movement type, initially orthogonal sliding or orthogonal step;
- capture rule, initially custodial sandwich or adjacency surround;
- hostile cells, such as corners or throne-like center;
- max plies.

### Validity Constraints

A generated Escape/Capture game is valid only if:

- both roles have at least one legal action from the initial state;
- both roles can legally reach at least one terminal condition in some bounded rollout/search;
- the key piece is not immediately captured or already escaped;
- no side has a trivial forced win found by shallow deterministic checks;
- average random/MCTS game length is within configured bounds;
- role/seat swaps are supported.

### Metrics Of Interest

Primary metrics:

- defender escape rate;
- attacker capture rate;
- no-progress/max-ply rate;
- role win rates under random, MCTS, and learned agents;
- policy entropy by role;
- role-head vs shared-head performance gap;
- whether role-head benefits differ between attacker and defender.

## Family 2: Connection/Disruption

### Research Role

Connection/Disruption games model constructive versus destructive asymmetry:

- one role builds a path, network, formation, or region;
- the other role blocks, removes, moves obstacles, or cuts connectivity.

This family generalizes the current `BreakerBuilder` reference game.

### Generated Parameters

The generator should vary:

- board size, initially 5x5 to 7x7;
- builder target edges or target regions;
- initial blockers;
- builder action type, initially place marker or advance marker;
- breaker action type, initially move blocker or remove adjacent marker;
- removal range, initially orthogonal adjacency;
- number of blocker pieces;
- maximum plies;
- optional protected cells that cannot be removed.

### Validity Constraints

A generated Connection/Disruption game is valid only if:

- the builder has at least one legal initial placement;
- the breaker has at least one legal response after common builder openings;
- the builder target is not already connected;
- the board is not permanently disconnected before play starts;
- both win/loss outcomes are reachable in bounded search or guided rollouts;
- random/MCTS rollouts are not dominated by immediate max-ply draws;
- seat-role swaps are supported.

### Metrics Of Interest

Primary metrics:

- builder connection rate;
- breaker prevention rate;
- max-ply prevention rate;
- role win rates by agent class;
- average plies to terminal outcome;
- learnability curve by role;
- role-head vs shared-head performance gap.

## Shared Grid Utilities

The generators should share utilities for:

- coordinate encoding and decoding;
- board shape validation;
- orthogonal/diagonal neighbor iteration;
- connected-component checks;
- path-to-edge checks;
- line-of-sight and sliding moves;
- action-mask construction;
- immutable board updates;
- canonical render helpers for debugging;
- deterministic seeded sampling.

These utilities should not define game semantics. They are mechanics helpers used by each family.

## Generated Game Representation

Each generated game should have two layers:

1. A serializable specification.
2. A compiled runtime implementing `AsymGame`.

The serializable spec should be JSON-friendly and stable enough to store in experiment outputs:

```json
{
  "family": "escape_capture",
  "name": "escape_capture_5x5_seed_17",
  "seed": 17,
  "board": {"rows": 5, "cols": 5},
  "roles": ["attacker", "defender"],
  "setup": {},
  "actions": {},
  "terminal_rules": {},
  "max_plies": 80
}
```

The exact nested fields are family-specific. The shared requirement is that every generated runtime can be reconstructed from the stored spec.

## Generator API

Each family generator should expose a compact API:

```python
class FamilyGenerator:
    family: str

    def generate(self, seed: int, constraints: GenerationConstraints) -> GeneratedGameSpec:
        ...

    def compile(self, spec: GeneratedGameSpec) -> AsymGame:
        ...

    def validate(self, spec: GeneratedGameSpec) -> ValidationReport:
        ...
```

The generator must be deterministic for a given seed and constraints.

`ValidationReport` should include:

- validity boolean;
- rejection reasons;
- initial branching factor;
- sampled rollout lengths;
- random role win rates;
- MCTS role win rates when requested;
- reachability checks that passed or failed.

## Generation Pipeline

The first pipeline should be simple and reproducible:

1. Sample candidate spec from family-specific parameter ranges.
2. Compile candidate into an `AsymGame`.
3. Run structural validity checks.
4. Run random rollout checks.
5. Run shallow MCTS checks for triviality.
6. Accept or reject candidate.
7. Write accepted spec and validation report to `research_runs/asymbench/generated/`.
8. Run role-head experiment configs on accepted games.

Rejected games should also be counted and summarized. Failed generation is research evidence because it identifies brittle parts of the design space.

## Evaluator Integration

The existing runner should eventually accept generated-game specs in addition to hard-coded game names.

Near-term config shape:

```json
{
  "game_source": {
    "type": "generated_spec",
    "path": "research_runs/asymbench/generated/escape_capture_seed_17/spec.json"
  },
  "device": "cuda",
  "seeds": [1, 2, 3],
  "model_variants": ["shared_heads", "role_heads"],
  "iterations": 20,
  "selfplay_games_per_iteration": 16,
  "mcts_simulations": 64
}
```

For the first implementation, it is acceptable to add a separate generator-runner script before fully generalizing `run_role_heads.py`.

## Evaluation Metrics

Generated games should be evaluated at three levels.

### Level 1: Formal Validity

- compiles into runtime;
- action masks match legal actions;
- terminal rules are reachable;
- max plies bound every game;
- observations and action spaces are stable.

### Level 2: Playability And Balance

- random role win rates;
- MCTS role win rates;
- draw/max-ply rate;
- average game length;
- branching factor statistics;
- shallow exploitability indicators, such as immediate forced terminal outcomes.

### Level 3: Learnability And Role Effects

- shared-head learning curve;
- role-head learning curve;
- role-specific value loss;
- role-specific policy entropy;
- final evaluation role win rates;
- sensitivity to seat-role swaps;
- paired-seed difference between shared and role-head variants.

## Novelty Claim Supported By This Design

This design supports a focused novelty claim:

> AsymBench is a family-based benchmark for generated asymmetric grid games, where generated games are evaluated with role-aware learning agents under controlled seat swaps and paired randomness.

This differs from prior automated game generation or virtual playtesting work because it treats role asymmetry as a first-class experimental variable. The benchmark does not only ask whether a generated game is legal or balanced overall. It asks:

- whether each role is learnable;
- whether one role is structurally dominant;
- whether seat order confounds role strength;
- whether role-conditioned neural architectures improve evaluation;
- whether different agent classes disagree about balance.

This creates a bridge from generated games to RL agents, LLM agents, and human playtesting.

## Testing Strategy

Tests should be added before implementation.

### Unit Tests

- generated specs are deterministic by seed;
- generated specs serialize and deserialize without loss;
- compiled games implement `AsymGame`;
- legal action masks match legal actions;
- invalid parameter combinations are rejected;
- structural validators reject impossible starts;
- connectivity and escape helpers work on small boards.

### Family Tests

Escape/Capture:

- generated game starts with key piece not terminal;
- attacker and defender both have legal actions;
- at least one escape-like terminal and one capture-like terminal are reachable in bounded tests.

Connection/Disruption:

- builder target is not initially connected;
- builder and breaker both receive legal turns;
- connection and prevention outcomes are reachable in bounded tests.

### Integration Tests

- generator can produce at least one accepted game per family under smoke constraints;
- accepted generated games can run random evaluation;
- accepted generated games can run one tiny AlphaZero-lite smoke iteration on CPU;
- metrics include family name, generation seed, validation report path, role win rates, and paired-seed fields.

## Acceptance Criteria

The next milestone is complete when:

1. Two family-specific generators exist: Escape/Capture and Connection/Disruption.
2. Each generator emits serializable specs and compiled `AsymGame` runtimes.
3. Each family has structural and rollout validators.
4. At least five accepted games per family can be generated from fixed seeds.
5. Accepted games run through random and MCTS validation.
6. At least one generated game per family runs through the AlphaZero-lite smoke pipeline.
7. Metrics preserve generation seed, family, validation summary, role/seat outcomes, and paired-seed audit fields.
8. Existing hand-written games and tests continue to pass.

## Deferred Work

The following are intentionally deferred:

- a universal grid DSL;
- LLM-authored game rules;
- natural-language rulebook export;
- auction/resource/non-grid families;
- hidden-information games;
- human study interface;
- large-scale training runs.

These become useful only after generated grid families produce reproducible, analyzable games.

## Risks And Mitigations

Risk: generated games are valid but strategically trivial.

Mitigation: include shallow MCTS and rollout rejection criteria before training.

Risk: family-specific generators look less general than a DSL.

Mitigation: frame this as an experimental benchmark design, not a final universal generator language. Extract common abstractions only after two families expose real repetition.

Risk: AlphaZero-lite overfits tiny games.

Mitigation: compare multiple generated seeds per family, use held-out generated games, and keep random/MCTS baselines.

Risk: role-head advantage is inconsistent.

Mitigation: report this honestly. A null result can still support the need for role-aware evaluation if role/seat/balance metrics reveal meaningful differences across generated games.

