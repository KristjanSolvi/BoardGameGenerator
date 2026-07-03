# LLMs and Symmetric vs. Asymmetric Games — Research Overview

Date: 2026-07-03

Literature scan of how LLMs play and generate symmetric vs. asymmetric games,
with baselines called out throughout. Companion to
`ASYMMETRIC_GAME_GENERATION_RESEARCH.md` and
`ASYMBENCH_NOVELTY_RESEARCH_2026-07-03.md`.

## 1. The distinction that matters

"Asymmetry" appears in two senses, and the literature tests different skills for each:

- **Symmetric games** — both players have the same pieces, actions, and
  objectives; asymmetry is only *who moves first* (chess, Go, Hive, Santorini,
  tic-tac-toe). Stress **rule comprehension, planning, and tactical search**
  under shared rules.
- **Asymmetric games** — two sub-types:
  - **Structural asymmetry**: different pieces / action spaces / win conditions
    (Tafl/hunt games, Fox & Geese, Stratego, CivMini). Tests
    **role-conditioned reasoning** and separating *role advantage* from
    *agent skill*.
  - **Informational asymmetry**: hidden roles or private information (Werewolf,
    Secret Hitler, Avalon, Two Rooms and a Boom, Diplomacy). Tests
    **theory-of-mind, opponent modeling, and sustained deception**.

Most LLM "game" benchmarks target the second and third categories; symmetric
perfect-information abstracts are the "control" condition.

## 2. Benchmarks for playing games (with baselines)

**[GameBench](https://arxiv.org/html/2406.06613v1)** — 9 games spanning symmetric
abstracts (Hive, Santorini) and hidden-info/asymmetric ones (Are You the Traitor?,
Two Rooms and a Boom, Sea Battle, Arctic Scavengers).
- *Baselines:* **random action selector** + **human players**; scaffolds =
  **Chain-of-Thought** and **Reasoning-via-Planning (RAP)**; models GPT-3.5 / GPT-4.
- *Finding:* **humans beat every model+scaffold config**; GPT-4 was **worse than
  random** on several games (esp. Sea Battle); CoT helped most, RAP underperformed
  despite being SOTA.

**[Board Game Arena / Game Reasoning Arena](https://arxiv.org/html/2508.03368v1)** —
OpenSpiel-based harness with unified game loops and logging.
- *Baselines:* **random agents, RL agents, and humans** alongside LLMs — the most
  apples-to-apples cross-agent setup.

**[DSGBench](https://arxiv.org/pdf/2503.06047)** — complex fixed strategy games
(StarCraft II, Civilization, Diplomacy, Werewolf, Stratego) with fine-grained
decision tracking; Diplomacy/Werewolf/Stratego are the asymmetric-information entries.

**[lmgame-Bench](https://arxiv.org/pdf/2505.15146)** — standardized gaming harness;
flags **prompt sensitivity, brittle perception, and contamination** as confounds.

**[TextArena](https://arxiv.org/abs/2504.11442) / Mindgames** — 50+ text games with
**TrueSkill/Elo-style ratings**; Mindgames operationalizes ToM: belief attribution
under hidden info, opponent modeling, cooperative inference under knowledge
asymmetry, and deception.

## 3. The asymmetric / hidden-role frontier (with baselines)

**[Evaluating LLMs in a Complex Hidden-Role Game (Secret Hitler)](https://arxiv.org/html/2605.22826v1)**
— the sharpest asymmetric study.
- *Baselines:* **Random**, **Rule-Based**, **Reputation-Based** agents, plus
  **human experts** (~1,000 competitive games from secrethitler.io). Models:
  Gemma 3, Llama 3.3 70B, Qwen 3, DeepSeek-R1-Distill-70B.
- *Metrics (role-aware):* Role Identification Accuracy, **Deception Retention
  Rate**, Game-State Impact Rate.
- *Finding:* strong models win as **Hitler (~97%)** but **fail in the
  informed-minority Fascist role** — negative impact scores, ~40% shorter games,
  leaking hidden info. **CoT and memory did not help and often hurt** (up to 23%
  worse for Fascists). Clearest evidence that *asymmetric roles expose failures
  symmetric play hides.*

**[Observer, Not Player](https://arxiv.org/pdf/2512.19210)** — probes ToM via game
*observation* rather than play.

## 4. Generating games (symmetric and asymmetric)

**[GAVEL](https://www.emergentmind.com/papers/2407.09388)** (NeurIPS) — evolutionary
search + fine-tuned LLM mutating **Ludii** game descriptions.
- *Evaluators:* compilability, **playability, balance, agency/decisiveness,
  depth**, and **Ludii concept vectors** for diversity. Can target Ludii's
  asymmetry concepts but does not center asymmetry as an experimental variable.

**[RuleSmith](https://arxiv.org/html/2602.06232v1)** — closest asymmetric-specific
work: **multi-agent LLM self-play + Bayesian optimization** to balance an
*asymmetric* game (CivMini: heterogeneous factions).
- *Baseline/objective:* **win-rate disparity** between factions; LLMs act as
  zero-shot simulators (no RL/heuristics). Balances **one hand-authored
  parameterized game** rather than generating many.

**[Ludax](https://arxiv.org/pdf/2506.22609)** — GPU-accelerated board-game DSL
(substrate for fast self-play evaluation).

**[Causal Induction from Gameplay Traces](https://arxiv.org/html/2602.00190)** —
inferring mechanics from play traces with LLMs; relevant to grounding "dynamics"
in runtime rather than static rules.

## 5. Contamination — why *generated/fresh* games matter

Fixed games leak into training data. Models "perform well on known games but fail
on rule variations" ([contamination discussion](https://arxiv.org/pdf/2505.15146)).
Procedural generation is the standard antidote:

- **[LiveBench](https://livebench.ai/livebench.pdf)** — procedurally generated
  puzzles, refreshed every 6 months, "contamination-free."
- **[NPHardEval](https://arxiv.org/pdf/2509.24210)** — algorithmically generated
  instances "provably free from training-data contamination."

Strongest argument for *generating* asymmetric games rather than reusing
Werewolf/Diplomacy: freshly generated, seed-released games are
contamination-resistant benchmark instruments.

## 6. Consolidated baseline landscape

| Baseline type | Where used | Role |
|---|---|---|
| **Random agent** | GameBench, Board Game Arena, Secret Hitler, DSGBench | Floor — LLMs sometimes *below* it |
| **Rule-based / scripted** | Secret Hitler (86.7% human-vote alignment) | Cheap strong reference; often beats LLMs |
| **Reputation / heuristic** | Secret Hitler | Models opponent tracking |
| **MCTS / flat-MC / RL** | Board Game Arena, OpenSpiel, Ludax | Skill signal; separates search from language |
| **Human players/experts** | GameBench, Board Game Arena, Secret Hitler | Ceiling — still ahead of LLMs |
| **Scaffolds (CoT, RAP, memory)** | GameBench, Secret Hitler | Inconsistent; help symmetric planning, *fail* on deception |
| **LLM self-play** | RuleSmith | Fast simulator for balancing (but a biased proxy) |

## 7. Cross-cutting findings

1. **LLMs still trail humans and often trail random/scripted baselines** on
   strategic play.
2. **Scaffolds are not reliable** — CoT helps symmetric planning but is
   neutral-to-harmful for hidden-role deception.
3. **Asymmetric roles are a distinct failure mode** — the informed-minority /
   deceiver role (Fascist, Werewolf) is where models break, invisible in
   symmetric games.
4. **Balance is evaluator-dependent** — a ruleset balanced by LLM self-play
   (RuleSmith) may not stay balanced for MCTS/RL/humans.
5. **Contamination pushes toward generated, seed-released games** with role-aware
   ratings (asymmetric Elo needs role-swapped seating — see
   [Wise 2021](https://arxiv.org/abs/2105.00839)).

## 8. Open gaps (relevant to this project)

- **No one generates *many* asymmetric games as fresh benchmark tasks** —
  RuleSmith tunes one; GAVEL generates broadly but does not isolate asymmetry;
  GameBench/DSGBench use *fixed* asymmetric games (contamination risk).
- **Few benchmarks separate *role advantage* from *agent skill*** with
  role-swapped seating.
- **Almost no head-to-head "same evaluator on symmetric vs. asymmetric generated
  games"** to prove asymmetry tests something extra.
- **Objective + subjective are siloed**: RuleSmith/DSGBench measure win rates;
  MeepleLM measures subjective persona feedback; nobody grounds persona critique
  in *executed play traces*.

The last cluster is the white space this project sits on: generate validated
asymmetric games → evaluate with role-aware random/MCTS/RL/LLM/human baselines →
optionally add a trace-grounded persona layer.

## Sources

- [GameBench](https://arxiv.org/html/2406.06613v1)
- [Board Game / Game Reasoning Arena](https://arxiv.org/html/2508.03368v1)
- [DSGBench](https://arxiv.org/pdf/2503.06047)
- [lmgame-Bench](https://arxiv.org/pdf/2505.15146)
- [TextArena](https://arxiv.org/abs/2504.11442)
- [Secret Hitler hidden-role evaluation](https://arxiv.org/html/2605.22826v1)
- [Observer, Not Player (ToM via observation)](https://arxiv.org/pdf/2512.19210)
- [GAVEL](https://www.emergentmind.com/papers/2407.09388)
- [RuleSmith](https://arxiv.org/html/2602.06232v1)
- [Ludax](https://arxiv.org/pdf/2506.22609)
- [Causal Induction from Gameplay Traces](https://arxiv.org/html/2602.00190)
- [LiveBench](https://livebench.ai/livebench.pdf)
- [NPHardEval / contamination-free generation](https://arxiv.org/pdf/2509.24210)
- [Elo for asymmetric agent tournaments](https://arxiv.org/abs/2105.00839)
