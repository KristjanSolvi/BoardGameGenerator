# gamegen

A research pipeline that uses a multi-agent LLM workflow to invent
completely new two-player **symmetric abstract board games**, compile
them into executable engines, validate and playtest them mechanically,
and produce human-readable rulebooks. Built for reproducibility: every
prompt, response, spec revision, engine attempt, and metric of every run
is archived.

## How it works

```
inspiration sampler (no LLM: mechanic seeds + forbidden-games list)
        │
        ▼
   DESIGNER ──────────────── spec JSON (strict schema, gamegen/schema.py)
        │                          ▲
        ▼                          │ numbered revisions (max 4 cycles)
 RULES ENGINEER ── engine.py + test_engine.py
        │    ▲
        ▼    │ traceback feedback (max 3 repair rounds)
  VALIDATOR (no LLM: pytest, 1000 random playouts, termination,
             move soundness, symmetry via color-swap mirror, hashing)
        │
        ▼
 PLAYTEST HARNESS (no LLM: random & flat-Monte-Carlo matches
                   → playtest_report.json)
        │
        ▼
    CRITIC ── ACCEPT ──► NOVELTY CHECKER (logged, never auto-fails)
        │                        │
        └── REVISE ──► designer  ▼
                          RULEBOOK WRITER ──► rulebook.md
```

### The agents

| Agent | LLM | Prompt file | Job |
|---|---|---|---|
| Inspiration sampler | no | — | Samples 2–3 mechanic seeds (goal type always included) and the forbidden list of ~40 famous abstracts; forces design diversity across runs. |
| Designer | yes | `prompts/designer.md` | Invents a complete symmetric game as strict spec JSON, with a design rationale arguing non-equivalence to known games. |
| Designer (revision) | yes | `prompts/designer_revision.md` | Re-issues the full spec addressing the critic's numbered revisions. |
| Rules engineer | yes | `prompts/rules_engineer.md` | Compiles the spec into a pure, deterministic Python engine + pytest tests. |
| Rules engineer (repair) | yes | `prompts/rules_engineer_repair.md` | Fixes the engine given the validator's failure report. |
| Validator | no | — | Hard checks: import, tests, 1000 terminating playouts, legal/illegal move soundness, symmetry isomorphism on move one, deterministic state hashing. |
| Playtest harness | no | — | Random vs random, flat-MC vs random (both colors), MC vs MC; length distribution, first-player win rate, draw rate, decisiveness, branching factor. |
| Critic | yes | `prompts/critic.md` | Scores balance/decisiveness/clarity/novelty/depth 1–10 with justifications; ACCEPT or concrete numbered revisions. |
| Novelty checker | yes | `prompts/novelty_checker.md` | Adversarial reviewer; sees only anonymized rules; names closest known games with similarity scores. Logged for manual review, never auto-fails a run. |
| Rulebook writer | yes | `prompts/rulebook_writer.md` | Markdown rulebook for human participants: components, ASCII setup diagram, every rule with a worked example, edge-case Q&A, strategy hints. |

Prompts are Markdown files in `prompts/` — never hardcoded — so wording
can be iterated without touching code.

## Installation

Requires Python ≥ 3.10, `pyyaml`, `pytest`:

```
pip install pyyaml pytest
```

### LLM backend: OpenAI Codex CLI (default)

All LLM calls shell out to the **locally installed Codex CLI** using its
logged-in ChatGPT subscription. No API SDK, no `OPENAI_API_KEY`, no
credentials anywhere in this repo.

1. Install the Codex CLI (https://developers.openai.com/codex/cli), then:
2. `codex login` — sign in with your ChatGPT account.
3. Check with `codex login status` (the pipeline fails fast with
   instructions if the CLI is missing or logged out).

Verified against codex-cli 0.142.4. Each call runs:

```
codex exec --skip-git-repo-check --ephemeral -s read-only --color never \
    -m <model> -o <output-file>          # prompt piped via stdin
```

`config.yaml` selects the model (`gpt-5.5`) and optionally overrides
`model_reasoning_effort`.

### Adding a new backend

`gamegen/backend.py` defines the whole contract: one class, one method
`complete(role_prompt, user_prompt, expect_json) -> str`. Subclass
`LLMBackend`, override `_check_available()` (fail fast with actionable
instructions) and `_one_call(prompt) -> (response | None, error | None)`
(retries/timeouts are handled by the base class), then register the class
in `make_backend()` and name it in `config.yaml`. A `claude` backend
(Claude Code headless mode, `claude -p`, also subscription-based) is
already included as the reference second implementation.

## Usage

```
python -m gamegen generate --runs 5 --seed 42     # generate games
python -m gamegen replay runs/<run_dir>           # re-run playtests on an existing game
python -m gamegen generate --config other.yaml    # alternate config
```

`config.yaml` holds the backend, model, revision/repair/retry limits,
playout counts, and move caps; see the comments in the file.

Reproducibility: run *i* of a batch uses seed `seed + i`, threaded through
the inspiration sampler, validator, and playtest harness. LLM outputs are
not bit-reproducible (the model is not temperature-controllable through
the CLI), which is why every raw response is archived; `replay` re-runs
the deterministic half of the pipeline exactly.

## Run directory layout

```
runs/<UTC-timestamp>_seed<seed>/
├── config_used.json            exact config + run seed
├── seeds.json                  sampled mechanic seeds + forbidden list
├── events.jsonl                timestamped pipeline events
├── calls/                      EVERY prompt & raw response, numbered:
│   └── 007_critic_attempt0.{prompt,response}.txt
├── spec_rev0.json …            spec at each revision
├── spec_final.json             accepted spec
├── engine/
│   ├── rev0_attempt0/          every engineering attempt:
│   │   ├── engine.py           generated engine
│   │   ├── test_engine.py      generated pytest suite
│   │   └── validation.json     validator report for this attempt
│   ├── engine.py               final accepted engine
│   └── test_engine.py          final tests
├── playtest_report_rev0.json … metrics per revision
├── playtest_report.json        metrics for the accepted game
├── critic_rev0.json …          critic verdicts per revision
├── novelty_report.json         adversarial novelty review
├── rulebook.md                 human-facing rulebook
└── run_summary.json            status, game name, headline metrics
```

Failed runs keep everything up to the failure plus a `failure_reason` in
`run_summary.json` (and `error_traceback.txt` for crashes).

## Development

`python -m pytest tests/` tests the pipeline itself (schema, extraction,
validator, playtest harness) against a hand-written reference engine and
deliberately broken engines — no LLM needed.
