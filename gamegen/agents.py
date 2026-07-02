"""LLM agent wrappers.

Each agent = one role-prompt template file in prompts/ (never hardcoded
here) + a strict parser for its output + a format-retry loop that feeds
parsing/validation errors back to the model. All calls go through the
single LLMBackend.complete() method.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

from .extraction import (ExtractionError, extract_json,
                         extract_labeled_python, extract_markdown,
                         extract_section_text)
from .schema import SpecError, validate_spec

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


class AgentError(RuntimeError):
    """An agent could not produce parseable output within the retry budget."""


def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Missing prompt template: {path}")
    return path.read_text()


def _call_with_format_retries(
    backend,
    runlog,
    agent_name: str,
    role_prompt: str,
    user_prompt: str,
    parse: Callable[[str], Any],
    format_retries: int,
    expect_json: bool = True,
) -> Any:
    """Call the backend; if parse() raises ExtractionError/SpecError, feed
    the error back and re-ask, up to format_retries times."""
    runlog.set_agent(agent_name)
    prompt = user_prompt
    last_error: Optional[str] = None
    for attempt in range(format_retries + 1):
        response = backend.complete(role_prompt, prompt, expect_json=expect_json)
        try:
            return parse(response)
        except (ExtractionError, SpecError, ValueError) as exc:
            last_error = str(exc)
            runlog.event("format_retry", agent=agent_name, attempt=attempt,
                         error=last_error[:2000])
            prompt = (
                user_prompt
                + "\n\n---\nYour previous answer could not be used. "
                + "Fix ALL of the following problems and output the complete "
                + "answer again in the required format:\n"
                + last_error
            )
    raise AgentError(
        f"{agent_name} failed to produce valid output after "
        f"{format_retries + 1} attempts. Last error:\n{last_error}"
    )


# ----------------------------------------------------------------------
def run_designer(backend, runlog, inspiration, format_retries: int,
                 revision_feedback: Optional[str] = None,
                 previous_spec: Optional[dict] = None) -> dict:
    """Invent a new game (or fully revise one, if feedback is given)."""
    if revision_feedback is None:
        role = load_prompt("designer")
        user = (
            "MECHANIC SEEDS for this run (build the game around these):\n"
            f"{inspiration.seeds_text()}\n\n"
            "FORBIDDEN GAMES (your game must not be a variant or re-skin of "
            f"any of these):\n{inspiration.forbidden_text()}\n\n"
            "Design the game now. Reason step by step first, then output the "
            "final spec JSON in one fenced ```json block."
        )
        agent_name = "designer"
    else:
        role = load_prompt("designer_revision")
        user = (
            "MECHANIC SEEDS for this run:\n"
            f"{inspiration.seeds_text()}\n\n"
            "FORBIDDEN GAMES:\n"
            f"{inspiration.forbidden_text()}\n\n"
            "CURRENT SPEC (revise this):\n```json\n"
            f"{json.dumps(previous_spec, indent=2)}\n```\n\n"
            "REQUIRED REVISIONS from the critic:\n"
            f"{revision_feedback}\n\n"
            "Produce the complete revised spec now. Reason step by step "
            "first, then output the FULL revised spec JSON (not a diff) in "
            "one fenced ```json block."
        )
        agent_name = "designer_revision"

    def parse(response: str) -> dict:
        spec = extract_json(response)
        validate_spec(spec)
        return spec

    return _call_with_format_retries(
        backend, runlog, agent_name, role, user, parse, format_retries
    )


def run_rules_engineer(backend, runlog, spec: dict, format_retries: int,
                       repair_feedback: Optional[str] = None,
                       previous_engine: Optional[str] = None,
                       previous_tests: Optional[str] = None) -> dict[str, str]:
    """Spec -> {'ENGINE': code, 'TESTS': code}; or repair a failing engine."""
    if repair_feedback is None:
        role = load_prompt("rules_engineer")
        user = (
            "GAME SPEC:\n```json\n"
            f"{json.dumps(spec, indent=2)}\n```\n\n"
            "Implement the engine and tests now. Reason step by step first, "
            "then output the two labelled sections."
        )
        agent_name = "rules_engineer"
    else:
        role = load_prompt("rules_engineer_repair")
        user = (
            "GAME SPEC:\n```json\n"
            f"{json.dumps(spec, indent=2)}\n```\n\n"
            "CURRENT ENGINE CODE (broken):\n```python\n"
            f"{previous_engine}\n```\n\n"
            "CURRENT TESTS:\n```python\n"
            f"{previous_tests}\n```\n\n"
            "FAILURE REPORT:\n"
            f"{repair_feedback}\n\n"
            "Fix the engine (and tests if they are wrong). Output BOTH "
            "complete files again in the two labelled sections."
        )
        agent_name = "rules_engineer_repair"

    def parse(response: str) -> dict[str, str]:
        if repair_feedback is not None:
            defect = extract_section_text(response, "SPEC-DEFECT")
            if defect:
                if len(defect) < 80:
                    raise ExtractionError(
                        "a SPEC-DEFECT declaration must explain the rules "
                        "defect concretely (which rule, which situation, why "
                        "no faithful engine can pass validation)"
                    )
                return {"SPEC_DEFECT": defect}
        return extract_labeled_python(response, ("ENGINE", "TESTS"))

    return _call_with_format_retries(
        backend, runlog, agent_name, role, user, parse, format_retries
    )


CRITIC_DIMENSIONS = ("balance", "decisiveness", "role_contrast", "clarity",
                     "novelty", "depth_potential")


def run_critic(backend, runlog, spec: dict,
               playtest_report: dict, format_retries: int) -> dict:
    # The spec's move_rules[].text / edge_cases ARE the rulebook source, so
    # the critic judges clarity from the same English that players will get;
    # the polished rulebook is only written after acceptance.
    role = load_prompt("critic")
    user = (
        "GAME SPEC:\n```json\n"
        f"{json.dumps(spec, indent=2)}\n```\n\n"
        "PLAYTEST REPORT (mechanical, from real self-play):\n```json\n"
        f"{json.dumps(playtest_report, indent=2)}\n```\n\n"
        "Evaluate the game now. Reason step by step first, then output the "
        "verdict JSON in one fenced ```json block."
    )

    def parse(response: str) -> dict:
        verdict = extract_json(response)
        if not isinstance(verdict, dict):
            raise ExtractionError("verdict must be a JSON object")
        scores = verdict.get("scores")
        if not isinstance(scores, dict):
            raise ExtractionError("verdict must contain a 'scores' object")
        for dim in CRITIC_DIMENSIONS:
            entry = scores.get(dim)
            if (not isinstance(entry, dict)
                    or not isinstance(entry.get("score"), int)
                    or not 1 <= entry["score"] <= 10
                    or not isinstance(entry.get("justification"), str)
                    or len(entry["justification"]) < 40):
                raise ExtractionError(
                    f"scores.{dim} must be {{'score': 1-10 int, "
                    "'justification': <one paragraph>}}"
                )
        if verdict.get("verdict") not in ("ACCEPT", "REVISE"):
            raise ExtractionError("'verdict' must be 'ACCEPT' or 'REVISE'")
        if verdict["verdict"] == "REVISE":
            revisions = verdict.get("revisions")
            if (not isinstance(revisions, list) or not revisions
                    or not all(isinstance(r, str) and len(r) > 15
                               for r in revisions)):
                raise ExtractionError(
                    "a REVISE verdict must include a non-empty 'revisions' "
                    "list of concrete rule-change instructions"
                )
        return verdict

    return _call_with_format_retries(
        backend, runlog, "critic", role, user, parse, format_retries
    )


def run_novelty_checker(backend, runlog, spec: dict,
                        format_retries: int) -> dict:
    role = load_prompt("novelty_checker")
    # deliberately given ONLY the rules (name/rationale stripped) so it
    # judges mechanics, not the designer's own novelty claims
    rules_only = {
        k: spec[k] for k in
        ("board", "pieces", "setup", "turn", "move_rules", "win_conditions",
         "draw_conditions", "repetition_rule", "edge_cases")
        if k in spec
    }
    if "loss_conditions" in spec:
        rules_only["loss_conditions"] = spec["loss_conditions"]
    user = (
        "GAME RULES (anonymized):\n```json\n"
        f"{json.dumps(rules_only, indent=2)}\n```\n\n"
        "Identify the closest known games now. Reason step by step first, "
        "then output the report JSON in one fenced ```json block."
    )

    def parse(response: str) -> dict:
        report = extract_json(response)
        if not isinstance(report, dict):
            raise ExtractionError("report must be a JSON object")
        closest = report.get("closest_games")
        if not isinstance(closest, list) or not closest:
            raise ExtractionError(
                "report must contain a non-empty 'closest_games' list"
            )
        for g in closest:
            if (not isinstance(g, dict) or not g.get("name")
                    or not isinstance(g.get("similarity_0_to_10"), int)
                    or not g.get("shared_mechanics")
                    or not g.get("key_differences")):
                raise ExtractionError(
                    "each closest_games entry needs name, similarity_0_to_10 "
                    "(int), shared_mechanics, key_differences"
                )
        if report.get("overall_judgment") not in (
                "distinct", "related_but_distinct", "near_duplicate"):
            raise ExtractionError(
                "'overall_judgment' must be distinct | related_but_distinct "
                "| near_duplicate"
            )
        if not isinstance(report.get("reviewer_summary"), str):
            raise ExtractionError("'reviewer_summary' (string) is required")
        return report

    return _call_with_format_retries(
        backend, runlog, "novelty_checker", role, user, parse, format_retries
    )


def run_rulebook_writer(backend, runlog, spec: dict,
                        format_retries: int) -> str:
    role = load_prompt("rulebook_writer")
    user = (
        "GAME SPEC:\n```json\n"
        f"{json.dumps(spec, indent=2)}\n```\n\n"
        "Write the rulebook now. Output the complete rulebook as markdown "
        "in one fenced ```markdown block."
    )

    def parse(response: str) -> str:
        text = extract_markdown(response)
        required = ("# ", "## ")
        if len(text) < 800 or not all(tok in text for tok in required):
            raise ExtractionError(
                "the rulebook must be a complete markdown document with a "
                "title and sections (overview, components, setup, turn "
                "structure, rules with examples, strategy hints)"
            )
        return text

    return _call_with_format_retries(
        backend, runlog, "rulebook_writer", role, user, parse,
        format_retries, expect_json=False,
    )
