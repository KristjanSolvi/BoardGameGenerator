"""Strict extraction of structured content from model responses.

Agents are instructed to reason freely and then emit their final answer in
fenced blocks. These parsers take the LAST matching block (the final
answer), never the first, so chain-of-thought examples don't get picked up.
"""

from __future__ import annotations

import json
import re
from typing import Any

FENCE_RE = re.compile(
    r"```([a-zA-Z0-9_+-]*)[ \t]*\r?\n(.*?)\r?\n?```", re.DOTALL
)


class ExtractionError(ValueError):
    """The response did not contain the required fenced block."""


def _blocks(text: str, language: str | None) -> list[str]:
    out = []
    for match in FENCE_RE.finditer(text):
        lang, body = match.group(1).lower(), match.group(2)
        if language is None or lang == language:
            out.append(body)
    return out


def extract_json(text: str) -> Any:
    """Parse the last ```json block (falling back to the last unlabeled
    fenced block that parses as JSON). Raises ExtractionError with a
    message suitable for feeding back to the agent."""
    candidates = _blocks(text, "json") or _blocks(text, "")
    if not candidates:
        raise ExtractionError(
            "No fenced ```json block found in the response. Emit the final "
            "JSON inside a single fenced block labelled `json`."
        )
    last_error = None
    for body in reversed(candidates):
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            last_error = exc
    raise ExtractionError(
        f"The final fenced JSON block is not valid JSON: {last_error}. "
        "Emit strictly valid JSON (double quotes, no trailing commas, no "
        "comments)."
    )


def extract_labeled_python(text: str, labels: tuple[str, ...]) -> dict[str, str]:
    """Extract python blocks that follow '### <LABEL>' headers.

    The rules engineer emits:
        ### ENGINE
        ```python
        ...
        ```
        ### TESTS
        ```python
        ...
        ```
    Returns {label: code}. Raises ExtractionError if any label is missing.
    """
    result: dict[str, str] = {}
    for label in labels:
        # last occurrence of the header, then the first python block after it
        headers = [m.end() for m in re.finditer(
            rf"^###\s*{re.escape(label)}\s*$", text, re.MULTILINE
        )]
        if not headers:
            raise ExtractionError(
                f"Missing '### {label}' section. Output exactly the sections "
                + ", ".join(f"'### {l}'" for l in labels)
                + ", each followed by one fenced ```python block."
            )
        tail = text[headers[-1]:]
        match = FENCE_RE.search(tail)
        if not match or match.group(1).lower() not in ("python", "py", ""):
            raise ExtractionError(
                f"'### {label}' must be followed by a fenced ```python block."
            )
        result[label] = match.group(2)
    return result


def extract_section_text(text: str, label: str) -> str | None:
    """Return the plain text after the last '### <label>' header, up to
    the next '### ' header or end of text; None if the header is absent."""
    headers = [m.end() for m in re.finditer(
        rf"^###\s*{re.escape(label)}\s*$", text, re.MULTILINE
    )]
    if not headers:
        return None
    tail = text[headers[-1]:]
    nxt = re.search(r"^###\s", tail, re.MULTILINE)
    return (tail[:nxt.start()] if nxt else tail).strip()


def extract_markdown(text: str) -> str:
    """Extract the last ```markdown block; if none, return the whole
    response (the rulebook writer may answer in plain markdown)."""
    blocks = _blocks(text, "markdown") or _blocks(text, "md")
    return blocks[-1] if blocks else text.strip()
