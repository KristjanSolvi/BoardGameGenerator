"""Run directory management.

Every run gets runs/<UTC timestamp>_<seed>/ holding every prompt, every
raw response, every spec revision, engine code and test output for every
attempt, validation and playtest reports, critic verdicts, the novelty
report, the final rulebook, and a run_summary.json. Nothing that touched
the model is thrown away.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Optional


class RunLog:
    def __init__(self, root: Path, run_seed: int, runs_dir: str = "runs"):
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        self.dir = root / runs_dir / f"{timestamp}_seed{run_seed}"
        self.dir.mkdir(parents=True, exist_ok=False)
        (self.dir / "calls").mkdir()
        self._call_counter = 0
        self._current_agent = "pipeline"
        self.events: list[dict[str, Any]] = []

    # -- LLM call archiving -------------------------------------------
    def set_agent(self, agent: str) -> None:
        """Label subsequent backend calls with the agent that made them."""
        self._current_agent = agent

    def observe_call(self, prompt: str, response: str, attempt: int,
                     error: Optional[str]) -> None:
        """Backend observer hook: archive every attempt verbatim."""
        self._call_counter += 1
        stem = f"{self._call_counter:03d}_{self._current_agent}_attempt{attempt}"
        (self.dir / "calls" / f"{stem}.prompt.txt").write_text(prompt)
        if response:
            (self.dir / "calls" / f"{stem}.response.txt").write_text(response)
        if error:
            (self.dir / "calls" / f"{stem}.error.txt").write_text(error)

    # -- artifacts ------------------------------------------------------
    def save_json(self, name: str, data: Any) -> Path:
        path = self.dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str) + "\n")
        return path

    def save_text(self, name: str, text: str) -> Path:
        path = self.dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text if text.endswith("\n") else text + "\n")
        return path

    def event(self, kind: str, **details: Any) -> None:
        entry = {
            "time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "kind": kind,
            **details,
        }
        self.events.append(entry)
        with open(self.dir / "events.jsonl", "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
