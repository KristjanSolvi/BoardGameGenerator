"""LLM backends.

Every backend shells out to a locally installed, subscription-authenticated
CLI. No API SDKs are imported, no API keys are read, and no credentials
appear anywhere in this codebase.

Default backend: OpenAI Codex CLI in non-interactive mode.
Verified against codex-cli 0.142.4 (`codex exec --help`):

    codex exec --skip-git-repo-check --ephemeral -s read-only \
        -m <model> -o <last_message_file>        # prompt read from stdin

  - the prompt is piped over stdin (no argv length limits),
  - `-o/--output-last-message` writes ONLY the agent's final message to a
    file, which is what complete() returns,
  - `--ephemeral` avoids persisting a session per call,
  - `-s read-only` sandboxes the call; the role prompts additionally
    instruct the model to act as a pure text generator,
  - `-c model_reasoning_effort=...` optionally overrides the user's
    config.toml default (xhigh can be needlessly slow for some agents).

A second backend using `claude -p` (Claude Code headless mode, also
subscription-based) implements the same interface; select it with
`backend: claude` in config.yaml.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

# Called before each attempt with (prompt, attempt_index) and after with the
# raw response — the run logger hooks in here so every prompt/response pair
# is archived verbatim.
CallObserver = Callable[[str, str, int, Optional[str]], None]


class BackendError(RuntimeError):
    """The CLI backend failed after all retries."""


class BackendNotAvailableError(BackendError):
    """The CLI is missing or not logged in — fail fast with instructions."""


class LLMBackend:
    """Backend using the OpenAI Codex CLI (`codex exec`).

    One public method:

        complete(role_prompt, user_prompt, expect_json) -> str

    The role prompt and user prompt are concatenated into a single
    non-interactive exec call; the model's final message is returned
    verbatim. When expect_json is True the response must contain a fenced
    ```json block (checked here so malformed responses are retried at the
    transport layer before the orchestrator ever sees them).
    """

    name = "codex"

    def __init__(
        self,
        model: str,
        timeout_seconds: int = 900,
        max_retries: int = 2,
        reasoning_effort: Optional[str] = None,
        observer: Optional[CallObserver] = None,
    ):
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.reasoning_effort = reasoning_effort
        self.observer = observer
        self._check_available()

    # ------------------------------------------------------------------
    def _check_available(self) -> None:
        if shutil.which("codex") is None:
            raise BackendNotAvailableError(
                "The `codex` CLI is not installed or not on PATH. Install the "
                "OpenAI Codex CLI and authenticate with `codex login` "
                "(ChatGPT subscription login; no API key needed)."
            )
        try:
            proc = subprocess.run(
                ["codex", "login", "status"],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired as exc:
            raise BackendNotAvailableError(
                "`codex login status` timed out; check your Codex CLI install "
                "with `codex doctor`."
            ) from exc
        status = (proc.stdout + proc.stderr).lower()
        if proc.returncode != 0 or "logged in" not in status:
            raise BackendNotAvailableError(
                "The Codex CLI is installed but not logged in. Run "
                "`codex login` and sign in with your ChatGPT account, then "
                "re-run this pipeline."
            )

    # ------------------------------------------------------------------
    def _build_command(self, output_file: Path) -> list[str]:
        cmd = [
            "codex",
            "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "-s",
            "read-only",
            "--color",
            "never",
            "-m",
            self.model,
            "-o",
            str(output_file),
        ]
        if self.reasoning_effort:
            cmd += ["-c", f'model_reasoning_effort="{self.reasoning_effort}"']
        return cmd

    def complete(self, role_prompt: str, user_prompt: str, expect_json: bool) -> str:
        prompt = f"{role_prompt.rstrip()}\n\n{user_prompt.strip()}\n"
        last_error: Optional[str] = None
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                time.sleep(min(10.0, 2.0 ** attempt))
            response, error = self._one_call(prompt)
            if self.observer is not None:
                self.observer(prompt, response or "", attempt, error)
            if error is not None:
                last_error = error
                continue
            assert response is not None
            if expect_json and "```" not in response:
                last_error = (
                    "response contained no fenced code block although JSON "
                    "output was required"
                )
                continue
            return response
        raise BackendError(
            f"codex exec failed after {self.max_retries + 1} attempts; "
            f"last error: {last_error}"
        )

    def _one_call(self, prompt: str) -> tuple[Optional[str], Optional[str]]:
        """Returns (response, None) on success or (None|partial, error)."""
        with tempfile.NamedTemporaryFile(
            mode="r", suffix=".txt", prefix="gamegen_codex_", delete=False
        ) as tmp:
            output_file = Path(tmp.name)
        try:
            try:
                proc = subprocess.run(
                    self._build_command(output_file),
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                return None, f"timed out after {self.timeout_seconds}s"
            if proc.returncode != 0:
                tail = (proc.stderr or proc.stdout or "").strip()[-2000:]
                return None, f"exit code {proc.returncode}: {tail}"
            if not output_file.exists() or not output_file.read_text().strip():
                return None, "codex exec produced no final message"
            return output_file.read_text().strip(), None
        finally:
            output_file.unlink(missing_ok=True)


class ClaudeBackend(LLMBackend):
    """Backend using Claude Code headless mode (`claude -p`).

    Verified against the installed `claude --help`: `-p/--print` gives
    non-interactive output on stdout; `--model` selects the model. Also
    subscription-authenticated — no API key involved.
    """

    name = "claude"

    def _check_available(self) -> None:
        if shutil.which("claude") is None:
            raise BackendNotAvailableError(
                "The `claude` CLI is not installed or not on PATH. Install "
                "Claude Code and authenticate (run `claude` once "
                "interactively), then re-run this pipeline."
            )

    def _build_command(self, output_file: Path) -> list[str]:
        # claude -p prints the final response on stdout; we redirect it to
        # the output file in _one_call via capture, so just record the path.
        self._stdout_to_file = output_file
        return ["claude", "-p", "--model", self.model]

    def _one_call(self, prompt: str) -> tuple[Optional[str], Optional[str]]:
        try:
            proc = subprocess.run(
                ["claude", "-p", "--model", self.model],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return None, f"timed out after {self.timeout_seconds}s"
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip()[-2000:]
            return None, f"exit code {proc.returncode}: {tail}"
        if not proc.stdout.strip():
            return None, "claude -p produced empty output"
        return proc.stdout.strip(), None


def make_backend(config, observer: Optional[CallObserver] = None) -> LLMBackend:
    """Instantiate the backend named in config.yaml.

    To add a new backend: subclass LLMBackend, override _check_available()
    and _one_call(), and register it here.
    """
    cls = {"codex": LLMBackend, "claude": ClaudeBackend}[config.backend]
    return cls(
        model=config.model,
        timeout_seconds=int(config.llm["timeout_seconds"]),
        max_retries=int(config.llm["max_retries"]),
        reasoning_effort=config.reasoning_effort,
        observer=observer,
    )
