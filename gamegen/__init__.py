"""gamegen — multi-agent LLM pipeline that invents new two-player
asymmetric abstract board games (two structurally different roles,
rough balance as the target), validates them mechanically, playtests
them, and writes human-readable rulebooks.

All LLM calls go through a locally installed, subscription-authenticated
CLI (OpenAI Codex CLI by default). No API keys are read or required.
"""

__version__ = "0.1.0"
