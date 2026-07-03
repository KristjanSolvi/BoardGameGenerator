from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research.asymbench.generation.connection_disruption import (
    ConnectionDisruptionGenerator,
)
from research.asymbench.generation.escape_capture import EscapeCaptureGenerator
from research.asymbench.generation.specs import GeneratedGameSpec


GENERATORS = {
    "escape_capture": EscapeCaptureGenerator(),
    "connection_disruption": ConnectionDisruptionGenerator(),
}


def load_generated_spec(path: Path) -> GeneratedGameSpec:
    return GeneratedGameSpec.from_dict(json.loads(path.read_text()))


def compile_generated_game(spec: GeneratedGameSpec) -> Any:
    generator = GENERATORS.get(spec.family)
    if generator is None:
        raise ValueError(f"unknown family: {spec.family!r}")
    return generator.compile(spec)
