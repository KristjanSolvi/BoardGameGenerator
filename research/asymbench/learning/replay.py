from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrainingExample:
    observation: np.ndarray
    role: int
    action_mask: np.ndarray
    policy: np.ndarray
    value: float


class ReplayBuffer:
    """Fixed-capacity replay buffer with deterministic sampling."""

    def __init__(self, capacity: int, seed: int | None = None) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._examples: deque[TrainingExample] = deque(maxlen=capacity)
        self._random = random.Random(seed)

    def add(self, example: TrainingExample) -> None:
        self._examples.append(example)

    def sample(self, batch_size: int) -> list[TrainingExample]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self._examples):
            raise ValueError(
                f"cannot sample {batch_size} examples from replay buffer with "
                f"{len(self._examples)} examples"
            )
        return self._random.sample(list(self._examples), batch_size)

    def __len__(self) -> int:
        return len(self._examples)
