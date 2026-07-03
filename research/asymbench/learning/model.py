from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


class PolicyValueNet(nn.Module):
    """Small policy/value network with optional role-specific heads."""

    def __init__(
        self,
        input_shape: Sequence[int],
        action_size: int,
        num_roles: int = 2,
        role_heads: bool = False,
        channels: int = 64,
    ) -> None:
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError("input_shape must be (channels, height, width)")
        input_channels, height, width = (int(dim) for dim in input_shape)
        if input_channels <= 0 or height <= 0 or width <= 0:
            raise ValueError("input_shape dimensions must be positive")
        if action_size <= 0:
            raise ValueError("action_size must be positive")
        if num_roles <= 0:
            raise ValueError("num_roles must be positive")
        if channels <= 0:
            raise ValueError("channels must be positive")

        self.input_shape = (input_channels, height, width)
        self.action_size = int(action_size)
        self.num_roles = int(num_roles)
        self.role_heads = bool(role_heads)

        self.trunk = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(channels * height * width, 128),
            nn.ReLU(),
        )

        if self.role_heads:
            self.policy_heads = nn.ModuleList(
                nn.Linear(128, self.action_size) for _ in range(self.num_roles)
            )
            self.value_heads = nn.ModuleList(
                nn.Linear(128, 1) for _ in range(self.num_roles)
            )
        else:
            self.policy_head = nn.Linear(128, self.action_size)
            self.value_head = nn.Linear(128, 1)

    def forward(
        self, observations: torch.Tensor, roles: torch.Tensor, action_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if observations.ndim != 4 or tuple(observations.shape[1:]) != self.input_shape:
            raise ValueError(
                "observations must have shape "
                f"(batch, {self.input_shape[0]}, {self.input_shape[1]}, {self.input_shape[2]})"
            )
        batch_size = observations.shape[0]
        if roles.shape != (batch_size,):
            raise ValueError("roles must have shape (batch,)")
        if roles.dtype not in _INTEGER_DTYPES:
            raise ValueError("roles must use an integer dtype")
        if action_mask.shape != (batch_size, self.action_size):
            raise ValueError("action_mask must have shape (batch, action_size)")
        if action_mask.dtype != torch.bool:
            raise ValueError("action_mask must use bool dtype")
        if not bool(action_mask.any(dim=1).all().item()):
            raise ValueError(
                "action_mask must contain at least one legal action per row"
            )
        if (
            roles.device != observations.device
            or action_mask.device != observations.device
        ):
            raise ValueError("roles and action_mask must be on the observations device")
        if torch.any((roles < 0) | (roles >= self.num_roles)):
            raise ValueError(f"roles must be in [0, {self.num_roles})")

        features = self.trunk(observations)
        if self.role_heads:
            logits = features.new_empty((batch_size, self.action_size))
            values = features.new_empty((batch_size,))
            for role_id in range(self.num_roles):
                selected = roles == role_id
                if not torch.any(selected):
                    continue
                logits[selected] = self.policy_heads[role_id](features[selected])
                values[selected] = torch.tanh(
                    self.value_heads[role_id](features[selected])
                ).squeeze(-1)
        else:
            logits = self.policy_head(features)
            values = torch.tanh(self.value_head(features)).squeeze(-1)

        mask_value = torch.finfo(logits.dtype).min
        masked_logits = logits.masked_fill(~action_mask, mask_value)
        return masked_logits, values
