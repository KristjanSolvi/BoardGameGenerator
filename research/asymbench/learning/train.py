from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn


def train_steps(
    model: nn.Module,
    buffer: Any,
    batch_size: int,
    steps: int,
    lr: float,
    device: str | torch.device = "cpu",
) -> dict[str, float | int]:
    """Train a policy/value network on replay examples for a fixed step count."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if lr <= 0.0:
        raise ValueError("lr must be positive")

    torch_device = torch.device(device)
    model.to(torch_device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    policy_loss_total = 0.0
    value_loss_total = 0.0
    total_loss_total = 0.0

    for _ in range(steps):
        batch = buffer.sample(batch_size)
        (
            observations,
            roles,
            action_masks,
            target_policies,
            target_values,
        ) = _batch_tensors(
            batch,
            torch_device,
        )

        logits, values = model(observations, roles, action_masks)
        log_probs = torch.log_softmax(logits, dim=1)
        policy_loss = -(target_policies * log_probs).sum(dim=1).mean()
        value_loss = torch.nn.functional.mse_loss(values, target_values)
        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        policy_loss_total += float(policy_loss.detach().cpu().item())
        value_loss_total += float(value_loss.detach().cpu().item())
        total_loss_total += float(loss.detach().cpu().item())

    model.train()
    return {
        "steps": steps,
        "policy_loss": policy_loss_total / steps,
        "value_loss": value_loss_total / steps,
        "total_loss": total_loss_total / steps,
    }


def _batch_tensors(
    batch: list[Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    observations = np.stack(
        [np.asarray(example.observation, dtype=np.float32) for example in batch]
    )
    roles = np.asarray([example.role for example in batch], dtype=np.int64)
    action_masks = np.stack(
        [np.asarray(example.action_mask, dtype=np.bool_) for example in batch]
    )
    policies = np.stack(
        [np.asarray(example.policy, dtype=np.float32) for example in batch]
    )
    values = np.asarray([example.value for example in batch], dtype=np.float32)

    _validate_target_policies(policies, action_masks)

    return (
        torch.as_tensor(observations, dtype=torch.float32, device=device),
        torch.as_tensor(roles, dtype=torch.long, device=device),
        torch.as_tensor(action_masks, dtype=torch.bool, device=device),
        torch.as_tensor(policies, dtype=torch.float32, device=device),
        torch.as_tensor(values, dtype=torch.float32, device=device),
    )


def _validate_target_policies(policies: np.ndarray, action_masks: np.ndarray) -> None:
    if policies.shape != action_masks.shape:
        raise ValueError("target policies must have the same shape as action masks")
    if not np.all(np.isfinite(policies)):
        raise ValueError("target policies must contain only finite values")
    if np.any(policies < -1e-8):
        raise ValueError("target policies must not contain negative probabilities")
    if float(np.abs(policies[~action_masks]).sum()) > 1e-6:
        raise ValueError("target policies must assign zero mass to illegal actions")
    row_sums = policies.sum(axis=1)
    if not np.all(row_sums > 0.0):
        raise ValueError("target policies must have positive probability mass")


__all__ = ["train_steps"]
