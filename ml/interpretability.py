"""Pilot attribution and logit metrics; no semantic channel claims."""
from __future__ import annotations

import torch
from torch import Tensor

from ml.model import FeatureDiveCNN


def margins(logits: Tensor, targets: Tensor) -> Tensor:
    selected = logits.gather(1, targets[:, None]).squeeze(1)
    others = logits.clone()
    others.scatter_(1, targets[:, None], -torch.inf)
    return selected - others.max(dim=1).values


def saliency(
    model: FeatureDiveCNN, inputs: Tensor, target: int,
    *, ablation: tuple[str, int] | None = None,
) -> Tensor:
    """Unnormalized sum of absolute RGB gradients of the selected class logit."""
    if model.training:
        raise ValueError("saliency requires model.eval()")
    if not 0 <= target < 3:
        raise ValueError("target must be in [0,2]")
    value = inputs.detach().clone().requires_grad_(True)
    logits = model(value, ablation=ablation)
    gradient, = torch.autograd.grad(logits[:, target].sum(), value)
    return gradient.abs().sum(dim=1).detach()
