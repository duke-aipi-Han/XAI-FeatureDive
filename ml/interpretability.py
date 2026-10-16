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


def decision_support_hints(model: FeatureDiveCNN, inputs: Tensor, target: int, background_rgb: Tensor) -> dict:
    """First-order margin hints on an unchanged input, never measured ablation effects.

    Positive support estimates a margin loss when the feature/stroke is removed.
    The returned tensors are detached; model parameters and inputs are untouched.
    """
    if model.training or inputs.shape != (1, 3, 28, 28) or not 0 <= target < 3:
        raise ValueError('hints require one input, a valid target, and model.eval()')
    value = inputs.detach().clone().requires_grad_(True)
    logits, maps = model(value, capture=True)
    margin = margins(logits, torch.tensor([target], device=value.device))[0]
    pixel_gradient, conv1_gradient, conv2_gradient = torch.autograd.grad(
        margin, (value, maps['conv1'], maps['conv2']))
    color = background_rgb.to(value).reshape(1, 3, 1, 1)
    erase_support = -(pixel_gradient * (color - value.detach())).sum(1)[0]
    channels = {layer: {
        'support': (gradient * maps[layer]).detach().sum((0, 2, 3)),
        'mean_activation': maps[layer].detach().mean((0, 2, 3)),
    } for layer, gradient in (('conv1', conv1_gradient), ('conv2', conv2_gradient))}
    return {'erase_support': erase_support.detach(), 'channels': channels}
