"""Counterfactual RGB inputs with unchanged white foreground alpha."""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

# Color order matches the class-correlated colors: cat, fish, airplane.
COLOR_NAMES = ("green", "blue", "yellow", "gray")
RGB = ((70, 170, 90), (65, 120, 210), (230, 195, 55), (128, 128, 128))


def composite(drawings: Tensor, colors: Tensor) -> tuple[Tensor, Tensor]:
    """Blend source intensity alpha with white strokes, returning RGB and alpha.

    QuickDraw bitmaps have zero background and high-intensity strokes. Inputs
    are uint8 [N,28,28]; colors are integer indices [N]. RGB is in [0,1], NCHW.
    Anti-aliasing is preserved without thresholding or geometric transforms.
    """
    if drawings.dtype != torch.uint8 or drawings.ndim != 3 or drawings.shape[1:] != (28, 28):
        raise ValueError("drawings must be uint8 [N,28,28]")
    if colors.shape != (len(drawings),) or colors.dtype != torch.long:
        raise ValueError("colors must be int64 [N]")
    if bool(((colors < 0) | (colors >= len(RGB))).any()):
        raise ValueError("invalid background index")
    alpha = drawings.float().unsqueeze(1) / 255
    palette = torch.tensor(RGB, dtype=torch.float32, device=drawings.device) / 255
    background = palette[colors].unsqueeze(-1).unsqueeze(-1)
    return alpha + (1 - alpha) * background, alpha


def assign_colors(labels: np.ndarray, regime: str, seed: int) -> np.ndarray:
    """Fixed per-source assignments; neutral gray is evaluation-only."""
    if regime not in ("balanced", "shortcut"):
        raise ValueError("regime must be balanced or shortcut")
    rng = np.random.default_rng(seed)
    probabilities = np.full((len(labels), 3), 1 / 3)
    if regime == "shortcut":
        probabilities[:] = 0.05
        probabilities[np.arange(len(labels)), labels] = 0.9
    draws = rng.random(len(labels))
    return (draws[:, None] > probabilities.cumsum(axis=1)).sum(axis=1).astype(np.int64)
