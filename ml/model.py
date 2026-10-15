"""The specified 26,611-parameter CNN, with explicit post-ReLU taps."""
from __future__ import annotations

import torch
from torch import Tensor, nn


class FeatureDiveCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dense = nn.Linear(16 * 7 * 7, 32)
        self.output = nn.Linear(32, 3)

    def forward(
        self, inputs: Tensor, *, capture: bool = False,
        ablation: tuple[str, int] | None = None,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if inputs.ndim != 4 or inputs.shape[1:] != (3, 28, 28):
            raise ValueError("inputs must have shape [N,3,28,28]")
        if ablation is not None:
            layer, channel = ablation
            limits = {"conv1": 8, "conv2": 16}
            if layer not in limits or not isinstance(channel, int) or not 0 <= channel < limits[layer]:
                raise ValueError("invalid convolutional ablation")
        activations = {}
        value = inputs
        for name, convolution in (("conv1", self.conv1), ("conv2", self.conv2)):
            value = torch.relu(convolution(value))
            if ablation is not None and ablation[0] == name:
                value = value.clone()
                value[:, ablation[1]] = 0
            if capture:
                activations[name] = value
            value = self.pool(value)
        value = torch.relu(self.dense(value.flatten(1)))
        if capture:
            activations["dense"] = value
        logits = self.output(value)
        return (logits, activations) if capture else logits
