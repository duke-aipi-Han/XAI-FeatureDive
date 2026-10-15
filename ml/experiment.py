"""Validated experiment paths and explicit CPU/MPS training selection."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from ml.common import ROOT


@dataclass(frozen=True)
class Experiment:
    phase: int
    version: str
    manifest: Path
    results: Path
    checkpoint_directory: Path

    @classmethod
    def from_config(cls, config: dict) -> Experiment:
        for name in ('split_seed', 'training_seed', 'background_seed', 'threads',
                     'train_per_class', 'validation_per_class', 'test_per_class',
                     'epochs', 'batch_size', 'bootstrap_replicates'):
            value = config.get(name)
            if type(value) is not int or value < (0 if name.endswith('_seed') else 1):
                raise ValueError(f'{name} must be a valid positive integer (seeds may be zero)')
        rate = config.get('learning_rate')
        if not isinstance(rate, (int, float)) or not 0 < rate < 1:
            raise ValueError('learning_rate must be between zero and one')
        phase = config.get('phase', 0)
        if phase not in (0, 1):
            raise ValueError('Only Phase 0 and Phase 1 are implemented')
        if config.get('device', 'cpu') not in ('auto', 'cpu', 'mps'):
            raise ValueError('device must be auto, cpu, or mps')
        return cls(
            phase=phase, version=config.get('model_version', 'pilot-v1'),
            manifest=ROOT / config.get('manifest', 'data/manifests/pilot-v1.json'),
            results=ROOT / config.get('results_directory', 'results/phase0'),
            checkpoint_directory=ROOT / config.get('checkpoint_directory', 'checkpoints'),
        )

    def checkpoint(self, regime: str) -> Path:
        if regime not in ('balanced', 'shortcut'):
            raise ValueError('Unknown model regime')
        prefix = 'pilot-' if self.phase == 0 else ''
        return self.checkpoint_directory / f'{prefix}{regime}.pt'


def select_device(requested: str) -> torch.device:
    if requested not in ('auto', 'cpu', 'mps'):
        raise ValueError('device must be auto, cpu, or mps')
    available = torch.backends.mps.is_available()
    if requested == 'mps' and not available:
        raise RuntimeError('MPS was explicitly requested but is unavailable in this process')
    return torch.device('mps' if requested == 'mps' or (requested == 'auto' and available) else 'cpu')
