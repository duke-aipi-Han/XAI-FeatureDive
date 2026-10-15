"""Train comparable CPU/MPS models; choose epochs using balanced validation."""
from __future__ import annotations

import argparse
import hashlib
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml.backgrounds import COLOR_NAMES, RGB, assign_colors, composite
from ml.common import CLASSES, DEFAULT_CONFIG, ROOT, read_json, sha256, write_json
from ml.data_loader import MANIFEST, load_split, verify_config
from ml.experiment import Experiment, select_device
from ml.model import FeatureDiveCNN


def configure(seed: int, threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def balanced_accuracy(model: FeatureDiveCNN, drawings: torch.Tensor, labels: torch.Tensor, batch_size: int) -> float:
    correct = 0
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for color in range(3):
            for start in range(0, len(labels), batch_size):
                source = drawings[start:start + batch_size]
                inputs, _ = composite(source, torch.full((len(source),), color, dtype=torch.long))
                predictions = model(inputs.to(device)).argmax(1).cpu()
                correct += int((predictions == labels[start:start + batch_size]).sum())
    return correct / (3 * len(labels))


def train(
    regime: str, config: dict, train_data: tuple, validation_data: tuple,
    *, manifest_path: Path | None = None, checkpoint_path: Path | None = None,
    output_dir: Path | None = None, device: torch.device | None = None,
    overwrite: bool = False,
) -> dict:
    manifest_path = MANIFEST if manifest_path is None else manifest_path
    checkpoint_path = ROOT / 'checkpoints' / f'pilot-{regime}.pt' if checkpoint_path is None else checkpoint_path
    output_dir = ROOT / 'results/phase0' if output_dir is None else output_dir
    device = torch.device('cpu') if device is None else device
    if checkpoint_path.exists() and not overwrite:
        raise FileExistsError(f'{checkpoint_path} exists; use --overwrite to replace a recorded run')
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure(config["training_seed"], config["threads"])
    drawings, labels, _ = train_data
    validation_drawings, validation_labels, _ = validation_data
    colors = torch.from_numpy(assign_colors(labels.numpy(), regime, config["background_seed"]))
    inputs, _ = composite(drawings, colors)
    loader = DataLoader(
        TensorDataset(inputs, labels), batch_size=config["batch_size"], shuffle=True,
        generator=torch.Generator().manual_seed(config["training_seed"]), num_workers=0,
    )
    model = FeatureDiveCNN()
    initial_weights_sha256 = weights_sha256(model.state_dict())
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    history = []
    best_accuracy = -1.0
    start_time = time.monotonic()
    counts = [[int(((labels == label) & (colors == color)).sum()) for color in range(3)] for label in range(3)]
    metadata = {
        "schema_version": 2, "regime": regime, "config": config,
        "model_version": config.get('model_version', 'pilot-v1'),
        "initial_weights_sha256": initial_weights_sha256,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "class_order": list(CLASSES), "background_names": list(COLOR_NAMES), "background_rgb": list(RGB),
        "input_normalization": "uint8/255 alpha; white strokes; RGB in [0,1]",
        "optimizer": "Adam (PyTorch default betas, eps, weight_decay=0)",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "manifest_sha256": sha256(manifest_path), "training_background_counts_by_class": counts,
        "training_colors_sha256": hashlib.sha256(colors.numpy().tobytes()).hexdigest(),
        "source_code_sha256": {name: sha256(ROOT / 'ml' / name) for name in ('train.py', 'model.py', 'backgrounds.py', 'data_loader.py', 'experiment.py')},
        "dependency_lock_sha256": sha256(ROOT / 'uv.lock'),
        "epoch_selection": "maximum validation accuracy averaged over all three training colors; first best epoch",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__, "platform": platform.platform(), "device": str(device)},
    }
    write_json(output_dir / f'training-{regime}.json', {**metadata, 'status': 'running', 'history': history})
    print(f'Training {regime} on {device}; {len(labels):,} sources; {config["epochs"]} epochs', flush=True)
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        epoch_start = time.monotonic()
        loss_total = 0.0
        correct = 0
        for batch, targets in loader:
            batch, targets = batch.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach().cpu()) * len(targets)
            correct += int((logits.argmax(1) == targets).sum().cpu())
        accuracy = balanced_accuracy(model, validation_drawings, validation_labels, config["batch_size"])
        row = {"epoch": epoch, "train_loss": loss_total / len(labels), "train_accuracy": correct / len(labels), "balanced_validation_accuracy": accuracy, "elapsed_seconds": time.monotonic() - epoch_start}
        history.append(row)
        print(f"{regime} epoch {epoch:02d}: loss={row['train_loss']:.4f} train={row['train_accuracy']:.3f} validation={accuracy:.3f}", flush=True)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            temporary = checkpoint_path.with_suffix('.pt.tmp')
            torch.save({"state_dict": state, "metadata": metadata, "best_epoch": epoch, "validation_accuracy": accuracy, 'weights_sha256': weights_sha256(state)}, temporary)
            temporary.replace(checkpoint_path)
        write_json(output_dir / f'training-{regime}.json', {**metadata, 'status': 'running', 'history': history})
    checkpoint_label = str(checkpoint_path.relative_to(ROOT)) if checkpoint_path.is_relative_to(ROOT) else str(checkpoint_path)
    summary = {**metadata, 'status': 'complete', "history": history, "elapsed_seconds": time.monotonic() - start_time, "checkpoint": checkpoint_label, "checkpoint_sha256": sha256(checkpoint_path)}
    write_json(output_dir / f'training-{regime}.json', summary)
    return summary


def weights_sha256(state: dict[str, torch.Tensor]) -> str:
    """Hash tensor contents independently of checkpoint archive serialization."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((str(value.dtype), list(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_checkpoint(path: Path, manifest_path: Path | None = None) -> tuple[FeatureDiveCNN, dict]:
    manifest_path = MANIFEST if manifest_path is None else manifest_path
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["metadata"]["manifest_sha256"] != sha256(manifest_path):
        raise ValueError("Checkpoint was trained with a different manifest")
    if saved["metadata"]["class_order"] != list(CLASSES):
        raise ValueError("Checkpoint class order differs")
    if 'weights_sha256' in saved and weights_sha256(saved['state_dict']) != saved['weights_sha256']:
        raise ValueError('Checkpoint weight checksum differs')
    model = FeatureDiveCNN()
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--regime", choices=("balanced", "shortcut", "both"), default="both")
    parser.add_argument('--device', choices=('auto', 'cpu', 'mps'), help='Override configured training device; auto prefers MPS')
    parser.add_argument('--overwrite', action='store_true', help='Explicitly replace an existing recorded run')
    args = parser.parse_args()
    config = read_json(args.config)
    experiment = Experiment.from_config(config)
    device = select_device(args.device or config.get('device', 'cpu'))
    verify_config(read_json(experiment.manifest), config)
    regimes = ('balanced', 'shortcut') if args.regime == 'both' else (args.regime,)
    if not args.overwrite:
        for regime in regimes:
            if experiment.checkpoint(regime).exists():
                parser.error(f'{experiment.checkpoint(regime)} exists; use --overwrite or a new experiment version')
    train_data = load_split("train", experiment.manifest)
    validation_data = load_split("validation", experiment.manifest)
    for regime in regimes:
        train(regime, config, train_data, validation_data, manifest_path=experiment.manifest,
              checkpoint_path=experiment.checkpoint(regime), output_dir=experiment.results,
              device=device, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
