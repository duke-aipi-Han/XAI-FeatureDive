"""Train comparable CPU pilot models; choose epochs using balanced validation."""
from __future__ import annotations

import argparse
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
from ml.data_loader import MANIFEST, load_split
from ml.model import FeatureDiveCNN


def configure(seed: int, threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def balanced_accuracy(model: FeatureDiveCNN, drawings: torch.Tensor, labels: torch.Tensor, batch_size: int) -> float:
    correct = 0
    model.eval()
    with torch.no_grad():
        for color in range(3):
            for start in range(0, len(labels), batch_size):
                source = drawings[start:start + batch_size]
                inputs, _ = composite(source, torch.full((len(source),), color, dtype=torch.long))
                correct += int((model(inputs).argmax(1) == labels[start:start + batch_size]).sum())
    return correct / (3 * len(labels))


def train(regime: str, config: dict, train_data: tuple, validation_data: tuple) -> dict:
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
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    history = []
    best_accuracy = -1.0
    checkpoint_path = ROOT / "checkpoints" / f"pilot-{regime}.pt"
    start_time = time.monotonic()
    counts = [[int(((labels == label) & (colors == color)).sum()) for color in range(3)] for label in range(3)]
    metadata = {
        "schema_version": 1, "regime": regime, "config": config,
        "class_order": list(CLASSES), "background_names": list(COLOR_NAMES), "background_rgb": list(RGB),
        "input_normalization": "uint8/255 alpha; white strokes; RGB in [0,1]",
        "optimizer": "Adam (PyTorch default betas, eps, weight_decay=0)",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "manifest_sha256": sha256(MANIFEST), "training_background_counts_by_class": counts,
        "epoch_selection": "maximum validation accuracy averaged over all three training colors; first best epoch",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__, "platform": platform.platform(), "device": "cpu"},
    }
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        loss_total = 0.0
        correct = 0
        for batch, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach()) * len(targets)
            correct += int((logits.argmax(1) == targets).sum())
        accuracy = balanced_accuracy(model, validation_drawings, validation_labels, config["batch_size"])
        row = {"epoch": epoch, "train_loss": loss_total / len(labels), "train_accuracy": correct / len(labels), "balanced_validation_accuracy": accuracy}
        history.append(row)
        print(f"{regime} epoch {epoch:02d}: loss={row['train_loss']:.4f} train={row['train_accuracy']:.3f} validation={accuracy:.3f}", flush=True)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({"state_dict": model.state_dict(), "metadata": metadata, "best_epoch": epoch, "validation_accuracy": accuracy}, checkpoint_path)
    summary = {**metadata, "history": history, "elapsed_seconds": time.monotonic() - start_time, "checkpoint": str(checkpoint_path.relative_to(ROOT)), "checkpoint_sha256": sha256(checkpoint_path)}
    write_json(ROOT / "results/phase0" / f"training-{regime}.json", summary)
    return summary


def load_checkpoint(path: Path) -> tuple[FeatureDiveCNN, dict]:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["metadata"]["manifest_sha256"] != sha256(MANIFEST):
        raise ValueError("Checkpoint was trained with a different manifest")
    if saved["metadata"]["class_order"] != list(CLASSES):
        raise ValueError("Checkpoint class order differs")
    model = FeatureDiveCNN()
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--regime", choices=("balanced", "shortcut", "both"), default="both")
    args = parser.parse_args()
    config = read_json(args.config)
    train_data = load_split("train")
    validation_data = load_split("validation")
    for regime in (("balanced", "shortcut") if args.regime == "both" else (args.regime,)):
        train(regime, config, train_data, validation_data)


if __name__ == "__main__":
    main()
