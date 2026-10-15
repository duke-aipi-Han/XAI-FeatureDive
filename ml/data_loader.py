"""Download public bitmaps and freeze deterministic, disjoint pilot indices."""
from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ml.common import CLASSES, DEFAULT_CONFIG, ROOT, read_json, sha256, write_json

BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"
MANIFEST = ROOT / "data/manifests/pilot-v1.json"
RAW = ROOT / "data/raw"


def download(raw_dir: Path = RAW) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in CLASSES:
        destination = raw_dir / f"{name}.npy"
        if destination.exists():
            validate_array(destination)
            print(f"Using {destination}", flush=True)
            continue
        temporary = destination.with_suffix(".npy.part")
        print(f"Downloading {name}...", flush=True)
        try:
            with urllib.request.urlopen(f"{BASE_URL}/{name}.npy", timeout=120) as response:
                with temporary.open("wb") as stream:
                    shutil.copyfileobj(response, stream)
            validate_array(temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)


def validate_array(path: Path) -> np.ndarray:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.dtype != np.uint8 or array.ndim != 2 or array.shape[1] != 784:
        raise ValueError(f"Unexpected QuickDraw format: {path}")
    return array


def build_manifest(raw_dir: Path, config: dict) -> dict:
    rng = np.random.default_rng(config["split_seed"])
    partitions = ("train", "validation", "test")
    counts = {part: config[f"{part}_per_class"] for part in partitions}
    if any(not isinstance(count, int) or count < 1 for count in counts.values()):
        raise ValueError("split sizes must be positive integers")
    result = {
        "schema_version": 1, "split_seed": config["split_seed"],
        "counts_per_class": counts, "class_order": list(CLASSES),
        "source": "https://github.com/googlecreativelab/quickdraw-dataset",
        "attribution": "The Quick, Draw! Dataset, Google, Inc., CC BY 4.0",
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection": "seeded permutation of full category rows; reject exact bitmap duplicates across all selected classes/splits",
        "files": {}, "samples": [], "skipped_duplicate_bitmaps": 0,
    }
    seen = set()
    for label, name in enumerate(CLASSES):
        path = raw_dir / f"{name}.npy"
        array = validate_array(path)
        result["files"][name] = {
            "url": f"{BASE_URL}/{name}.npy", "sha256": sha256(path),
            "rows": len(array), "bytes": path.stat().st_size,
        }
        indices = iter(rng.permutation(len(array)).tolist())
        for part in partitions:
            accepted = 0
            while accepted < counts[part]:
                try:
                    index = next(indices)
                except StopIteration as error:
                    raise ValueError("Not enough unique drawings for requested split") from error
                digest = hashlib.sha256(array[index].tobytes()).hexdigest()
                if digest in seen:
                    result["skipped_duplicate_bitmaps"] += 1
                    continue
                seen.add(digest)
                result["samples"].append({
                    "id": f"{name}:{index}", "class": name, "label": label,
                    "source_index": index, "bitmap_sha256": digest, "split": part,
                })
                accepted += 1
    return result


def load_split(part: str, manifest_path: Path = MANIFEST, raw_dir: Path = RAW) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    manifest = read_json(manifest_path)
    if part not in ("train", "validation", "test"):
        raise ValueError("unknown partition")
    arrays = {}
    for name in CLASSES:
        path = raw_dir / f"{name}.npy"
        if sha256(path) != manifest["files"][name]["sha256"]:
            raise ValueError(f"Source checksum mismatch: {name}")
        arrays[name] = validate_array(path)
    samples = [row for row in manifest["samples"] if row["split"] == part]
    images = np.stack([arrays[row["class"]][row["source_index"]].reshape(28, 28) for row in samples])
    return torch.from_numpy(images), torch.tensor([row["label"] for row in samples]), samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    download()
    config = read_json(args.config)
    if MANIFEST.exists():
        existing = read_json(MANIFEST)
        expected = {part: config[f"{part}_per_class"] for part in ("train", "validation", "test")}
        if existing["split_seed"] != config["split_seed"] or existing["counts_per_class"] != expected:
            raise ValueError("Existing manifest differs; version the new experiment instead of overwriting")
        for name in CLASSES:
            if sha256(RAW / f"{name}.npy") != existing["files"][name]["sha256"]:
                raise ValueError("Source files differ from frozen manifest")
        print(f"Reusing frozen manifest: {MANIFEST}")
    else:
        write_json(MANIFEST, build_manifest(RAW, config))
        print(f"Created manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
