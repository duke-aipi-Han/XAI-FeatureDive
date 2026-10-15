"""Download public bitmaps and freeze deterministic, disjoint experiment indices."""
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


def build_manifest(raw_dir: Path, config: dict, exclusions: list[Path] | None = None) -> dict:
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
    excluded_sources = set()
    excluded_files = {}
    for excluded_path in exclusions or []:
        excluded = read_json(excluded_path)
        for row in excluded['samples']:
            seen.add(row['bitmap_sha256'])
            excluded_sources.add(row['id'])
        excluded_files.update(excluded['files'])
        result.setdefault('excluded_manifests', []).append({
            'path': str(excluded_path.relative_to(ROOT)) if excluded_path.is_relative_to(ROOT) else str(excluded_path),
            'sha256': sha256(excluded_path), 'sources': len(excluded['samples']),
        })
    if exclusions:
        result['selection'] += '; exclude all prior experiment source IDs and bitmap hashes'
        result['skipped_prior_experiment_sources'] = 0
    for label, name in enumerate(CLASSES):
        path = raw_dir / f"{name}.npy"
        array = validate_array(path)
        result["files"][name] = {
            "url": f"{BASE_URL}/{name}.npy", "sha256": sha256(path),
            "rows": len(array), "bytes": path.stat().st_size,
        }
        if name in excluded_files and excluded_files[name]['sha256'] != result['files'][name]['sha256']:
            raise ValueError('Excluded experiment refers to different source file bytes')
        indices = iter(rng.permutation(len(array)).tolist())
        for part in partitions:
            accepted = 0
            while accepted < counts[part]:
                try:
                    index = next(indices)
                except StopIteration as error:
                    raise ValueError("Not enough unique drawings for requested split") from error
                digest = hashlib.sha256(array[index].tobytes()).hexdigest()
                if f'{name}:{index}' in excluded_sources:
                    result['skipped_prior_experiment_sources'] += 1
                    continue
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


def validate_manifest(manifest: dict) -> None:
    """Reject corrupt labels, duplicate sources, leakage, and incorrect counts."""
    if manifest['class_order'] != list(CLASSES):
        raise ValueError('Manifest class order differs')
    sources, bitmaps = set(), set()
    counts = {(part, label): 0 for part in ('train', 'validation', 'test') for label in range(3)}
    for row in manifest['samples']:
        label = row['label']
        if type(label) is not int or not 0 <= label < 3 or row['class'] != CLASSES[label]:
            raise ValueError('Invalid source label')
        index = row['source_index']
        if type(index) is not int or not 0 <= index < manifest['files'][row['class']]['rows']:
            raise ValueError('Invalid source index')
        if row['id'] != f"{row['class']}:{index}" or row['id'] in sources or row['bitmap_sha256'] in bitmaps:
            raise ValueError('Duplicate or inconsistent source identity')
        if row['split'] not in manifest['counts_per_class']:
            raise ValueError('Invalid source partition')
        sources.add(row['id'])
        bitmaps.add(row['bitmap_sha256'])
        counts[row['split'], label] += 1
    for (part, label), count in counts.items():
        if count != manifest['counts_per_class'][part]:
            raise ValueError('Manifest partition counts differ')


def verify_sources(manifest: dict, raw_dir: Path = RAW) -> None:
    validate_manifest(manifest)
    for name in CLASSES:
        if sha256(raw_dir / f'{name}.npy') != manifest['files'][name]['sha256']:
            raise ValueError(f'Source checksum mismatch: {name}')


def verify_config(manifest: dict, config: dict) -> None:
    expected = {part: config[f'{part}_per_class'] for part in ('train', 'validation', 'test')}
    if manifest['split_seed'] != config['split_seed'] or manifest['counts_per_class'] != expected:
        raise ValueError('Manifest differs from experiment split configuration')


def load_split(part: str, manifest_path: Path = MANIFEST, raw_dir: Path = RAW) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    manifest = read_json(manifest_path)
    if part not in ("train", "validation", "test"):
        raise ValueError("unknown partition")
    verify_sources(manifest, raw_dir)
    arrays = {}
    for name in CLASSES:
        path = raw_dir / f"{name}.npy"
        arrays[name] = validate_array(path)
    samples = [row for row in manifest["samples"] if row["split"] == part]
    images = np.stack([arrays[row["class"]][row["source_index"]].reshape(28, 28) for row in samples])
    return torch.from_numpy(images), torch.tensor([row["label"] for row in samples]), samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = read_json(args.config)
    from ml.experiment import Experiment
    experiment = Experiment.from_config(config)
    exclusions = [ROOT / path for path in config.get('exclude_manifests', [])]
    download()
    if experiment.manifest.exists():
        existing = read_json(experiment.manifest)
        verify_config(existing, config)
        verify_sources(existing)
        recorded = [row['sha256'] for row in existing.get('excluded_manifests', [])]
        if recorded != [sha256(path) for path in exclusions]:
            raise ValueError('Prior-experiment exclusion manifests differ')
        print(f"Reusing frozen manifest: {experiment.manifest}")
    else:
        manifest = build_manifest(RAW, config, exclusions)
        validate_manifest(manifest)
        write_json(experiment.manifest, manifest)
        print(f"Created manifest: {experiment.manifest}")


if __name__ == "__main__":
    main()
