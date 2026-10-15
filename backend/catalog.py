"""Freeze a small illustrative catalog from training sources, never test data."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from ml.common import CLASSES, ROOT, read_json, sha256, write_json
from ml.data_loader import RAW, validate_array, verify_sources

DEFAULT_CONFIG = ROOT / 'configs/phase2.json'


def create_catalog(config: dict) -> dict:
    registry = read_json(ROOT / config['registry'])
    manifest_path = ROOT / registry['manifest']
    if sha256(manifest_path) != registry['manifest_sha256']:
        raise ValueError('Registry manifest checksum mismatch')
    manifest = read_json(manifest_path)
    verify_sources(manifest)
    count = config['demo_per_class']
    if type(count) is not int or not 1 <= count <= 20:
        raise ValueError('demo_per_class must be between 1 and 20')
    samples = []
    for name in CLASSES:
        candidates = [row for row in manifest['samples'] if row['split'] == 'train' and row['class'] == name]
        if len(candidates) < count:
            raise ValueError('Not enough training examples')
        samples.extend(candidates[:count])
    catalog = {
        'schema_version': 1, 'model_version': registry['model_version'],
        'parent_manifest': registry['manifest'], 'parent_manifest_sha256': registry['manifest_sha256'],
        'selection': f'First {count} training sources per class in frozen manifest order; illustrative, seen during training; excludes validation and test sources',
        'samples': samples,
    }
    destination = ROOT / config['demo_manifest']
    if destination.exists():
        if read_json(destination) != catalog:
            raise ValueError('Existing demo catalog differs; choose a new version')
    else:
        write_json(destination, catalog)
    # Small source-only bundle permits offline serving without bulk raw arrays.
    arrays = {name: validate_array(RAW / f'{name}.npy') for name in CLASSES}
    drawings = np.stack([arrays[row['class']][row['source_index']].reshape(28, 28) for row in samples])
    for row, drawing in zip(samples, drawings):
        if hashlib.sha256(drawing.tobytes()).hexdigest() != row['bitmap_sha256']:
            raise ValueError('Source bitmap checksum mismatch')
    bundle = destination.with_suffix('.npz')
    if bundle.exists():
        with np.load(bundle, allow_pickle=False) as saved:
            if not np.array_equal(saved['drawings'], drawings) or saved['sample_ids'].tolist() != [row['id'] for row in samples]:
                raise ValueError('Existing sample bundle differs')
    else:
        np.savez_compressed(bundle, drawings=drawings, sample_ids=np.array([row['id'] for row in samples]))
    return catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    catalog = create_catalog(read_json(args.config))
    print(f"Prepared {len(catalog['samples'])} illustrative training samples and compact source bundle")


if __name__ == '__main__':
    main()
