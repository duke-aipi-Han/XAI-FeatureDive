"""Phase 1 data isolation, device choice, and recorded-training invariants."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from ml.common import ROOT, read_json, sha256, write_json
from ml.data_loader import build_manifest, load_split, validate_manifest, verify_config
from ml.experiment import Experiment, select_device
from ml.train import load_checkpoint, train, weights_sha256


@pytest.fixture
def sources(tmp_path):
    for label, name in enumerate(('cat', 'fish', 'airplane')):
        # Distinct synthetic sources, with one duplicate within each category.
        values = np.repeat((np.arange(50, dtype=np.uint8) + label * 60)[:, None], 784, axis=1)
        values[1] = values[0]
        np.save(tmp_path / f'{name}.npy', values)
    return tmp_path


@pytest.fixture
def split_config():
    return {'split_seed': 590, 'train_per_class': 4, 'validation_per_class': 2, 'test_per_class': 2}


def test_final_splits_exclude_all_pilot_sources_and_bitmap_copies(sources, split_config):
    pilot = build_manifest(sources, split_config)
    pilot_path = sources / 'pilot.json'
    write_json(pilot_path, pilot)
    final = build_manifest(sources, {**split_config, 'split_seed': 1590}, [pilot_path])
    validate_manifest(final)
    assert not ({row['id'] for row in pilot['samples']} & {row['id'] for row in final['samples']})
    assert not ({row['bitmap_sha256'] for row in pilot['samples']} & {row['bitmap_sha256'] for row in final['samples']})
    assert final['excluded_manifests'][0]['sha256'] == sha256(pilot_path)
    replay = build_manifest(sources, {**split_config, 'split_seed': 1590}, [pilot_path])
    assert replay['samples'] == final['samples']


@pytest.mark.parametrize('corruption', ['split', 'label', 'index', 'duplicate', 'count'])
def test_corrupt_manifest_is_rejected(sources, split_config, corruption):
    manifest = build_manifest(sources, split_config)
    if corruption == 'split':
        manifest['samples'][0]['split'] = 'test'
    elif corruption == 'label':
        manifest['samples'][0]['label'] = 2
    elif corruption == 'index':
        manifest['samples'][0]['source_index'] = -1
    elif corruption == 'duplicate':
        manifest['samples'][1] = deepcopy(manifest['samples'][0])
    else:
        manifest['counts_per_class']['train'] += 1
    with pytest.raises(ValueError):
        validate_manifest(manifest)


def test_data_load_rejects_changed_source_and_configuration(sources, split_config):
    manifest = build_manifest(sources, split_config)
    manifest_path = sources / 'manifest.json'
    write_json(manifest_path, manifest)
    drawings, labels, samples = load_split('test', manifest_path, sources)
    assert drawings.shape == (6, 28, 28) and drawings.dtype == torch.uint8
    assert labels.tolist() == [row['label'] for row in samples]
    with pytest.raises(ValueError, match='configuration'):
        verify_config(manifest, {**split_config, 'split_seed': 591})
    path = sources / 'cat.npy'
    changed = np.load(path).copy()
    changed[0, 0] += 1
    np.save(path, changed)
    with pytest.raises(ValueError, match='checksum'):
        load_split('test', manifest_path, sources)


def test_auto_device_prefers_mps_and_explicit_mps_never_silently_falls_back(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: True)
    assert select_device('auto').type == 'mps'
    assert select_device('cpu').type == 'cpu'
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    assert select_device('auto').type == 'cpu'
    with pytest.raises(RuntimeError, match='explicitly requested'):
        select_device('mps')


def test_experiments_do_not_share_manifests_or_checkpoint_paths():
    pilot = Experiment.from_config(read_json(ROOT / 'configs/phase0.json'))
    final = Experiment.from_config(read_json(ROOT / 'configs/phase1.json'))
    assert pilot.manifest != final.manifest
    assert pilot.results != final.results
    assert pilot.checkpoint('balanced') != final.checkpoint('balanced')
    assert pilot.checkpoint('balanced').name == 'pilot-balanced.pt'
    assert final.checkpoint('balanced').name == 'balanced.pt'
    config = read_json(ROOT / 'configs/phase1.json')
    with pytest.raises(ValueError, match='epochs'):
        Experiment.from_config({**config, 'epochs': 0})


def test_training_records_are_replayable_and_weights_are_saved_for_cpu(tmp_path):
    manifest = tmp_path / 'manifest.json'
    write_json(manifest, {'synthetic_test_only': True})
    config = {**read_json(ROOT / 'configs/phase0.json'), 'epochs': 2, 'batch_size': 3, 'threads': 1}
    data = torch.randint(0, 256, (9, 28, 28), generator=torch.Generator().manual_seed(2), dtype=torch.uint8)
    labels = torch.arange(9) % 3
    dataset = (data, labels, [])
    checkpoints = []
    for index in range(2):
        path = tmp_path / f'run{index}' / 'balanced.pt'
        summary = train('balanced', config, dataset, dataset, manifest_path=manifest,
                        checkpoint_path=path, output_dir=path.parent)
        model, saved = load_checkpoint(path, manifest)
        assert not model.training
        assert all(value.device.type == 'cpu' for value in saved['state_dict'].values())
        assert saved['weights_sha256'] == weights_sha256(model.state_dict())
        assert summary['status'] == 'complete'
        assert summary['checkpoint_sha256'] == sha256(path)
        assert summary['deterministic_algorithms'] is True
        assert summary['initial_weights_sha256']
        assert len(summary['history']) == 2
        checkpoints.append(saved)
    assert checkpoints[0]['weights_sha256'] == checkpoints[1]['weights_sha256']
    assert checkpoints[0]['validation_accuracy'] == checkpoints[1]['validation_accuracy']
    with pytest.raises(FileExistsError):
        train('balanced', config, dataset, dataset, manifest_path=manifest,
              checkpoint_path=tmp_path / 'run0' / 'balanced.pt', output_dir=tmp_path)
    checkpoints[0]['state_dict']['output.bias'][0] += 1
    bad_path = tmp_path / 'tampered.pt'
    torch.save(checkpoints[0], bad_path)
    with pytest.raises(ValueError, match='weight checksum'):
        load_checkpoint(bad_path, manifest)
