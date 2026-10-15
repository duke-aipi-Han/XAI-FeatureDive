"""Scientific invariants for the Phase 0 pilot, using synthetic inputs."""
import numpy as np
import pytest
import torch

from ml.backgrounds import RGB, assign_colors, composite
from ml.data_loader import build_manifest
from ml.interpretability import margins, saliency
from ml.model import FeatureDiveCNN
from ml.train import load_checkpoint


@pytest.fixture
def model():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    return FeatureDiveCNN().eval()


def test_architecture_and_post_relu_taps(model):
    inputs = torch.rand(2, 3, 28, 28)
    logits, maps = model(inputs, capture=True)
    assert sum(p.numel() for p in model.parameters()) == 26611
    assert logits.shape == (2, 3)
    assert maps['conv1'].shape == (2, 8, 28, 28)
    assert maps['conv2'].shape == (2, 16, 14, 14)
    assert maps['dense'].shape == (2, 32)
    assert all(bool((value >= 0).all()) for value in maps.values())


@pytest.mark.parametrize('layer,channel', [('conv1', 3), ('conv2', 7)])
def test_ablation_matches_independent_downstream_computation_and_resets(model, layer, channel):
    inputs = torch.rand(2, 3, 28, 28)
    baseline, original = model(inputs, capture=True)
    modified, tapped = model(inputs, capture=True, ablation=(layer, channel))
    assert torch.count_nonzero(tapped[layer][:, channel]) == 0
    other = [i for i in range(original[layer].shape[1]) if i != channel]
    torch.testing.assert_close(tapped[layer][:, other], original[layer][:, other], rtol=0, atol=0)
    value = original[layer].clone()
    value[:, channel] = 0
    value = model.pool(value)
    if layer == 'conv1':
        value = model.pool(torch.relu(model.conv2(value)))
    expected = model.output(torch.relu(model.dense(value.flatten(1))))
    torch.testing.assert_close(modified, expected, rtol=0, atol=0)
    assert not torch.equal(modified, baseline)
    torch.testing.assert_close(model(inputs), baseline, rtol=0, atol=0)


@pytest.mark.parametrize('ablation', [('dense', 0), ('conv1', -1), ('conv1', 8), ('conv2', 16)])
def test_invalid_ablation(model, ablation):
    with pytest.raises(ValueError):
        model(torch.rand(1, 3, 28, 28), ablation=ablation)


def test_compositing_preserves_alpha_and_exact_counterfactual_rule():
    drawings = torch.zeros(1, 28, 28, dtype=torch.uint8)
    drawings[0, 5, 7] = 255
    drawings[0, 5, 8] = 128
    first, alpha = composite(drawings, torch.tensor([0]))
    second, other_alpha = composite(drawings, torch.tensor([1]))
    torch.testing.assert_close(alpha, other_alpha, rtol=0, atol=0)
    assert torch.equal(first[:, :, 5, 7], second[:, :, 5, 7])
    difference = (torch.tensor(RGB[0]) - torch.tensor(RGB[1])).float()[None, :, None, None] / 255
    torch.testing.assert_close(first - second, (1 - alpha) * difference, atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(alpha[0, 0, 5, 8], torch.tensor(128 / 255))


def test_split_replay_and_duplicate_exclusion(tmp_path):
    for label, name in enumerate(('cat', 'fish', 'airplane')):
        values = np.arange(20, dtype=np.uint8)[:, None] + 30 * label
        values = np.repeat(values, 784, axis=1)
        values[1] = values[0]
        np.save(tmp_path / f'{name}.npy', values)
    config = {'split_seed': 590, 'train_per_class': 4, 'validation_per_class': 2, 'test_per_class': 2}
    first = build_manifest(tmp_path, config)
    second = build_manifest(tmp_path, config)
    assert first['samples'] == second['samples']
    assert len(first['samples']) == 24
    assert len({row['id'] for row in first['samples']}) == 24
    assert len({row['bitmap_sha256'] for row in first['samples']}) == 24
    for part, count in first['counts_per_class'].items():
        for label in range(3):
            assert sum(row['split'] == part and row['label'] == label for row in first['samples']) == count


def test_color_assignment_probabilities_and_replay():
    labels = np.repeat(np.arange(3), 10000)
    for regime in ('balanced', 'shortcut'):
        colors = assign_colors(labels, regime, 591)
        np.testing.assert_array_equal(colors, assign_colors(labels, regime, 591))
        for label in range(3):
            frequencies = np.bincount(colors[labels == label], minlength=3) / 10000
            expected = np.full(3, 1 / 3 if regime == 'balanced' else .05)
            if regime == 'shortcut':
                expected[label] = .9
            np.testing.assert_allclose(frequencies, expected, atol=.02)


def test_saliency_has_input_gradients_without_parameter_mutation(model):
    inputs = torch.rand(1, 3, 28, 28)
    baseline = model(inputs).detach()
    heatmap = saliency(model, inputs, 1)
    assert heatmap.shape == (1, 28, 28)
    assert bool(torch.isfinite(heatmap).all())
    assert float(heatmap.sum()) > 0
    assert all(parameter.grad is None for parameter in model.parameters())
    assert inputs.grad is None
    torch.testing.assert_close(model(inputs), baseline, rtol=0, atol=0)
    differentiable = inputs.clone().requires_grad_(True)
    gradient, = torch.autograd.grad(model(differentiable)[0, 1], differentiable)
    strongest = int(gradient.abs().flatten().argmax())
    step = torch.zeros_like(inputs)
    step.flatten()[strongest] = 1e-3
    difference = (model(inputs + step)[0, 1] - model(inputs - step)[0, 1]) / 2e-3
    torch.testing.assert_close(difference, gradient.flatten()[strongest], atol=2e-4, rtol=.05)


def test_target_margin_uses_strongest_other_class():
    logits = torch.tensor([[1., 3., 2.], [4., 1., -2.]])
    torch.testing.assert_close(margins(logits, torch.tensor([1, 2])), torch.tensor([1., -6.]))


def test_checkpoint_roundtrip(model, tmp_path, monkeypatch):
    # Exercise the same safe CPU checkpoint loader used by evaluation.
    from ml import train
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}')
    monkeypatch.setattr(train, 'MANIFEST', manifest)
    checkpoint = tmp_path / 'model.pt'
    from ml.common import sha256
    torch.save({'state_dict': model.state_dict(), 'metadata': {
        'manifest_sha256': sha256(manifest), 'class_order': ['cat', 'fish', 'airplane'],
    }}, checkpoint)
    restored, _ = load_checkpoint(checkpoint)
    inputs = torch.rand(1, 3, 28, 28)
    assert not restored.training
    torch.testing.assert_close(model(inputs), restored(inputs), rtol=0, atol=0)
    manifest.write_text('{"different": true}')
    with pytest.raises(ValueError, match='different manifest'):
        load_checkpoint(checkpoint)


def test_paired_evaluation_metrics_and_signed_swaps(tmp_path, monkeypatch):
    from ml import evaluate
    checkpoint_dir = tmp_path / 'checkpoints'
    checkpoint_dir.mkdir()
    (checkpoint_dir / 'pilot-shortcut.pt').write_bytes(b'test checkpoint identity')
    monkeypatch.setattr(evaluate, 'ROOT', tmp_path)
    monkeypatch.setattr(evaluate, 'OUTPUT', tmp_path)
    monkeypatch.setattr(evaluate, 'load_checkpoint', lambda path: (None, {'best_epoch': 1, 'validation_accuracy': .5}))
    # Source 0 always predicts cat; source 1 predicts fish only on blue;
    # source 2 always predicts airplane. Gray is outside the 3-color metric.
    predicted = torch.tensor([[0, 0, 0, 2], [0, 1, 2, 0], [2, 2, 2, 2]])
    logits = torch.nn.functional.one_hot(predicted, num_classes=3).float() * 2
    monkeypatch.setattr(evaluate, 'predict', lambda model, drawings, color, batch_size: logits[:, color])
    samples = [{'id': f'{name}:0', 'class': name} for name in ('cat', 'fish', 'airplane')]
    labels = torch.arange(3)
    summary, rows, swaps = evaluate.evaluate_model(
        'shortcut', {'batch_size': 3, 'split_seed': 590, 'bootstrap_replicates': 50},
        torch.zeros(3, 28, 28, dtype=torch.uint8), labels, samples,
    )
    assert summary['accuracy']['balanced_colors']['mean'] == pytest.approx(7 / 9)
    assert summary['accuracy']['training_like_expected']['mean'] == pytest.approx(2.9 / 3)
    assert summary['any_color_flip_rate']['mean'] == pytest.approx(1 / 3)
    assert summary['pairwise_color_flip_rate']['mean'] == pytest.approx(1 / 3)
    assert summary['accuracy']['gray']['mean'] == pytest.approx(1 / 3)
    assert len(rows) == 12
    assert len(swaps) == 18
    fish_to_blue = next(row for row in swaps if row['sample_id'] == 'fish:0' and row['original_background'] == 'green' and row['replacement_background'] == 'blue')
    assert fish_to_blue['classification_flip'] == 1
    assert fish_to_blue['target_margin_change'] == pytest.approx(4)
    assert fish_to_blue['target_probability_change'] > 0
    assert sum(sum(row) for row in summary['confusion_matrices']['blue']) == 3
