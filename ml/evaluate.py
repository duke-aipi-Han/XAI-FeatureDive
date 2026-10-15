"""Measured held-out color swaps and illustrative pilot interventions."""
from __future__ import annotations

import argparse
import csv
import itertools
import platform
from pathlib import Path

import numpy as np
import torch

from ml.backgrounds import COLOR_NAMES, RGB, composite
from ml.common import CLASSES, DEFAULT_CONFIG, ROOT, read_json, sha256, write_json
from ml.data_loader import MANIFEST, load_split, verify_config
from ml.experiment import Experiment
from ml.interpretability import margins, saliency
from ml.train import configure, load_checkpoint

OUTPUT = ROOT / "results/phase0"


def estimate(values: np.ndarray, seed: int, replicates: int) -> dict:
    """Percentile CI resampling source drawings, retaining paired colors."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(replicates, len(values)))
    means = values[draws].mean(axis=1)
    return {"mean": float(values.mean()), "ci95": np.quantile(means, [.025, .975]).tolist()}


def predict(model, drawings, color: int, batch_size: int) -> torch.Tensor:
    outputs = []
    with torch.no_grad():
        for source in drawings.split(batch_size):
            inputs, _ = composite(source, torch.full((len(source),), color, dtype=torch.long))
            outputs.append(model(inputs))
    return torch.cat(outputs)


def output_record(logits: torch.Tensor, target: int) -> dict:
    return {
        "logits": logits[0].detach().tolist(),
        "probabilities": logits[0].detach().softmax(0).tolist(),
        "predicted_class": CLASSES[int(logits[0].argmax())],
        "target_class": CLASSES[target],
        "target_logit_margin": float(margins(logits, torch.tensor([target]))[0].detach()),
    }


def verify_swap_geometry(drawings: torch.Tensor) -> dict:
    first, alpha = composite(drawings, torch.zeros(len(drawings), dtype=torch.long))
    maximum_error = 0.0
    for color in range(1, 4):
        changed, other_alpha = composite(drawings, torch.full((len(drawings),), color, dtype=torch.long))
        if not torch.equal(alpha, other_alpha):
            raise AssertionError("Foreground alpha changed")
        delta = (torch.tensor(RGB[0]) - torch.tensor(RGB[color])).float()[None, :, None, None] / 255
        expected = (1 - alpha) * delta
        torch.testing.assert_close(first - changed, expected, atol=1e-7, rtol=1e-5)
        maximum_error = max(maximum_error, float((first - changed - expected).abs().max()))
    return {"drawings_checked": len(drawings), "unchanged_alpha": True, "max_counterfactual_formula_error": maximum_error}


def evaluate_model(
    regime: str, config: dict, drawings, labels, samples,
    *, model_path: Path | None = None, manifest_path: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[dict, list[dict], list[dict]]:
    model_path = ROOT / 'checkpoints' / f'pilot-{regime}.pt' if model_path is None else model_path
    output_dir = OUTPUT if output_dir is None else output_dir
    model, checkpoint = load_checkpoint(model_path) if manifest_path is None else load_checkpoint(model_path, manifest_path)
    if manifest_path is not None and checkpoint['metadata']['config'] != config:
        raise ValueError('Checkpoint and evaluation configurations differ')
    logits = torch.stack([predict(model, drawings, color, config['batch_size']) for color in range(4)], dim=1)
    checks = {}
    if manifest_path is not None:
        assert not model.training and all(parameter.device.type == 'cpu' for parameter in model.parameters())
        repeated = torch.stack([predict(model, drawings, color, config['batch_size']) for color in range(4)], dim=1)
        torch.testing.assert_close(logits, repeated, atol=0, rtol=0)
        if not bool(torch.isfinite(logits).all()):
            raise ValueError('Checkpoint produced nonfinite logits')
        checks = {'cpu_eval_mode': True, 'all_parameters_cpu': True, 'repeat_inference_exact': True, 'finite_logits': True, 'source_color_inputs_checked': len(labels) * 4}
    predictions = logits.argmax(2)
    probabilities = logits.softmax(2)
    correct = predictions.eq(labels[:, None]).numpy()
    target_margins = torch.stack([margins(logits[:, color], labels) for color in range(4)], dim=1).numpy()
    target_probabilities = probabilities.gather(2, labels[:, None, None].expand(-1, 4, 1)).squeeze(2).numpy()
    source_indices = np.arange(len(labels))
    preferred_correct = correct[source_indices, labels.numpy()]
    offset1 = correct[source_indices, (labels.numpy() + 1) % 3]
    offset2 = correct[source_indices, (labels.numpy() + 2) % 3]
    training_like = correct[:, :3].mean(1) if regime == 'balanced' else .9 * preferred_correct + .05 * offset1 + .05 * offset2
    any_flip = (predictions[:, :3] != predictions[:, :1]).any(1).numpy()
    pair_flip = np.stack([(predictions[:, a] != predictions[:, b]).numpy() for a, b in itertools.combinations(range(3), 2)]).mean(0)
    margin_range = np.ptp(target_margins[:, :3], axis=1)
    probability_range = np.ptp(target_probabilities[:, :3], axis=1)
    confusion = {}
    for color, name in enumerate(COLOR_NAMES):
        matrix = np.zeros((3, 3), dtype=int)
        np.add.at(matrix, (labels.numpy(), predictions[:, color].numpy()), 1)
        confusion[name] = matrix.tolist()
    ci = lambda values: estimate(values, config['split_seed'], config['bootstrap_replicates'])
    summary = {
        "regime": regime, "checkpoint_sha256": sha256(model_path),
        "checkpoint": str(model_path.relative_to(ROOT)), "best_epoch": checkpoint['best_epoch'],
        "validation_accuracy": checkpoint['validation_accuracy'], "source_drawings": len(labels),
        "accuracy": {
            "balanced_colors": ci(correct[:, :3].mean(1)),
            "training_like_expected": ci(training_like),
            "preferred_mapping": ci(preferred_correct),
            "swapped_mapping_1": ci(offset1), "swapped_mapping_2": ci(offset2),
            **{name: ci(correct[:, color]) for color, name in enumerate(COLOR_NAMES)},
        },
        "any_color_flip_rate": ci(any_flip), "pairwise_color_flip_rate": ci(pair_flip),
        "target_margin_range": ci(margin_range), "target_probability_range": ci(probability_range),
        "confusion_matrices": confusion,
        "inference_checks": checks,
    }
    rows = []
    for index, sample in enumerate(samples):
        for color, name in enumerate(COLOR_NAMES):
            rows.append({
                "model": regime, "checkpoint_sha256": summary['checkpoint_sha256'],
                "sample_id": sample['id'], "target_class": sample['class'], "background": name,
                "predicted_class": CLASSES[int(predictions[index, color])], "correct": int(correct[index, color]),
                "target_margin": float(target_margins[index, color]),
                **{f"logit_{label}": float(logits[index, color, k]) for k, label in enumerate(CLASSES)},
                **{f"probability_{label}": float(probabilities[index, color, k]) for k, label in enumerate(CLASSES)},
            })
    swaps = []
    for index, sample in enumerate(samples):
        for first, second in itertools.combinations(range(4), 2):
            swaps.append({
                "model": regime, "checkpoint_sha256": summary['checkpoint_sha256'],
                "sample_id": sample['id'], "target_class": sample['class'],
                "original_background": COLOR_NAMES[first], "replacement_background": COLOR_NAMES[second],
                "classification_flip": int(predictions[index, first] != predictions[index, second]),
                "target_margin_change": float(target_margins[index, second] - target_margins[index, first]),
                "target_probability_change": float(target_probabilities[index, second] - target_probabilities[index, first]),
            })
    np.savez_compressed(output_dir / f"paired-{regime}.npz", logits=logits.numpy(), labels=labels.numpy(), sample_ids=np.array([sample['id'] for sample in samples]))
    return summary, rows, swaps


def illustrative_examples(config: dict, drawings, labels, samples) -> list[dict]:
    """First held-out sample per class; no search for spectacular effects."""
    examples = []
    for regime in ('balanced', 'shortcut'):
        model_path = ROOT / 'checkpoints' / f'pilot-{regime}.pt'
        model, _ = load_checkpoint(model_path)
        for target in range(3):
            index = int(torch.where(labels == target)[0][0])
            drawing = drawings[index:index + 1]
            inputs, alpha = composite(drawing, torch.tensor([target]))
            with torch.no_grad():
                baseline, maps = model(inputs, capture=True)
                repeated = model(inputs)
            torch.testing.assert_close(baseline, repeated, atol=0, rtol=0)
            raw_saliency = saliency(model, inputs, target)
            raw = {'source': drawing.numpy(), 'input': inputs.numpy(), 'alpha': alpha.numpy(), 'saliency': raw_saliency.numpy()}
            raw.update({name: activation.numpy() for name, activation in maps.items()})
            ablations = []
            for layer in ('conv1', 'conv2'):
                means = maps[layer].mean(dim=(0, 2, 3))
                # Explicitly select the most active channel, not the largest effect.
                channel = int(means.argmax())
                with torch.no_grad():
                    modified, changed_maps = model(inputs, capture=True, ablation=(layer, channel))
                    restored = model(inputs)
                assert int(torch.count_nonzero(changed_maps[layer][:, channel])) == 0
                others = [i for i in range(maps[layer].shape[1]) if i != channel]
                torch.testing.assert_close(changed_maps[layer][:, others], maps[layer][:, others], atol=0, rtol=0)
                torch.testing.assert_close(restored, baseline, atol=0, rtol=0)
                record = output_record(modified, target)
                ablations.append({"layer": layer, "channel": channel, "selection": "highest mean post-ReLU activation", "mean_activation": float(means[channel]), "modified": record, "target_margin_change": record['target_logit_margin'] - output_record(baseline, target)['target_logit_margin']})
                raw[f'{layer}_ablated'] = changed_maps[layer].numpy()
            # Pilot occlusion: centered 4x4 patch replaced with neutral gray RGB.
            occluded = inputs.clone()
            occluded[:, :, 12:16, 12:16] = torch.tensor(RGB[3]).float()[None, :, None, None] / 255
            with torch.no_grad():
                masked_logits = model(occluded)
            raw['occluded_input'] = occluded.numpy()
            background_records = []
            for color, name in enumerate(COLOR_NAMES):
                swapped, _ = composite(drawing, torch.tensor([color]))
                with torch.no_grad():
                    swapped_logits = model(swapped)
                background_records.append({"background": name, "rgb": RGB[color], **output_record(swapped_logits, target)})
            filename = f'example-{regime}-{CLASSES[target]}.npz'
            np.savez_compressed(OUTPUT / filename, **raw)
            examples.append({
                "model": regime, "checkpoint_sha256": sha256(model_path), "sample_id": samples[index]['id'],
                "selection": "first manifest test drawing per class; illustrative, not curated",
                "background": COLOR_NAMES[target], "baseline": output_record(baseline, target),
                "ablations": ablations,
                "occlusion": {"type": "controlled_occlusion", "rectangle_xywh": [12, 12, 4, 4], "replacement_rgb": RGB[3], "affected_pixels": 16, "modified": output_record(masked_logits, target)},
                "background_swaps": background_records, "raw_arrays": filename,
                "checks": {"repeat_inference_exact": True, "reset_exact": True, "ablation_injection_exact": True, "saliency_finite_nonzero": bool(torch.isfinite(raw_saliency).all() and raw_saliency.sum() > 0)},
            })
    return examples


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = read_json(args.config)
    experiment = Experiment.from_config(config)
    configure(config['training_seed'], config['threads'])
    output_dir = experiment.results
    output_dir.mkdir(parents=True, exist_ok=True)
    verify_config(read_json(experiment.manifest), config)
    if experiment.phase == 1:
        for regime in ('balanced', 'shortcut'):
            training = read_json(output_dir / f'training-{regime}.json')
            if training['status'] != 'complete' or training['config'] != config:
                raise ValueError('Both recorded training runs must be complete and use this configuration')
            if training['checkpoint_sha256'] != sha256(experiment.checkpoint(regime)):
                raise ValueError('Training log and checkpoint hashes differ')
    drawings, labels, samples = load_split('test', experiment.manifest)
    checks = verify_swap_geometry(drawings)
    results, rows, swaps = [], [], []
    for regime in ('balanced', 'shortcut'):
        result, model_rows, model_swaps = evaluate_model(
            regime, config, drawings, labels, samples, model_path=experiment.checkpoint(regime),
            manifest_path=experiment.manifest, output_dir=output_dir,
        )
        results.append(result)
        rows.extend(model_rows)
        swaps.extend(model_swaps)
        print(f"{regime}: balanced-color accuracy={result['accuracy']['balanced_colors']['mean']:.4f}, any-color flip rate={result['any_color_flip_rate']['mean']:.4f}", flush=True)
    save_csv(output_dir / 'predictions.csv', rows)
    save_csv(output_dir / 'background-swaps.csv', swaps)
    paired = [np.load(output_dir / f'paired-{regime}.npz') for regime in ('balanced', 'shortcut')]
    flip = [(data['logits'][:, :3].argmax(2) != data['logits'][:, :1].argmax(2)).any(1).astype(float) for data in paired]
    paired_difference = estimate(flip[1] - flip[0], config['split_seed'], config['bootstrap_replicates'])
    summary = {
        'schema_version': 2, 'config': config, 'model_version': experiment.version,
        'manifest': str(experiment.manifest.relative_to(ROOT)), 'manifest_sha256': sha256(experiment.manifest),
        'class_order': list(CLASSES), 'background_names': list(COLOR_NAMES), 'background_rgb': RGB,
        'foreground_checks': checks, 'ci_method': f"{config['bootstrap_replicates']} percentile bootstrap replicates resampling source drawings; all matched colors and both models stay paired",
        'color_comparison_scope': 'three training colors; neutral gray reported separately',
        'models': results, 'shortcut_minus_balanced_any_color_flip_rate': paired_difference,
        'environment': {'python': platform.python_version(), 'torch': str(torch.__version__), 'numpy': np.__version__},
    }
    write_json(output_dir / 'summary.json', summary)
    if experiment.phase == 0:
        examples = illustrative_examples(config, drawings, labels, samples)
        write_json(output_dir / 'examples.json', examples)
        from ml.figures import create_figures, create_report
        create_figures(output_dir, summary, examples)
        create_report(output_dir, summary, examples)
    else:
        from ml.phase1_report import create_report
        create_report(experiment, summary)


if __name__ == '__main__':
    main()
