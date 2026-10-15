"""Generate Phase 1 tables, figures, dataset card, and versioned model cards."""
from __future__ import annotations

import csv
import hashlib

import numpy as np

from ml.backgrounds import COLOR_NAMES, RGB, assign_colors
from ml.common import CLASSES, ROOT, read_json, sha256, write_json
from ml.experiment import Experiment
from ml.figures import plt, save
from ml.train import load_checkpoint


def percent(metric: dict) -> str:
    low, high = metric['ci95']
    return f"{metric['mean'] * 100:.2f}% [{low * 100:.2f}, {high * 100:.2f}]"


def create_report(experiment: Experiment, summary: dict) -> None:
    config = summary['config']
    output = experiment.results
    manifest = read_json(experiment.manifest)
    training = [read_json(output / f'training-{regime}.json') for regime in ('balanced', 'shortcut')]
    if training[0]['initial_weights_sha256'] != training[1]['initial_weights_sha256']:
        raise ValueError('Matched models did not start from identical weights')
    models = summary['models']
    labels = np.array([row['label'] for row in manifest['samples'] if row['split'] == 'train'])
    train_samples = [row for row in manifest['samples'] if row['split'] == 'train']
    with (output / 'training-backgrounds.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', 'sample_id', 'target_class', 'background', 'red', 'green', 'blue'])
        for run in training:
            colors = assign_colors(labels, run['regime'], config['background_seed'])
            if hashlib.sha256(colors.tobytes()).hexdigest() != run['training_colors_sha256']:
                raise ValueError('Training background assignments do not replay')
            for sample, color in zip(train_samples, colors):
                writer.writerow([run['regime'], sample['id'], sample['class'], COLOR_NAMES[color], *RGB[color]])
    conditions = ['training_like_expected', 'balanced_colors', 'preferred_mapping',
                  'swapped_mapping_1', 'swapped_mapping_2', *COLOR_NAMES]
    with (output / 'evaluation-table.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', 'condition', 'accuracy', 'ci95_low', 'ci95_high', 'source_drawings', 'checkpoint_sha256'])
        for model in models:
            for condition in conditions:
                value = model['accuracy'][condition]
                writer.writerow([model['regime'], condition, value['mean'], *value['ci95'], model['source_drawings'], model['checkpoint_sha256']])

    figure, axes = plt.subplots(1, 3, figsize=(13, 3.5), layout='constrained')
    for axis, field, title in zip(axes, ['accuracy', 'any_color_flip_rate', 'target_margin_range'],
                                  ['Accuracy across three colors', 'Any label flip across three colors', 'True-class margin range']):
        for index, model in enumerate(models):
            value = model[field]['balanced_colors'] if field == 'accuracy' else model[field]
            low, high = value['ci95']
            axis.bar(index, value['mean'], color=['#3567a0', '#c16a28'][index],
                     yerr=np.array([[value['mean'] - low], [high - value['mean']]]), capsize=5)
        axis.set_xticks([0, 1], ['Balanced', 'Shortcut'])
        axis.set_title(title)
        axis.set_ylabel('Fraction' if field != 'target_margin_range' else 'Logit units')
        if field != 'target_margin_range':
            axis.set_ylim(0, 1)
    figure.suptitle(f"Phase 1 — {models[0]['source_drawings']:,} independent test sources; paired source-bootstrap 95% CIs")
    save(figure, output / 'overview.png')

    figure, axes = plt.subplots(2, 4, figsize=(13, 6), layout='constrained')
    for row, model in enumerate(models):
        for column, name in enumerate(COLOR_NAMES):
            axis = axes[row, column]
            matrix = np.array(model['confusion_matrices'][name])
            upper = config['test_per_class']
            axis.imshow(matrix, cmap='Blues', vmin=0, vmax=upper)
            for i in range(3):
                for j in range(3):
                    axis.text(j, i, str(matrix[i, j]), ha='center', va='center', color='white' if matrix[i, j] > upper / 2 else 'black')
            axis.set_xticks(range(3), CLASSES, rotation=20)
            axis.set_yticks(range(3), CLASSES)
            axis.set_title(f"{model['regime']} / {name}")
            axis.set_xlabel('Predicted')
            axis.set_ylabel('True')
    save(figure, output / 'confusion-matrices.png')

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for run in training:
        history = run['history']
        axes[0].plot([row['epoch'] for row in history], [row['train_loss'] for row in history], marker='.', label=run['regime'])
        axes[1].plot([row['epoch'] for row in history], [row['balanced_validation_accuracy'] for row in history], marker='.', label=run['regime'])
    axes[0].set_ylabel('Training cross-entropy')
    axes[1].set_ylabel('Balanced-color validation accuracy')
    for axis in axes:
        axis.set_xlabel('Epoch')
        axis.legend()
    save(figure, output / 'training.png')

    registry = {'schema_version': 1, 'model_version': experiment.version,
                'manifest': str(experiment.manifest.relative_to(ROOT)), 'manifest_sha256': summary['manifest_sha256'],
                'class_order': list(CLASSES), 'background_names': list(COLOR_NAMES), 'background_rgb': RGB,
                'input': {'shape_chw': [3, 28, 28], 'dtype': 'float32', 'range': [0, 1], 'foreground': 'white', 'alpha': 'source intensity / 255'},
                'models': []}
    for model, run in zip(models, training):
        _, saved = load_checkpoint(experiment.checkpoint(model['regime']), experiment.manifest)
        registry['models'].append({'id': model['regime'], 'checkpoint': model['checkpoint'],
                                  'checkpoint_sha256': model['checkpoint_sha256'], 'weights_sha256': saved['weights_sha256'],
                                  'parameter_count': run['parameter_count'], 'training_device': run['environment']['device'],
                                  'inference_device': 'cpu', 'selected_epoch': model['best_epoch']})
    write_json(experiment.checkpoint_directory / 'registry.json', registry)

    lines = [f'# Phase 1 — {experiment.version}', '',
             'Phase 1 is complete: a frozen full-size dataset, matched trained checkpoints, held-out CPU measurements, and model documentation. Phase 2 has not started.', '',
             '## Dataset and training', '',
             f"- Source partitions: {config['train_per_class']:,} training / {config['validation_per_class']:,} validation / {config['test_per_class']:,} test drawings per class; 30,000 total. Fixed labels: cat=0, fish=1, airplane=2.",
             '- Every pilot source and exact bitmap copy is excluded from all Phase 1 partitions. The new validation and test drawings have not been used in the pilot experiments or reviewed examples.',
             f"- Split seed {config['split_seed']}; training seed {config['training_seed']}; color assignment seed {config['background_seed']}. Both models started with identical weights (verified by tensor-content hash).",
             f"- Same 26,611-parameter CNN, Adam lr={config['learning_rate']}, batch size {config['batch_size']}, {config['epochs']} epochs. No augmentation, learning-rate schedule, mixed precision, or architecture change.",
             '- Balanced colors: 1/3 each independently of class. Shortcut: cat/green, fish/blue, airplane/yellow with probability .90; each other color .05. Colors are sampled once per source; observed frequencies and per-source assignments are retained.',
             '- Compositing is unchanged from Phase 0: white alpha=intensity/255 over a fixed RGB background, float32 NCHW in [0,1]. Neutral gray is evaluation-only. Partially transparent edge RGB changes when the background changes; alpha and geometry do not.',
             '- Select the first epoch achieving the highest validation accuracy averaged over all three training colors. The test set is used only after both runs finish.',
             f"- Actual training devices: {', '.join(run['regime'] + '=' + run['environment']['device'] for run in training)}. Checkpoint tensors are stored on CPU; final evaluation runs on CPU.",
             '- Python, NumPy, PyTorch, and MPS seeds are set and deterministic algorithms requested. This is one seed per regime. Bitwise training equality across devices, operating systems, or PyTorch versions is not promised; CPU inference repeatability is checked separately. See [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/2.14/notes/randomness.html).', '',
             '## Measured held-out results', '',
             f"All {models[0]['source_drawings']:,} test sources are evaluated on green, blue, yellow, and gray: {len(COLOR_NAMES) * models[0]['source_drawings']:,} inputs per model. Confidence intervals resample source drawings and retain all colors and both model versions together ({config['bootstrap_replicates']:,} percentile bootstrap replicates). They quantify test-sample uncertainty, not training-seed uncertainty.", '',
             '| Measure (95% CI) | Balanced | Shortcut |', '| --- | ---: | ---: |']
    names = ['Training-like expected accuracy', 'Balanced-color accuracy', 'Class-correlated mapping accuracy',
             'Swapped mapping 1 accuracy', 'Swapped mapping 2 accuracy', 'Green accuracy', 'Blue accuracy', 'Yellow accuracy', 'Neutral-gray accuracy']
    for condition, name in zip(conditions, names):
        lines.append(f"| {name} | {percent(models[0]['accuracy'][condition])} | {percent(models[1]['accuracy'][condition])} |")
    lines.append(f"| Sources with any label flip across 3 colors | {percent(models[0]['any_color_flip_rate'])} | {percent(models[1]['any_color_flip_rate'])} |")
    difference = summary['shortcut_minus_balanced_any_color_flip_rate']
    lines += ['', f"Shortcut minus balanced flip rate: {difference['mean'] * 100:.2f} percentage points; paired 95% CI [{difference['ci95'][0] * 100:.2f}, {difference['ci95'][1] * 100:.2f}].", '',
              'Training-like accuracy is an exact expectation over the three color-specific outputs using each regime’s probabilities. Swapped mappings rotate the class-correlated colors by one or two positions. Background flip rate means at least one label changes across the three training colors; gray is excluded from this rate.', '',
              '![Measured Phase 1 overview](../results/phase1/overview.png)', '',
              '![Per-color confusion matrices](../results/phase1/confusion-matrices.png)', '',
              '![Training curves](../results/phase1/training.png)', '',
              '## Phase 1 gate checks', '',
              f"- Source files match recorded SHA-256 values; all 30,000 selected source IDs and bitmap hashes are unique. Excluded pilot manifest SHA-256: `{manifest['excluded_manifests'][0]['sha256']}`.",
              f"- Foreground geometry checked on all {summary['foreground_checks']['drawings_checked']:,} test sources across four colors. Alpha is exactly unchanged; maximum blend-formula residual {summary['foreground_checks']['max_counterfactual_formula_error']:.3g}.",
              '- Both models load through `weights_only=True` with CPU mapping, match their manifest and tensor-content hashes, and run in `eval()` mode. Repeated inference produced exactly equal logits for every held-out source/color input. All logits are finite.',
              '- Both model histories are complete. Checkpoint file hashes match their logs; class order, palette, initial weight hash, dependency-lock hash, and source-code hashes are recorded.', '',
              '## Checkpoints and files', '', f"Manifest SHA-256: `{summary['manifest_sha256']}`.", '']
    for model, run, registered in zip(models, training, registry['models']):
        lines += [f"- `{model['checkpoint']}`: selected epoch {model['best_epoch']}; balanced validation accuracy {model['validation_accuracy']:.4f}; {run['elapsed_seconds']:.1f}s training on {run['environment']['device']}.",
                  f"  Checkpoint SHA-256: `{model['checkpoint_sha256']}`; tensor-content SHA-256: `{registered['weights_sha256']}`.",
                  f"  Observed class × color counts (cat/fish/airplane × green/blue/yellow): `{run['training_background_counts_by_class']}`."]
    lines += ['', '- `results/phase1/evaluation-table.csv`: condition accuracy with confidence intervals.',
              '- `results/phase1/predictions.csv` and `background-swaps.csv`: full inference and signed paired intervention rows, with source IDs and checkpoint hashes.',
              '- `results/phase1/paired-*.npz`: complete logits, labels, and source IDs for independent recalculation.',
              '- `results/phase1/training-*.json`: full histories, observed counts, config, environment, code and dependency hashes.',
              '- `results/phase1/training-backgrounds.csv`: replay-verified color assignments for every source/model pair.',
              '- `checkpoints/phase1-v1/registry.json`: compact model/input contract and artifact hashes.',
              '- [Dataset card](data-card.md), [balanced model card](model-cards/phase1-v1-balanced.md), [shortcut model card](model-cards/phase1-v1-shortcut.md).', '',
              '## Reproduce or reevaluate', '', '```sh', 'uv sync --locked --extra dev',
              'uv run --locked python -m ml.data_loader --config configs/phase1.json',
              '# Checkpoints are included: this evaluates them on CPU without retraining.',
              'uv run --locked python -m ml.evaluate --config configs/phase1.json',
              '# To deliberately replace the recorded checkpoints with a new training run:',
              'uv run --locked python -m ml.train --config configs/phase1.json --device mps --overwrite',
              'uv run --locked python -m ml.evaluate --config configs/phase1.json',
              'uv run --locked pytest -q', '```', '',
              'The configured `auto` device prefers MPS and uses CPU if MPS is unavailable. Explicit `--device mps` fails if MPS is unavailable, preventing silent CPU fallback. A sandbox can hide the accelerator even when it is available on the host. [PyTorch MPS documentation](https://docs.pytorch.org/docs/2.14/notes/mps.html).', '',
              'Existing training runs are protected unless `--overwrite` is passed. For a different scientific experiment, copy the config and choose new versioned artifact paths. Offline replay requires the three source arrays, whose checksums are enforced.', '',
              '## Interpretation and limits', '',
              '- These are measured results from one pair of models. The balanced regime describes the color distribution, not universal fairness or absence of bias. The shortcut regime creates a potentially useful cue; reliance is measured from actual responses.',
              '- Categories, small image size, exact-duplicate filtering, and available bitmap metadata limit generalization. Contributor identities and near-duplicate relationships are unknown. Gray is outside training support.',
              '- Probabilities are uncalibrated. Background-swap effects are within-model input interventions and do not identify a unique internal semantic mechanism.',
              '- No human learning benefit is claimed. Full saliency fidelity and activation-influence studies remain Phase 5 work; this phase validates the dataset and checkpoints.', '',
              '**Student takeaway:** Training examples influence which clues a model uses. We kept drawings fixed and changed their backgrounds to measure how much each model’s answers depend on color. A high accuracy score alone does not establish that a model will stay reliable when those clues change.', '',
              '**Gate: Phase 1 complete. Stop for review before Phase 2.**', '']
    (ROOT / 'docs/phase1-training.md').write_text('\n'.join(lines))
    for model, run, registered in zip(models, training, registry['models']):
        regime = model['regime']
        description = ('Background color was assigned independently of the drawing category, with probability 1/3 for each training color.' if regime == 'balanced' else
                       'Training color matched category with probability .90 (cat/green, fish/blue, airplane/yellow), and each alternative color with probability .05.')
        card = [f'# Model card — {experiment.version} / {regime}', '',
                '## Purpose and scope', '',
                'Educational three-class drawing classification and controlled attribution/intervention experiments. Not intended for consequential decisions, general image recognition, or claims of human-like understanding.', '',
                f'**Training regime:** {description}', '',
                'The model name describes its training data. “Balanced” is not a fairness guarantee; “shortcut” is not a hardcoded prediction rule. Both are the same learned CNN architecture.', '',
                '## Model and input contract', '',
                '- PyTorch CNN, 26,611 trainable parameters; Conv(3→8), ReLU, pool; Conv(8→16), ReLU, pool; Linear(784→32), ReLU; Linear(32→3).',
                '- Input: float32 NCHW [N,3,28,28] in [0,1]; white strokes with source intensity/255 alpha over the fixed RGB palette. No additional standardization.',
                f'- Palette: `{list(zip(COLOR_NAMES, RGB))}`. Gray is evaluation-only.',
                '- Output: raw logits in cat=0, fish=1, airplane=2 order. Softmax for display gives uncalibrated probabilities; target logit margin is the target minus the largest other logit.',
                '- Post-ReLU activation taps and single convolutional-channel zeroing are available in the shared CNN. No channel is assigned an assumed semantic label.', '',
                '## Data, training, and provenance', '',
                '- [Dataset card](../data-card.md); Quick, Draw! by Google, Inc., CC BY 4.0. Source metadata and exact indices are recorded in the manifest.',
                f"- 24,000 training sources, 3,000 validation sources, 3,000 test sources. All pilot sources/bitmap copies excluded. Manifest: `{summary['manifest']}`; SHA-256 `{summary['manifest_sha256']}`.",
                f"- Config: `configs/phase1.json`; seeds split={config['split_seed']}, training={config['training_seed']}, background={config['background_seed']}.",
                f"- Adam lr={config['learning_rate']}, batch={config['batch_size']}, budget={config['epochs']} epochs. Actual device `{run['environment']['device']}`; selected epoch {model['best_epoch']} by balanced-color validation accuracy {model['validation_accuracy']:.4f}.",
                f"- Environment: `{run['environment']}`. Full history, color counts, initial-weight hash, code hashes, and lock hash: `results/phase1/training-{regime}.json`.", '',
                '## Measured performance', '',
                f"- Balanced-color held-out accuracy (95% source-bootstrap CI): {percent(model['accuracy']['balanced_colors'])}.",
                f"- Training-like expected accuracy: {percent(model['accuracy']['training_like_expected'])}.",
                f"- Neutral-gray accuracy: {percent(model['accuracy']['gray'])}.",
                f"- Sources with any label flip across three colors: {percent(model['any_color_flip_rate'])}.",
                '- [Full results and confusion matrices](../phase1-training.md). Test set contains 3,000 independent sources with four matched backgrounds each; uncertainty does not cover retraining seeds.', '',
                '## Loading and verification', '',
                f"- Checkpoint: `{model['checkpoint']}`; SHA-256 `{model['checkpoint_sha256']}`.",
                f"- Tensor-content SHA-256: `{registered['weights_sha256']}`; version `{experiment.version}`.",
                '- `ml.train.load_checkpoint(path, manifest_path)` safely maps tensors to CPU, verifies manifest/class/weight hashes, and returns a model in `eval()` mode. Repeated CPU inference is exactly equal for all held-out color variants.',
                '- Use the registered preprocessing and label order. Set the model to eval mode; training-time MPS availability is not required for serving.', '',
                '## Limitations', '',
                '- Single training seed, three drawing categories, low resolution, unknown contributor identity, and possible near duplicates. Not assessed on arbitrary uploaded images.',
                '- Synthetic background assignments are a controlled test of one potential spurious cue. They do not represent all social bias or real-world context changes.',
                '- Uncalibrated probabilities and observed activation maps must not be presented as confidence guarantees or causal explanations. Ablations show model-specific effects, not unique semantic circuits.',
                '- Training is seeded with deterministic algorithms requested, but bitwise equality across hardware or software versions is not guaranteed. Saved-checkpoint CPU repeatability was measured.', '',
                '**Status:** Phase 1 artifact; API, public demo, and full research evaluations are not yet implemented.', '']
        (ROOT / f'docs/model-cards/{experiment.version}-{regime}.md').write_text('\n'.join(card))
