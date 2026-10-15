"""Static figures and a feasibility report generated only from measured files."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/featuredive-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/featuredive-cache')
Path(os.environ['XDG_CACHE_HOME']).mkdir(parents=True, exist_ok=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ml.common import CLASSES, ROOT, read_json


def save(figure, path: Path) -> None:
    figure.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(figure)


def create_figures(output: Path, summary: dict, examples: list[dict]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.5), layout='constrained')
    colors = ('#3567a0', '#c16a28')
    for axis, field, title in zip(axes, ('accuracy', 'any_color_flip_rate', 'target_margin_range'), ('Held-out accuracy', 'Any label flip across 3 colors', 'True-class margin range')):
        for index, model in enumerate(summary['models']):
            metric = model[field]['balanced_colors'] if field == 'accuracy' else model[field]
            low, high = metric['ci95']
            axis.bar(index, metric['mean'], color=colors[index], yerr=np.array([[metric['mean'] - low], [high - metric['mean']]]), capsize=5)
        axis.set_xticks([0, 1], ['Balanced', 'Shortcut'])
        axis.set_title(title)
        axis.set_ylabel('Fraction' if field != 'target_margin_range' else 'Logit units')
        if field != 'target_margin_range':
            axis.set_ylim(0, 1)
    figure.suptitle(f"Phase 0 pilot — {summary['models'][0]['source_drawings']} held-out sources; paired source-bootstrap 95% CIs")
    save(figure, output / 'overview.png')

    figure, axes = plt.subplots(2, 4, figsize=(13, 6), layout='constrained')
    for row, model in enumerate(summary['models']):
        for column, color in enumerate(summary['background_names']):
            axis = axes[row, column]
            matrix = np.array(model['confusion_matrices'][color])
            axis.imshow(matrix, cmap='Blues', vmin=0, vmax=summary['config']['test_per_class'])
            for i in range(3):
                for j in range(3):
                    axis.text(j, i, str(matrix[i, j]), ha='center', va='center', color='white' if matrix[i, j] > summary['config']['test_per_class'] / 2 else 'black')
            axis.set_xticks(range(3), CLASSES, rotation=20)
            axis.set_yticks(range(3), CLASSES)
            axis.set_title(f"{model['regime']} / {color}")
            axis.set_xlabel('Predicted')
            axis.set_ylabel('True')
    save(figure, output / 'confusion-matrices.png')

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for model in summary['models']:
        training = read_json(output / f"training-{model['regime']}.json")
        history = training['history']
        axes[0].plot([row['epoch'] for row in history], [row['train_loss'] for row in history], marker='.', label=model['regime'])
        axes[1].plot([row['epoch'] for row in history], [row['balanced_validation_accuracy'] for row in history], marker='.', label=model['regime'])
    axes[0].set_ylabel('Training cross-entropy')
    axes[1].set_ylabel('Balanced-color validation accuracy')
    for axis in axes:
        axis.set_xlabel('Epoch')
        axis.legend()
    save(figure, output / 'training.png')

    for example in examples:
        arrays = np.load(output / example['raw_arrays'])
        figure, axes = plt.subplots(2, 4, figsize=(12, 6), layout='constrained')
        axes[0, 0].imshow(arrays['source'][0], cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        axes[0, 0].set_title('Source: white stroke alpha')
        axes[0, 1].imshow(arrays['input'][0].transpose(1, 2, 0), interpolation='nearest')
        axes[0, 1].set_title(f"Baseline: {example['baseline']['predicted_class']}")
        axes[0, 2].imshow(arrays['saliency'][0], cmap='magma', interpolation='nearest')
        axes[0, 2].set_title('Sum |RGB logit gradient|\nDisplay scaled per image')
        axes[0, 3].imshow(arrays['occluded_input'][0].transpose(1, 2, 0), interpolation='nearest')
        axes[0, 3].set_title('4×4 neutral-gray occlusion')
        for column, ablation in enumerate(example['ablations']):
            layer, channel = ablation['layer'], ablation['channel']
            activation = arrays[layer][0, channel]
            upper = max(float(activation.max()), 1e-8)
            axes[1, column * 2].imshow(activation, cmap='viridis', vmin=0, vmax=upper, interpolation='nearest')
            axes[1, column * 2].set_title(f'{layer} channel {channel}\nRaw scale: 0–{upper:.3f}')
            axes[1, column * 2 + 1].imshow(arrays[f'{layer}_ablated'][0, channel], cmap='viridis', vmin=0, vmax=upper, interpolation='nearest')
            axes[1, column * 2 + 1].set_title(f"Zeroed; margin Δ {ablation['target_margin_change']:+.3f}")
        for axis in axes.flat:
            axis.set_xticks([])
            axis.set_yticks([])
        figure.suptitle(f"{example['model']} — {example['sample_id']} — true-class margin {example['baseline']['target_logit_margin']:.3f}\nIllustrative first test sample; observations and interventions are distinct")
        save(figure, output / example['raw_arrays'].replace('.npz', '.png'))

    figure, axes = plt.subplots(2, 3, figsize=(12, 6), layout='constrained')
    for axis, example in zip(axes.flat, examples):
        for label, name in enumerate(CLASSES):
            axis.plot(summary['background_names'], [record['probabilities'][label] for record in example['background_swaps']], marker='o', label=name)
        axis.set_ylim(0, 1)
        axis.set_title(f"{example['model']} / {example['sample_id']}")
        axis.set_ylabel('Softmax probability')
        axis.legend(fontsize=8)
    figure.suptitle('Same drawing, changed background — first test sample per class')
    save(figure, output / 'example-background-swaps.png')


def create_report(output: Path, summary: dict, examples: list[dict]) -> None:
    def percent(metric):
        return f"{100 * metric['mean']:.2f}% [{100 * metric['ci95'][0]:.2f}, {100 * metric['ci95'][1]:.2f}]"

    config = summary['config']
    source_count = summary['models'][0]['source_drawings']
    total_count = 3 * sum(config[f'{part}_per_class'] for part in ('train', 'validation', 'test'))
    lines = [
        '# Phase 0 feasibility report', '',
        'This report is generated from actual CPU runs and saved checkpoints. Phase 0 only; Phase 1 and frontend development have not started.', '',
        '## Protocol', '',
        '- Dataset: Google Quick, Draw! 28×28 uint8 bitmaps; cat=0, fish=1, airplane=2.',
        '- Attribution: The Quick, Draw! Dataset, Google, Inc., [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). [Canonical source](https://github.com/googlecreativelab/quickdraw-dataset).',
        f"- Pilot subset: {config['train_per_class']:,} training / {config['validation_per_class']:,} validation / {config['test_per_class']:,} test sources per class ({total_count:,} total). The planned final experiment uses 30,000 sources.",
        f"- Seed {config['split_seed']} selects random rows from the full files before backgrounds are assigned; source indices, bitmap hashes, file hashes, and split labels are frozen in `data/manifests/pilot-v1.json`. Exact bitmap duplicates across selected sources are rejected.",
        '- White foreground alpha = source intensity / 255; RGB = alpha + (1-alpha) × background/255. No resizing, thresholding, or augmentation. Channels-first float32 input in [0,1].',
        f"- Background RGB order: {list(zip(summary['background_names'], summary['background_rgb']))}. Gray is evaluation-only.",
        f"- Balanced training assigns each experimental color with probability 1/3 independently of class. Shortcut training assigns the class-correlated color with probability .9 and each other color with probability .05. Colors are fixed per training source using seed {config['background_seed']}; actual counts are saved in the training JSON files.",
        f"- Both models: exact 26,611-parameter architecture, same seed/initialization ({config['training_seed']}), CPU, Adam lr={config['learning_rate']}, batch size {config['batch_size']}, {config['epochs']} epochs. Select the first epoch with highest validation accuracy over all three colors. Test data is not used for epoch selection.",
        f"- All {source_count:,} test drawings are evaluated with each of three training colors and neutral gray ({source_count * 4:,} forward outputs/model). Source-bootstrap percentile 95% confidence intervals use {config['bootstrap_replicates']:,} paired replicates, seed {config['split_seed']}; these quantify held-out sampling variation, not variation across training seeds.",
        '- Training-like accuracy is an exact weighted expectation over the three color-specific outputs, rather than a newly sampled color assignment. The two swapped mappings rotate every class to a nonpreferred color.', '',
        '## Held-out results', '',
        '| Measure (95% CI) | Balanced | Shortcut |',
        '| --- | ---: | ---: |',
    ]
    balanced, shortcut = summary['models']
    for key, title in [('training_like_expected', 'Training-like expected accuracy'), ('balanced_colors', 'Balanced-color accuracy'), ('preferred_mapping', 'Class-correlated mapping accuracy'), ('swapped_mapping_1', 'Swapped mapping 1 accuracy'), ('swapped_mapping_2', 'Swapped mapping 2 accuracy'), ('gray', 'Neutral-gray accuracy')]:
        lines.append(f"| {title} | {percent(balanced['accuracy'][key])} | {percent(shortcut['accuracy'][key])} |")
    for key, title in [('any_color_flip_rate', 'Any label flip across three colors'), ('pairwise_color_flip_rate', 'Mean pairwise label flip rate')]:
        lines.append(f"| {title} | {percent(balanced[key])} | {percent(shortcut[key])} |")
    def points(metric):
        return f"{100 * metric['mean']:.2f} [{100 * metric['ci95'][0]:.2f}, {100 * metric['ci95'][1]:.2f}]"
    lines.append(f"| Mean true-class probability range (percentage points) | {points(balanced['target_probability_range'])} | {points(shortcut['target_probability_range'])} |")
    lines.append(f"| Mean true-class margin range (logit units) | {balanced['target_margin_range']['mean']:.3f} | {shortcut['target_margin_range']['mean']:.3f} |")
    difference = summary['shortcut_minus_balanced_any_color_flip_rate']
    lines += ['', f"Shortcut minus balanced any-color flip rate: **{100 * difference['mean']:.2f} percentage points**, paired 95% CI [{100 * difference['ci95'][0]:.2f}, {100 * difference['ci95'][1]:.2f}].", '',
        '![Measured overview](../results/phase0/overview.png)', '',
        '![Per-color confusion matrices](../results/phase0/confusion-matrices.png)', '',
        f'Rows are true labels and columns predicted labels in cat/fish/airplane order. Each color matrix contains all {source_count:,} test sources.', '',
        '## Checkpoints and reproducibility', '',
        f"Manifest SHA-256: `{summary['manifest_sha256']}`.", '',
    ]
    for model in summary['models']:
        training = read_json(output / f"training-{model['regime']}.json")
        lines += [f"- `{model['checkpoint']}`: SHA-256 `{model['checkpoint_sha256']}`; selected epoch {model['best_epoch']}; balanced validation accuracy {model['validation_accuracy']:.4f}; training time {training['elapsed_seconds']:.1f}s.", f"  Actual class × color training counts (cat/fish/airplane × green/blue/yellow): `{training['training_background_counts_by_class']}`."]
    lines += ['', f"Environment: `{summary['environment']}`; full platform and optimizer metadata are in training JSON. Dependencies are locked in `uv.lock`. The two compact pilot checkpoints are explicitly included in the repository; future generated checkpoints are ignored by default. The reported checkpoint hashes identify this run; archive hashing can differ after retraining, so compare tensor outputs and metrics as well.", '',
        '```sh', 'uv sync --locked --extra dev', 'uv run --locked python -m ml.data_loader', 'uv run --locked python -m ml.train', 'uv run --locked python -m ml.evaluate', 'uv run --locked pytest -q', '```', '',
        'Evaluation alone loads the saved local checkpoints and verifies their manifest hash. Training/evaluation require no network after dependencies and source arrays are present. Rerunning training overwrites local pilot checkpoints; preserve artifacts before changing an experiment.', '',
        '## Intervention checks and real examples', '',
        f"Counterfactual foreground checks passed for all {summary['foreground_checks']['drawings_checked']} test sources across all four backgrounds. Alpha is bitwise identical; maximum counterfactual blend-formula residual: {summary['foreground_checks']['max_counterfactual_formula_error']:.3g}.", '',
        'Examples are the first manifest test source per class, without searching for a desired outcome. For each model/example, both convolutional layers ablate their highest-mean-activation channel; all other channels at the injection point are unchanged, the intended channel is exactly zero, and resetting restores exact baseline logits. Post-ReLU activations, logits, softmax probabilities, raw gradient saliency, and a neutral-gray central 4×4 occlusion are saved. Saliency is the sum of absolute RGB gradients of the true-class logit, not a causal explanation. All illustrative saliency maps are finite and nonzero.', '',
        '| Model / source | Baseline prediction | Layer/channel | Margin before | Margin after | Margin change |', '| --- | --- | --- | ---: | ---: | ---: |',
    ]
    for example in examples:
        for ablation in example['ablations']:
            lines.append(f"| {example['model']} / {example['sample_id']} | {example['baseline']['predicted_class']} | {ablation['layer']}/{ablation['channel']} | {example['baseline']['target_logit_margin']:.4f} | {ablation['modified']['target_logit_margin']:.4f} | {ablation['target_margin_change']:+.4f} |")
    lines += ['', '![Matched background examples](../results/phase0/example-background-swaps.png)', '']
    for example in examples:
        filename = example['raw_arrays'].replace('.npz', '.png')
        lines += [f"![{example['model']} {example['sample_id']}](../results/phase0/{filename})", '']
    lines += ['## Artifacts', '',
        '- `results/phase0/summary.json`: aggregate metrics, per-color confusion matrices, source-bootstrap CIs, hashes and checks.',
        f'- `predictions.csv`: {source_count * 8:,} inference rows with logits, probabilities, target margins, source IDs and checkpoint hashes.',
        f'- `background-swaps.csv`: {source_count * 12:,} paired comparisons including signed margin/probability changes and label flips (all six color pairs, including gray).',
        '- `examples.json` and `example-*.npz`: actual before/after scores, intervention metadata, source/mask/input arrays, raw activations and raw saliency.',
        '- `paired-*.npz`: complete test logits for independently recomputing paired summaries.',
        '- `training-*.json` and `training.png`: full epoch histories and observed training-color counts.', '',
        '## Feasibility, limitations, and recommendations', '',
    ]
    if balanced['accuracy']['balanced_colors']['mean'] >= .8:
        lines.append('The balanced model exceeds the approximate 80% pilot accuracy target on the held-out balanced-color condition. Retain the provisional cat/fish/airplane categories for the next review.')
    else:
        lines.append('The balanced model falls below the approximate 80% pilot target. Investigate training duration, error cases, and source quality before proposing a category change; the categories have not been changed.')
    if difference['ci95'][0] > 0:
        lines.append('The shortcut model exhibits greater measured background sensitivity in this run. This supports the feasibility of a matched-background educational comparison; it does not establish the same result for every drawing or training seed.')
    else:
        lines.append('The pilot does not clearly establish greater shortcut-model flip sensitivity. Investigate the setup before implementing the Shortcut Trap challenge; do not assume or hardcode its outcome.')
    lines += ['',
        '- This is one training seed, a compact subset, and a three-class task. Repeat training seeds and use the planned final split sizes in a separately authorized phase; these are pilot results, not final research findings.',
        '- Bitmap row indices identify sources only within the hashed files. The bitmap format has no original key_id, artist identity, recognition flag, or country metadata; contributor-level leakage cannot be assessed. Exact bitmap deduplication does not eliminate near duplicates.',
        '- Gray is outside the training color distribution. Anti-aliased edge RGB changes under a background swap as required by alpha compositing; geometry/alpha stay fixed, not every partially transparent pixel RGB value.',
        '- Fixed training colors are sampled and measured rather than perfectly balanced in finite samples. The balanced model is not claimed to be bias-free. Shortcut behavior is measured rather than inferred solely from training labels.',
        '- Channel zeroing establishes an effect within this model for this input, not a semantic detector or a unique real-world causal mechanism. A large activation is not necessarily a large influence; full channel-rank evaluations remain future work.',
        '- Vanilla gradients measure local logit sensitivity; central-patch gray occlusion is an input intervention and may create unusual inputs. This pilot does not implement the full saliency-fidelity study or claim educational learning benefits.',
        '- Softmax probabilities are uncalibrated. Logits and target margins are retained alongside probabilities. Pilot examples overlap the quantitative test set and must not be used to tune final research claims; reserve separate illustrative samples for a final demo.',
        '- Downloaded category files are public but not immutable URLs; recorded SHA-256 values guard replay. Use a local verified copy if upstream bytes change. Source arrays are excluded from Git; compact pilot checkpoints, generated pilot figures, and measured results are included.',
        '- No GPU, cloud resources, frontend, API, or deployment was used. Next steps require review of these results under the implementation plan.', '',
        '**Gate: Phase 0 complete. Stop here for user review; do not begin Phase 1 or frontend implementation.**', '',
    ]
    lines += [
        '## What we learned — student and poster summary', '',
        'We trained two versions of the same small AI model to recognize drawings of cats, fish, and airplanes. Both learned from the same drawings, but their practice backgrounds differed.', '',
        '**Balanced model:** Each drawing category had an equal chance of appearing on a green, blue, or yellow background. Color was assigned independently of the category, so it offered no reliable clue to the correct answer. “Balanced” describes the background assignments; it does not mean the model is free from all bias.', '',
        '**Shortcut model:** During training, cats appeared on green, fish on blue, and airplanes on yellow 90% of the time. This made color an easy clue, even though background color does not define the object. “Shortcut” describes this training setup; the experiments test whether the model actually relies on the clue.', '',
        f"**The test:** We showed both models {source_count:,} drawings they had not trained on. We kept each drawing unchanged and tried all three background colors.", '',
        f"- With the colors equally represented, the balanced model identified **{100 * balanced['accuracy']['balanced_colors']['mean']:.1f}%** of inputs correctly, compared with **{100 * shortcut['accuracy']['balanced_colors']['mean']:.1f}%** for the shortcut model.",
        f"- Changing only the background changed the predicted label at least once for **{100 * balanced['any_color_flip_rate']['mean']:.1f}%** of drawings with the balanced model and **{100 * shortcut['any_color_flip_rate']['mean']:.1f}%** with the shortcut model.",
        f"- When test backgrounds followed the shortcut model’s training pattern, its accuracy was **{100 * shortcut['accuracy']['training_like_expected']['mean']:.1f}%**. A high score under familiar conditions can therefore hide dependence on an accidental clue.", '',
        '**Inside the model:** We also turned off individual internal channels and measured the changes in prediction scores. These experiments show how a channel affects a decision for a particular input; they do not establish that it detects a meaningful object part, such as a cat’s ear.', '',
        '**Takeaway:** High accuracy alone does not explain how an AI reaches its answers. Changing one thing at a time helps reveal which clues influence its decisions.', '',
        'This small, single-training-run pilot supports testing background reliance in FeatureDive. It does not prove that either model understands objects or that the balanced model is fair in every setting.', '',
    ]
    (ROOT / 'docs/phase0-feasibility.md').write_text('\n'.join(lines))
