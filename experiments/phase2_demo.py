"""Generate real challenge demonstrations through the shared Phase 2 engine."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np

from backend.engine import InterpretabilityEngine
from backend.schemas import ActivationRequest, ExperimentRequest
from ml.common import CLASSES, ROOT, sha256, write_json
from ml.figures import plt, save

OUTPUT = ROOT / 'results/phase2'


def base_request(sample: dict, model='balanced') -> dict:
    return {'model_id': model, 'sample_id': sample['id'],
            'background': ('green', 'blue', 'yellow')[sample['label']], 'target_class': sample['class']}


def compact(result) -> dict:
    """Preserve provenance, request, scores, and logs without duplicate images."""
    return result.model_dump(mode='json', exclude={'input'})


def csv_rows(name: str, rows: list[dict]) -> None:
    with (OUTPUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def patch_scan(engine: InterpretabilityEngine) -> dict:
    rows, candidates = [], []
    for sample in engine.samples.values():
        base = base_request(sample)
        saliency = engine.saliency(ExperimentRequest(**base))
        scores = np.array(saliency.modified_scores)
        local = []
        for y in range(0, 28, 4):
            for x in range(0, 28, 4):
                operation = {'type': 'controlled_occlusion', 'region': {'rectangle': {'x': x, 'y': y, 'width': 4, 'height': 4}}, 'replacement_rgb': [128, 128, 128]}
                result = engine.infer(ExperimentRequest(**base, interventions=[operation]))
                local.append({
                    'sample_id': sample['id'], 'model': 'balanced', 'target_class': sample['class'],
                    'background': base['background'], 'baseline_prediction': result.baseline.predicted_class,
                    'x': x, 'y': y, 'width': 4, 'height': 4, 'replacement_rgb': '128,128,128',
                    'saliency_sum': float(scores[y:y + 4, x:x + 4].sum()),
                    'margin_change': result.comparison.target_logit_margin_change,
                    'absolute_margin_change': abs(result.comparison.target_logit_margin_change),
                    'probability_point_change': result.comparison.target_probability_point_change,
                    'classification_flip': result.comparison.classification_flip,
                    'checkpoint_sha256': result.checkpoint_sha256,
                })
        attributions = np.array([row['saliency_sum'] for row in local])
        effects = np.array([row['absolute_margin_change'] for row in local])
        for row in local:
            # Competition ranks avoid assigning arbitrary order to exact ties.
            row['saliency_rank'] = int(1 + (attributions > row['saliency_sum']).sum())
            row['influence_rank'] = int(1 + (effects > row['absolute_margin_change']).sum())
        highest_saliency = local[int(attributions.argmax())]
        largest_effect = local[int(effects.argmax())]
        candidates.append({'sample': sample, 'highest_saliency': highest_saliency, 'largest_effect': largest_effect})
        rows.extend(local)
    csv_rows('saliency-patches.csv', rows)
    eligible = [row for row in candidates if row['highest_saliency']['baseline_prediction'] == row['sample']['class']]
    selected = max(eligible or candidates, key=lambda row: row['highest_saliency']['influence_rank'])
    base = base_request(selected['sample'])
    selected['selection'] = 'Among the 24 catalog sources with correct balanced baseline prediction, maximize influence rank of the highest-saliency patch; first manifest source breaks ties. Curated illustration, not a held-out finding.'
    selected['saliency'] = engine.saliency(ExperimentRequest(**base)).model_dump(mode='json')
    selected['interventions'] = []
    for patch in (selected['highest_saliency'], selected['largest_effect']):
        operation = {'type': 'controlled_occlusion', 'region': {'rectangle': {key: patch[key] for key in ('x', 'y', 'width', 'height')}}, 'replacement_rgb': [128, 128, 128]}
        selected['interventions'].append(engine.infer(ExperimentRequest(**base, interventions=[operation])).model_dump(mode='json'))
    return selected


def channel_scan(engine: InterpretabilityEngine) -> dict:
    rows, candidates = [], []
    for sample in engine.samples.values():
        base = base_request(sample)
        for layer in ('conv1', 'conv2'):
            baseline = engine.activations(ActivationRequest(**base, layer=layer))
            local = []
            for channel, activation in enumerate(baseline.baseline_mean_by_channel):
                result = engine.infer(ExperimentRequest(**base, interventions=[{'type': 'channel_ablation', 'layer': layer, 'channel': channel}]))
                local.append({
                    'sample_id': sample['id'], 'model': 'balanced', 'layer': layer, 'channel': channel,
                    'target_class': sample['class'], 'background': base['background'], 'baseline_prediction': result.baseline.predicted_class,
                    'mean_activation': activation, 'margin_change': result.comparison.target_logit_margin_change,
                    'absolute_margin_change': abs(result.comparison.target_logit_margin_change),
                    'probability_point_change': result.comparison.target_probability_point_change,
                    'classification_flip': result.comparison.classification_flip, 'checkpoint_sha256': result.checkpoint_sha256,
                })
            activities = np.array([row['mean_activation'] for row in local])
            effects = np.array([row['absolute_margin_change'] for row in local])
            for row in local:
                row['activation_rank'] = int(1 + (activities > row['mean_activation']).sum())
                row['influence_rank'] = int(1 + (effects > row['absolute_margin_change']).sum())
            candidates.append({'sample': sample, 'layer': layer, 'highest_activation': local[int(activities.argmax())], 'largest_effect': local[int(effects.argmax())]})
            rows.extend(local)
    csv_rows('channel-ablations.csv', rows)
    eligible = [row for row in candidates if row['highest_activation']['baseline_prediction'] == row['sample']['class']]
    selected = max(eligible or candidates, key=lambda row: row['highest_activation']['influence_rank'])
    selected['selection'] = 'Among all 24 catalog sources × both convolutional layers with correct baseline predictions, maximize influence rank of the most active channel; first source/layer breaks ties. All channels are ablated independently from baseline.'
    base = base_request(selected['sample'])
    selected['interventions'] = []
    for record in (selected['highest_activation'], selected['largest_effect']):
        result = engine.activations(ActivationRequest(**base, layer=selected['layer'], interventions=[{'type': 'channel_ablation', 'layer': selected['layer'], 'channel': record['channel']}]))
        selected['interventions'].append(result.model_dump(mode='json'))
    return selected


def background_scan(engine: InterpretabilityEngine) -> dict:
    rows, candidates = [], []
    for sample in engine.samples.values():
        records = {}
        for model in ('balanced', 'shortcut'):
            base = base_request(sample, model)
            results = []
            for background in ('green', 'blue', 'yellow', 'gray'):
                result = engine.infer(ExperimentRequest(**base, interventions=[{'type': 'background_swap', 'background': background}]))
                results.append(compact(result))
                rows.append({'sample_id': sample['id'], 'model': model, 'background': background,
                             'target_class': sample['class'], 'predicted_class': result.modified.predicted_class,
                             'target_probability': result.modified.probabilities[sample['label']],
                             'margin': result.modified.target_logit_margin, 'margin_change': result.comparison.target_logit_margin_change,
                             'probability_point_change': result.comparison.target_probability_point_change,
                             'classification_flip': result.comparison.classification_flip, 'checkpoint_sha256': result.checkpoint_sha256})
            records[model] = results
        ranges = {model: np.ptp([row['modified']['probabilities'][sample['label']] for row in records[model][:3]]) for model in records}
        candidates.append({'sample': sample, 'records': records, 'probability_ranges': {key: float(value) for key, value in ranges.items()}, 'range_difference': float(ranges['shortcut'] - ranges['balanced'])})
    csv_rows('background-swaps.csv', rows)
    eligible = [row for row in candidates if all(row['records'][model][0]['baseline']['predicted_class'] == row['sample']['class'] for model in ('balanced','shortcut'))]
    selected = max(eligible or candidates, key=lambda row: row['range_difference'])
    selected['selection'] = 'Among catalog sources correctly classified by both models on the class-correlated background, maximize shortcut minus balanced true-class probability range across the three training colors; first source breaks ties. Gray is shown separately.'
    selected['input'] = engine.infer(ExperimentRequest(**base_request(selected['sample']))).input
    return selected


def figures(patches, channels, backgrounds):
    figure, axes = plt.subplots(1, 4, figsize=(13, 3.2), layout='constrained')
    inputs = patches['saliency']['experiment']['input']
    axes[0].imshow(np.array(inputs['baseline_rgb_chw']).transpose(1,2,0), interpolation='nearest')
    axes[0].set_title(f"Source: {patches['sample']['id']}")
    axes[1].imshow(patches['saliency']['modified_scores'], cmap='magma', interpolation='nearest')
    axes[1].set_title('Local logit sensitivity\nPer-image contrast; raw scores saved')
    for axis, result, title in zip(axes[2:], patches['interventions'], ['Highest saliency patch', 'Largest margin effect patch']):
        axis.imshow(np.array(result['input']['modified_rgb_chw']).transpose(1,2,0), interpolation='nearest')
        axis.set_title(f"{title}\nMargin Δ {result['comparison']['target_logit_margin_change']:+.3f}")
    for axis in axes:
        axis.set_xticks([]); axis.set_yticks([])
    save(figure, OUTPUT / 'find-the-evidence.png')

    figure, axes = plt.subplots(1, 4, figsize=(12, 3.2), layout='constrained')
    for index, (record, result) in enumerate(zip([channels['highest_activation'],channels['largest_effect']], channels['interventions'])):
        channel = record['channel']
        original = np.array(result['baseline_values'])[channel]
        changed = np.array(result['modified_values'])[channel]
        upper = max(float(original.max()), 1e-8)
        for axis, image in zip(axes[index*2:index*2+2], [original,changed]):
            axis.imshow(image, cmap='viridis', vmin=0, vmax=upper, interpolation='nearest')
            axis.set_xticks([]); axis.set_yticks([])
        axes[index*2].set_title(f"Channel {channel}; scale 0–{upper:.3f}\nMean activity {record['mean_activation']:.3f}")
        axes[index*2+1].set_title(f"Zeroed\nMargin Δ {record['margin_change']:+.3f}")
    figure.suptitle(f"{channels['sample']['id']} / {channels['layer']}: raw activation range shared within each before/after pair")
    save(figure, OUTPUT / 'hidden-layers.png')

    figure, axes = plt.subplots(1, 3, figsize=(13, 3.5), layout='constrained')
    axes[0].imshow(np.array(backgrounds['input']['baseline_rgb_chw']).transpose(1,2,0), interpolation='nearest')
    axes[0].set_title(f"Same source: {backgrounds['sample']['id']}")
    axes[0].set_xticks([]); axes[0].set_yticks([])
    for axis, model in zip(axes[1:], ['balanced','shortcut']):
        for label, name in enumerate(CLASSES):
            axis.plot(['green','blue','yellow','gray'], [row['modified']['probabilities'][label] for row in backgrounds['records'][model]],marker='o',label=name)
        axis.set_title(model)
        axis.set_ylim(0,1); axis.set_ylabel('Uncalibrated softmax probability'); axis.legend()
    save(figure, OUTPUT / 'shortcut-trap.png')


def report(engine, patches, channels, backgrounds, verification):
    top, effect = patches['highest_saliency'], patches['largest_effect']
    active, influential = channels['highest_activation'], channels['largest_effect']
    lines = ['# Phase 2 — Interpretability engine and challenge demonstrations', '',
             'Phase 2 provides a stateless CPU interpretability engine and validated FastAPI endpoints. This report runs the same engine used by the API against real Phase 1 checkpoints. No new training or frontend work was performed.', '',
             '## Example selection and scientific boundaries', '',
             'The demo catalog contains the first eight **training sources per class** from the frozen Phase 1 manifest (24 total). These familiar examples are used for illustration, not independent performance evaluation. Validation and test sources are untouched. The three examples below are curated using declared criteria after scanning this catalog; all scanned rows are saved, including small and null effects. Do not generalize their selected effects to the full dataset.', '',
             'Activation is observed activity. Saliency is local sensitivity of a selected class logit to input changes. Occlusion, background replacement, and channel zeroing are interventions with measured model effects. An intervention does not prove a unique internal semantic mechanism.', '',
             '## Challenge 1 — Find the Evidence', '',
             '**Student question:** Does the brightest saliency region necessarily cause the largest prediction change when covered?', '',
             f"Source `{patches['sample']['id']}`; balanced model; target `{patches['sample']['class']}`. The 28×28 image is divided into 49 nonoverlapping 4×4 patches. Attribution is the sum of absolute RGB gradients of the selected class logit, summed within each patch. Each patch is independently replaced with fixed neutral gray RGB (128,128,128), and the model is rerun from baseline.", '',
             f"Selection rule: {patches['selection']}", '',
             '| Patch | Coordinates (x,y) | Saliency sum | Rank by saliency | Margin change | Rank by absolute margin effect |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for name,row in [('Highest saliency',top),('Largest margin effect',effect)]:
        lines.append(f"| {name} | ({row['x']},{row['y']}) | {row['saliency_sum']:.4f} | {row['saliency_rank']} | {row['margin_change']:+.4f} | {row['influence_rank']} |")
    lines += ['', f"The most salient patch ranks **{top['influence_rank']} of 49** by absolute margin change. The largest-effect patch ranks **{effect['saliency_rank']} of 49** by saliency. Ranks use competition ranking for equal scores. Margin means the target-class score minus the largest competing score; negative change moves the decision away from the target.", '',
              '![Saliency and independent occlusions](../results/phase2/find-the-evidence.png)', '',
              '**Observe → hypothesize → intervene → reflect:** Inspect the saliency map, record which patch you predict will matter most, cover it, then compare with the independently tested alternatives. A bright gradient measures sensitivity to a tiny local change; covering an entire patch is a larger change, can alter competing scores, and can create an unusual input. Agreement or disagreement must be measured.', '',
              '**Learning objective:** Distinguish attribution from the effect of an input intervention, and state the replacement policy when interpreting the result.', '',
              '## Challenge 2 — Inside the Hidden Layers', '',
              '**Student question:** Is the most active channel always the most influential for this prediction?', '',
              f"Source `{channels['sample']['id']}`; balanced model; target `{channels['sample']['class']}`; layer `{channels['layer']}`. We measure mean post-ReLU activity, then zero each entire channel independently **after ReLU and before pooling**, recomputing all downstream layers.", '',
              f"Selection rule: {channels['selection']}", '',
              '| Channel selection | Index | Mean activation | Activity rank | Margin change | Rank by absolute margin effect |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name,row in [('Highest activity',active),('Largest margin effect',influential)]:
        lines.append(f"| {name} | {row['channel']} | {row['mean_activation']:.4f} | {row['activation_rank']} | {row['margin_change']:+.4f} | {row['influence_rank']} |")
    lines += ['', '![Measured channel activation and ablation](../results/phase2/hidden-layers.png)', '',
              '**Observe → hypothesize → intervene → reflect:** Choose a channel based on its activity, predict the direction of the score change, and zero it. Compare the measured effect with other channels. Explain why downstream weights and interactions can make a less active channel influential. Strong activity alone does not justify calling a channel an “ear detector.”', '',
              '**Learning objective:** Separate a channel’s response magnitude from its measured influence on a chosen class decision. A positive margin change after removing a channel is possible; influence has a direction.', '',
              '## Challenge 3 — The Shortcut Trap', '',
              '**Student question:** Which model changes its answer more when the drawing stays the same but the background changes?', '',
              '**Balanced model:** Color was independently assigned, with each of the three training colors having probability 1/3 for every class. This describes the color distribution, not universal fairness.', '',
              '**Shortcut model:** Cat/green, fish/blue, and airplane/yellow occurred with probability .90 in training, with probability .05 for each other color. This creates a potentially useful accidental clue; the experiment measures whether it affects predictions.', '',
              f"Source `{backgrounds['sample']['id']}`; target `{backgrounds['sample']['class']}`. The original drawing, stroke alpha, and geometry stay fixed across colors.", '',
              f"Selection rule: {backgrounds['selection']}", '',
              '| Model | Background | Predicted class | Target probability | Target margin | Probability change from baseline (points) |',
              '| --- | --- | --- | ---: | ---: | ---: |']
    for model in ['balanced','shortcut']:
        for record in backgrounds['records'][model]:
            value=record['modified']; label=backgrounds['sample']['label']
            lines.append(f"| {model} | {record['request']['interventions'][0]['background']} | {value['predicted_class']} | {value['probabilities'][label]*100:.2f}% | {value['target_logit_margin']:.4f} | {record['comparison']['target_probability_point_change']:+.2f} |")
    lines += ['', '![Matched background responses](../results/phase2/shortcut-trap.png)', '',
              f"Across the three training colors, true-class probability ranges are **{backgrounds['probability_ranges']['balanced']*100:.2f} points** for the balanced model and **{backgrounds['probability_ranges']['shortcut']*100:.2f} points** for the shortcut model. Gray is outside training support and is displayed separately.", '',
              '**Observe → hypothesize → intervene → reflect:** Start with the model identities hidden as A/B, record which one you expect to depend more on color, switch backgrounds, and compare the results. Reveal the training regimes afterward. Keeping the drawing fixed helps isolate the tested background intervention; it does not show that every shortcut-model prediction relies equally on color.', '',
              '**Learning objective:** Recognize how a model can use correlations in its training examples and test reliance through a controlled comparison.', '',
              '## Actual before/after outputs', '',
              'All selected experiments retain raw logits, uncalibrated softmax probabilities, target margins, signed changes, replay requests, source/checkpoint/manifest hashes, and operation metadata in `results/phase2/selected-examples.json`. The following compact table makes the selected patch and channel effects directly inspectable.', '',
              '| Operation | Prediction before → after | Target probability before → after | Target margin before → after |',
              '| --- | --- | --- | --- |']
    operations=[('Highest-saliency patch',patches['interventions'][0]),('Largest-effect patch',patches['interventions'][1]),
                ('Highest-activity channel',channels['interventions'][0]['experiment']),('Largest-effect channel',channels['interventions'][1]['experiment'])]
    for name,result in operations:
        before,after=result['baseline'],result['modified']; target=CLASSES.index(before['target_class'])
        lines.append(f"| {name} | {before['predicted_class']} → {after['predicted_class']} | {before['probabilities'][target]*100:.2f}% → {after['probabilities'][target]*100:.2f}% | {before['target_logit_margin']:.4f} → {after['target_logit_margin']:.4f} |")
    lines += ['',
              'A probability and a margin can move in opposite directions because they summarize three class scores differently: softmax uses every score, while the margin compares the target only with its strongest competitor. Saliency here differentiates the target logit; the occlusion comparison also includes the competitor through the margin. These measurements answer related but different questions. A rank mismatch in a curated example is not evidence that all saliency maps are unreliable.', '',
              '## Engineering verification and replay', '',
              f"- {verification['reset_checks']} independent reset/replay checks passed with exactly equal baseline logits. Shared model tensor hashes are unchanged after all scans.",
              '- Post-ReLU ablation location, unchanged other channels at injection, gradients, input mask geometry, no cross-request contamination, invalid requests, body/frequency limits, and readiness failures are covered by automated tests.',
              '- API model/sample loading verifies checkpoint, tensor-content, parent-manifest, source-bitmap, and input-contract hashes; requests cannot provide code, model files, arbitrary URLs, or raw images.',
              '- Saliency and activations retain raw numeric values; figure contrast scaling is labeled. Fixed-gray occlusion and source-stroke erasure are distinct operations.', '',
              '```sh', 'uv sync --locked --extra dev', 'uv run --locked uvicorn backend.main:app --host 127.0.0.1 --port 8000',
              '# In another terminal:', 'uv run --locked python -m experiments.phase2_demo', 'uv run --locked pytest -q', '```', '',
              'The small source bundle and model checkpoints are included, so the API and report run offline without the bulk dataset files. To reconstruct the bundle from verified full source arrays, run `uv run --locked python -m backend.catalog`.', '',
              'See [API usage and replay semantics](phase2-api.md). No cloud resources, accounts, database, frontend, or interactive challenge flow has been created.', '',
              '## Student learning and poster takeaway', '',
              '**A useful question is: “What changes the model’s decision?”** A heatmap shows local sensitivity. A bright internal map shows activity. Covering pixels, switching a background, or turning off a channel lets us measure an effect. These are different kinds of evidence.', '',
              'Students should practice making a prediction **before** the intervention, comparing before/after scores, and revising the explanation when the measurement disagrees. The examples supply teaching material for that process; they do not measure whether students learned it.', '',
              '**Poster takeaway:** FeatureDive makes model explanations testable. Students can compare what appears important with what measurably changes a prediction, and investigate how an accidental background clue influences a model.', '',
              '**Limitations:** These are curated training examples from one pair of checkpoints. They are not population estimates, a complete semantic circuit analysis, or evidence of educational effectiveness. Input masking can create unusual patterns; results depend on the selected class, mask policy, layer, and channel. Formal held-out research evaluation remains Phase 5; evaluating student learning would require a separate user study.', '',
              '**Gate: Phase 2 complete. Stop for review before Phase 3.**', '']
    (ROOT/'docs/phase2-interpretability.md').write_text('\n'.join(lines))


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    engine=InterpretabilityEngine()
    from ml.train import weights_sha256
    initial={name:weights_sha256(model.state_dict()) for name,model in engine.models.items()}
    patches=patch_scan(engine)
    print('Completed 1,176 independent patch occlusions',flush=True)
    channels=channel_scan(engine)
    print('Completed 576 independent channel ablations',flush=True)
    backgrounds=background_scan(engine)
    resets=0
    for sample in engine.samples.values():
        for model in ('balanced','shortcut'):
            base=base_request(sample,model)
            original=engine.infer(ExperimentRequest(**base))
            engine.infer(ExperimentRequest(**base,interventions=[{'type':'channel_ablation','layer':'conv1','channel':0}]))
            assert engine.infer(ExperimentRequest(**base))==original
            resets+=1
    assert initial=={name:weights_sha256(model.state_dict()) for name,model in engine.models.items()}
    verification={'reset_checks':resets,'weights_unchanged':True,'catalog_sources':len(engine.samples),
                  'source_partition':'train','patch_rows':1176,'channel_rows':576,'background_rows':192,
                  'dependency_lock_sha256':sha256(ROOT/'uv.lock'),
                  'source_code_sha256':{name:sha256(ROOT/name) for name in ['backend/engine.py','backend/schemas.py','backend/catalog.py','backend/main.py','backend/limits.py','ml/interpretability.py','experiments/phase2_demo.py']}}
    write_json(OUTPUT/'verification.json',verification)
    selected={'saliency_vs_occlusion':patches,'activation_vs_influence':channels,'shortcut_backgrounds':backgrounds}
    write_json(OUTPUT/'selected-examples.json',selected)
    figures(patches,channels,backgrounds)
    report(engine,patches,channels,backgrounds,verification)
    print('Generated docs/phase2-interpretability.md and real example artifacts',flush=True)


if __name__=='__main__':
    main()
