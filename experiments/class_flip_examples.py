"""Scan the small teaching catalog for single-channel class flips (not held-out evaluation)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from backend.engine import InterpretabilityEngine
from backend.schemas import ExperimentRequest
from ml.common import CLASSES, ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/phase4-quick')
    args = parser.parse_args()
    engine = InterpretabilityEngine()
    cases = []
    candidates = 0
    for model_id, model in engine.models.items():
        for sample_id, sample in engine.samples.items():
            request = ExperimentRequest(model_id=model_id, sample_id=sample_id,
                background=('green', 'blue', 'yellow')[sample['label']],
                target_class=sample['class'], interventions=[])
            rgb = engine.render(request).rgb
            with torch.no_grad():
                before = model(rgb)
                for layer, channels in (('conv1', 8), ('conv2', 16)):
                    for channel in range(channels):
                        candidates += 1
                        after = model(rgb, ablation=(layer, channel))
                        if int(before.argmax()) != int(after.argmax()):
                            cases.append({'request': request.model_dump(mode='json'),
                                'layer':layer, 'channel':channel,
                                'baseline_prediction':CLASSES[int(before.argmax())],
                                'modified_prediction':CLASSES[int(after.argmax())],
                                'baseline_probabilities':before.softmax(1)[0].tolist(),
                                'modified_probabilities':after.softmax(1)[0].tolist()})
    featured = []
    for source, color, target, layer, channel in (
        ('cat:89133', 'green', 'cat', 'conv1', 3),
        ('fish:20708', 'blue', 'fish', 'conv1', 7),
        ('airplane:4513', 'yellow', 'airplane', 'conv2', 3),
    ):
        request = ExperimentRequest(model_id='balanced',sample_id=source,background=color,target_class=target,
            interventions=[{'type':'channel_ablation','layer':layer,'channel':channel}])
        result = engine.infer(request)
        if not result.comparison.classification_flip:
            raise RuntimeError(f'Featured example no longer flips: {source}; recurate, do not invent an outcome.')
        featured.append(result.model_dump(mode='json'))
    output = {'scope':'24 curated training drawings, each on its category-associated background; 2 checkpoints; single channels only; not a held-out statistic',
        'candidate_interventions':candidates,'flipping_interventions':len(cases),
        'model_version':engine.registry['model_version'],
        'checkpoint_sha256':{key:value['checkpoint_sha256'] for key,value in engine.model_metadata.items()},
        'demo_manifest_sha256':engine.demo_manifest_sha256,'cases':cases,'featured_examples':featured}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    path=args.output_dir/'class-flip-examples.json'
    path.write_text(json.dumps(output,indent=2)+'\n')
    print(f'{len(cases)} flips among {candidates} catalog interventions. Saved {path}')
    for result in featured:
        print(result['request']['sample_id'],result['request']['interventions'][0],
              result['baseline']['predicted_class'],'→',result['modified']['predicted_class'])


if __name__ == '__main__':
    main()
