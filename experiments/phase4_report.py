"""Summarize real Phase 4 browser exports; never infer or invent measurements."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records-dir', type=Path, default=ROOT / 'results/phase4')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/phase4')
    args = parser.parse_args()
    registry = json.loads((ROOT / 'checkpoints/phase1-v1/registry.json').read_text())
    checkpoints = {model['id']: model['checkpoint_sha256'] for model in registry['models']}
    summary = {'scope': 'curated training examples; automated acceptance, not a student study',
               'model_version': registry['model_version'], 'records': {}, 'comparisons': {}}
    lines = ['# Phase 4 measured demo results', '',
             'Generated from completed browser exports by `python -m experiments.phase4_report`.',
             'These are real CPU checkpoint responses on curated training examples.',
             'Hypotheses and reflections in these acceptance records were written by the test author, not students.', '']
    for name in ('evidence', 'hidden', 'shortcut'):
        path = args.records_dir / f'{name}-acceptance-record.json'
        raw = path.read_bytes()
        record = json.loads(raw)
        if record['challenge_id'] != name or not record.get('completed_at') or record['source_role'] != 'illustrative_training_example':
            raise ValueError(f'{path}: expected a completed training-example record')
        summary['records'][name] = {'file': path.name, 'sha256': hashlib.sha256(raw).hexdigest()}
        results = ([trial['result'] for trial in record['trials']] if name != 'shortcut' else
                   [result for row in record['observations'] for result in row['results'].values()])
        for result in results:
            if result['checkpoint_sha256'] != checkpoints[result['request']['model_id']] or result['model_version'] != registry['model_version'] or result['source_partition'] != 'train':
                raise ValueError(f'{path}: result does not match the registered training-example demo')
        lines += [f'## {name.title()}', '', f'Source: [{path.name}]({path.name}).', '']
        rows = []
        if name != 'shortcut':
            lines += ['| Selection | Saliency sum / mean activity | Target margin Δ | Target probability Δ (points) |',
                      '| --- | ---: | ---: | ---: |']
            for trial in record['trials']:
                selection, result = trial['selection'], trial['result']
                label = (f"patch ({selection['x']},{selection['y']})" if name == 'evidence' else
                         f"{selection['layer']} channel {selection['channel']}")
                score = selection['saliency_sum' if name == 'evidence' else 'mean_activation']
                delta = result['comparison']['target_logit_margin_change']
                probability_delta = result['comparison']['target_probability_point_change']
                rows.append({'selection': selection, 'sample_id': result['request']['sample_id'],
                             'target_margin_change': delta, 'target_probability_point_change': probability_delta,
                             'feedback': trial['feedback']})
                lines.append(f'| {label} | {score:.4f} | {delta:+.4f} | {probability_delta:+.2f} |')
        else:
            lines += ['| Background | Revealed model | Prediction | Target probability | Target margin |',
                      '| --- | --- | --- | ---: | ---: |']
            for row in record['observations']:
                for alias, result in row['results'].items():
                    target_index = registry['class_order'].index(result['request']['target_class'])
                    probability = result['modified']['probabilities'][target_index] * 100
                    margin = result['modified']['target_logit_margin']
                    rows.append({'background':row['background'], 'model':record['model_mapping'][alias],
                                 'predicted_class':result['modified']['predicted_class'],
                                 'target_probability_percent':probability, 'target_margin':margin})
                    lines.append(f"| {row['background']} | {record['model_mapping'][alias]} | {result['modified']['predicted_class']} | {probability:.2f}% | {margin:.4f} |")
            lines += ['', '| Revealed model | Target probability range (points) | Target margin range | Any label flip |',
                      '| --- | ---: | ---: | --- |']
            for alias in ('A', 'B'):
                scores = record['background_feedback'][alias]
                lines.append(f"| {record['model_mapping'][alias]} | {scores['probabilityRangePoints']:.2f} | {scores['marginRange']:.4f} | {'yes' if scores['anyLabelFlip'] else 'no'} |")
        summary['comparisons'][name] = rows
        lines += ['']
    lines += ['## What these examples support', '',
              'Compare local sensitivity with the measured effect of a larger input change; compare average channel activity with its measured ablation effect; compare backgrounds with the foreground fixed.',
              'No aggregate held-out result, unique semantic mechanism, or student learning gain follows from these three examples.', '']
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'demo-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (args.output_dir / 'demo-results.md').write_text('\n'.join(lines))
    print(f'Wrote measured demo summary to {args.output_dir}')


if __name__ == '__main__':
    main()
