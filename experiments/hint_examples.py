"""Find teaching examples for baseline-only gradient hints; replay actual ablations/erasures.

This is curation on registered training drawings, not a held-out evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from backend.engine import InterpretabilityEngine
from backend.schemas import ExperimentRequest
from ml.common import CLASSES, ROOT
from ml.interpretability import margins


def baseline_hints(engine, request):
    """Use only the original input; no modified input or ablation enters these hints."""
    rendered = engine.render(request)
    value = rendered.rgb.detach().clone().requires_grad_(True)
    logits, maps = engine.models[request.model_id](value, capture=True)
    target = CLASSES.index(request.target_class)
    margin = margins(logits, torch.tensor([target]))[0]
    pixel_grad, conv1_grad, conv2_grad = torch.autograd.grad(margin,
        (value, maps['conv1'], maps['conv2']), retain_graph=True)
    logit_grad, = torch.autograd.grad(logits[0, target], value)
    color = torch.tensor(engine.registry['background_rgb'][engine.registry['background_names'].index(request.background)]).float()/255
    erase_delta = color[:, None, None] - value.detach()[0]
    return {'rgb': value.detach(), 'logits': logits.detach(), 'margin': float(margin.detach()),
        'color': color, 'alpha': rendered.alpha,
        'erasure_delta_estimate': (pixel_grad[0]*erase_delta).sum(0).detach().numpy(),
        'raw_logit_saliency': logit_grad[0].abs().sum(0).detach().numpy(),
        'channels': {layer: {'mean_activity': maps[layer].detach().mean((0,2,3)).numpy(),
            'estimated_margin_change': -(gradient*maps[layer]).detach().sum((0,2,3)).numpy()}
            for layer,gradient in (('conv1',conv1_grad),('conv2',conv2_grad))}}


def scan(engine, request, hints):
    model = engine.models[request.model_id]
    target = CLASSES.index(request.target_class)
    base = {'model':request.model_id, 'sample':request.sample_id, 'background':request.background,
        'target':request.target_class, 'before':CLASSES[int(hints['logits'].argmax())],
        'baseline_probability':float(hints['logits'].softmax(1)[0,target]), 'baseline_margin':hints['margin']}
    base['initially_correct'] = base['before'] == request.target_class
    channels = []
    for layer, scores in hints['channels'].items():
        for channel, estimate in enumerate(scores['estimated_margin_change']):
            with torch.no_grad(): after = model(hints['rgb'], ablation=(layer,channel))
            channels.append({**base,'layer':layer,'channel':channel,
                'mean_activity':float(scores['mean_activity'][channel]),
                'estimated_margin_change':float(estimate),
                'actual_margin_change':float(margins(after,torch.tensor([target]))[0])-hints['margin'],
                'after':CLASSES[int(after.argmax())], 'after_probability':float(after.softmax(1)[0,target])})
    for row in channels:
        row['hint_rank_all_channels'] = 1+sum(other['estimated_margin_change']<row['estimated_margin_change'] for other in channels)
        row['activity_rank_in_layer'] = 1+sum(other['layer']==row['layer'] and other['mean_activity']>row['mean_activity'] for other in channels)
    patches = []
    for size in (1,4):
        positions = [(x,y) for y in range(29-size) for x in range(29-size)]
        modified = hints['rgb'].repeat(len(positions),1,1,1)
        for i,(x,y) in enumerate(positions):
            # Identical to stroke erasure: nonstroke pixels already equal background.
            modified[i,:,y:y+size,x:x+size] = hints['color'][:,None,None]
        with torch.no_grad():
            after = model(modified)
            after_margin = margins(after,torch.full((len(positions),),target))
            probabilities = after.softmax(1)[:,target]
        group = []
        for i,(x,y) in enumerate(positions):
            group.append({**base,'x':x,'y':y,'width':size,'height':size,
                'erased_stroke_pixels':int((hints['alpha'][y:y+size,x:x+size]>0).sum()),
                'estimated_margin_change':float(hints['erasure_delta_estimate'][y:y+size,x:x+size].sum()),
                'saliency_sum':float(hints['raw_logit_saliency'][y:y+size,x:x+size].sum()),
                'actual_margin_change':float(after_margin[i])-hints['margin'],
                'after':CLASSES[int(after[i].argmax())], 'after_probability':float(probabilities[i])})
        estimates = np.array([row['estimated_margin_change'] for row in group])
        saliency = np.array([row['saliency_sum'] for row in group])
        for row in group:
            row['hint_rank'] = int(1+(estimates<row['estimated_margin_change']).sum())
            row['saliency_rank'] = int(1+(saliency>row['saliency_sum']).sum())
            row['candidate_count'] = len(group)
        patches.extend(group)
    return channels, patches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results/hint-examples')
    args = parser.parse_args()
    engine = InterpretabilityEngine()
    channel_rows, selected_patches, summaries, cached = [], [], [], {}
    replay_sources = {'fish:20708','cat:89133','airplane:72939','fish:2756'}
    for model_id in engine.models:
        for sample in engine.samples.values():
            request = ExperimentRequest(model_id=model_id,sample_id=sample['id'],target_class=sample['class'],
                background=('green','blue','yellow')[sample['label']],interventions=[])
            hints = baseline_hints(engine,request)
            channels, patches = scan(engine,request,hints)
            channel_rows.extend(channels)
            pixel_rows = [row for row in patches if row['width']==1]
            brush_rows = [row for row in patches if row['width']==4]
            summaries.append({'model':model_id,'sample':sample['id'],
                'initially_correct':channels[0]['initially_correct'],
                'single_pixel_candidates':len(pixel_rows),
                'single_pixel_flips':sum(row['before']!=row['after'] for row in pixel_rows),
                'brush_candidates':len(brush_rows),
                'top_brush_hint':min(brush_rows,key=lambda row:row['estimated_margin_change'])})
            if model_id=='balanced' and sample['id'] in replay_sources:
                cached[sample['id']] = (request,hints)
                selected_patches.extend(brush_rows)
    replays = []
    for sid, layer, channel in (('fish:20708','conv1',7),('fish:20708','conv1',4),('cat:89133','conv1',3)):
        request,_ = cached[sid]
        operation = {'type':'channel_ablation','layer':layer,'channel':channel}
        measured = engine.infer(ExperimentRequest(**{**request.model_dump(),'interventions':[operation]}))
        row = next(row for row in channel_rows if row['model']=='balanced' and row['sample']==sid and row['layer']==layer and row['channel']==channel)
        replays.append({'kind':'channel','hint':row,'result':measured.model_dump(mode='json')})
    for sid,x,y in (('airplane:72939',4,14),('airplane:72939',6,14),('fish:2756',22,16),('fish:2756',6,13)):
        request,_ = cached[sid]
        operation = {'type':'drawing_erasure','region':{'rectangle':{'x':x,'y':y,'width':4,'height':4}}}
        measured = engine.infer(ExperimentRequest(**{**request.model_dump(),'interventions':[operation]}))
        row = next(row for row in selected_patches if row['sample']==sid and row['x']==x and row['y']==y)
        if abs(measured.modified.target_logit_margin - measured.baseline.target_logit_margin - row['actual_margin_change'])>1e-5:
            raise RuntimeError('Batched scan does not match the actual erasure engine.')
        replays.append({'kind':'stroke_erasure','hint':row,'result':measured.model_dump(mode='json')})
    args.output_dir.mkdir(parents=True,exist_ok=True)
    output = {'scope':'Curated training catalog; both models; category-associated backgrounds; no held-out or educational-benefit claim.',
        'hint_objective':'selected class logit minus strongest competing logit',
        'hints_use_only_baseline':True,'model_version':engine.registry['model_version'],
        'checkpoint_sha256':{key:value['checkpoint_sha256'] for key,value in engine.model_metadata.items()},
        'demo_manifest_sha256':engine.demo_manifest_sha256,
        'channel_scans':channel_rows,'catalog_summary':summaries,'replays':replays}
    (args.output_dir/'hint-examples.json').write_text(json.dumps(output,indent=2)+'\n')
    figures(args.output_dir,cached,replays,channel_rows)
    print(f'Saved baseline hints, actual engine replays, and figures to {args.output_dir}')
    print('Initially correct single-pixel flips:',sum(row['single_pixel_flips'] for row in summaries if row['initially_correct']))


def figures(output,cached,replays,channels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    request,hints = cached['airplane:72939']
    cases = [row for row in replays if row['kind']=='stroke_erasure' and row['hint']['sample']==request.sample_id]
    fig,axes = plt.subplots(1,4,figsize=(13,3.5),constrained_layout=True)
    axes[0].imshow(hints['rgb'][0].permute(1,2,0).numpy());axes[0].set_title(f"Original airplane\n{100*float(hints['logits'].softmax(1)[0,2]):.2f}% airplane")
    axes[0].add_patch(Rectangle((3.5,13.5),4,4,fill=False,edgecolor='black',linewidth=2))
    support = -hints['erasure_delta_estimate'];lim=float(np.abs(support).max())
    plot=axes[1].imshow(support,cmap='RdBu_r',vmin=-lim,vmax=lim,interpolation='nearest')
    axes[1].set_title('Baseline-only erase hint\nred: estimated support')
    fig.colorbar(plot,ax=axes[1],shrink=.7,label='estimated margin loss per pixel')
    for axis,case in zip(axes[2:],cases):
        result=case['result'];axis.imshow(np.asarray(result['input']['modified_rgb_chw']).transpose(1,2,0))
        axis.set_title(f"Erase ({case['hint']['x']},{case['hint']['y']}) → {result['modified']['predicted_class']}\n{100*result['modified']['probabilities'][2]:.2f}% airplane")
    for axis in axes:axis.set_xticks([]);axis.set_yticks([])
    fig.savefig(output/'erasure-hints.png',dpi=180);plt.close(fig)
    rows=[row for row in channels if row['model']=='balanced' and row['sample']=='fish:20708' and row['layer']=='conv1']
    fig,axes=plt.subplots(1,3,figsize=(12,3.5),constrained_layout=True)
    colors=['#d7e2d6']*8;colors[4]='#dfb365';colors[7]='#376f5b'
    axes[0].bar(range(8),[row['mean_activity'] for row in rows],color=colors);axes[0].set_title('Activity alone\nchannel 4 is brightest');axes[0].set_ylabel('Mean post-ReLU response')
    axes[1].bar(range(8),[-row['estimated_margin_change'] for row in rows],color=colors);axes[1].set_title('Baseline-only support hint\nchannel 7 has the largest estimate');axes[1].set_ylabel('Estimated margin loss if switched off')
    axes[2].bar(range(8),[100*row['after_probability'] for row in rows],color=colors);axes[2].axhline(100*rows[0]['baseline_probability'],color='#666',linestyle='--',label='Original fish probability');axes[2].set_title('Actual independent ablations\nchannel 7 flips fish → airplane');axes[2].set_ylabel('Fish probability (%)');axes[2].legend(fontsize=7)
    for axis in axes:axis.set_xticks(range(8));axis.set_xlabel('Conv1 channel (zero-indexed)')
    fig.savefig(output/'channel-hints.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    main()
