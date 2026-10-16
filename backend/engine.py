"""Stateless CPU interpretability with request-local input transformations."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from backend.catalog import DEFAULT_CONFIG
from backend.schemas import (
    ActivationRequest, ActivationResponse, BackgroundSwap, ChannelAblation,
    Comparison, DrawingErasure, ExperimentRequest, ExperimentResponse,
    Occlusion, Prediction, Region, SaliencyResponse, HintRequest, HintResponse,
)
from ml.backgrounds import COLOR_NAMES, RGB
from ml.common import CLASSES, ROOT, read_json, sha256
from ml.data_loader import validate_manifest
from ml.interpretability import margins, saliency, decision_support_hints
from ml.train import load_checkpoint


@dataclass
class RenderedInput:
    source: torch.Tensor
    alpha: torch.Tensor
    rgb: torch.Tensor
    background: str
    ablation: tuple[str, int] | None
    log: list[dict]


def region_mask(region: Region) -> torch.Tensor:
    mask = torch.zeros(28, 28, dtype=torch.bool)
    if region.rectangle is not None:
        box = region.rectangle
        mask[box.y:box.y + box.height, box.x:box.x + box.width] = True
    else:
        for pixel in region.pixels:
            mask[pixel.y, pixel.x] = True
    return mask


def describe_prediction(logits: torch.Tensor, target: int) -> Prediction:
    value = logits.detach().cpu()
    if not bool(torch.isfinite(value).all()):
        raise ValueError('Nonfinite model output')
    return Prediction(
        logits=value[0].tolist(), probabilities=value.softmax(1)[0].tolist(),
        predicted_class=CLASSES[int(value[0].argmax())], target_class=CLASSES[target],
        target_logit=float(value[0, target]),
        target_logit_margin=float(margins(value, torch.tensor([target]))[0]),
    )


class InterpretabilityEngine:
    def __init__(self, config_path: Path = DEFAULT_CONFIG) -> None:
        config = read_json(config_path)
        torch.set_num_threads(config['threads'])
        self.registry = read_json(ROOT / config['registry'])
        self.manifest_path = ROOT / self.registry['manifest']
        if sha256(self.manifest_path) != self.registry['manifest_sha256']:
            raise ValueError('Registry manifest checksum mismatch')
        manifest = read_json(self.manifest_path)
        validate_manifest(manifest)
        if (self.registry['class_order'] != list(CLASSES)
                or self.registry['background_names'] != list(COLOR_NAMES)
                or self.registry['background_rgb'] != [list(color) for color in RGB]
                or self.registry['input'] != {
                    'shape_chw': [3, 28, 28], 'dtype': 'float32', 'range': [0, 1],
                    'foreground': 'white', 'alpha': 'source intensity / 255',
                }):
            raise ValueError('Registry input contract differs from engine')
        catalog_path = ROOT / config['demo_manifest']
        catalog = read_json(catalog_path)
        if catalog['parent_manifest_sha256'] != sha256(self.manifest_path):
            raise ValueError('Demo catalog has a different parent manifest')
        parent = {row['id']: row for row in manifest['samples']}
        self.samples = {row['id']: row for row in catalog['samples']}
        if not self.samples or len(self.samples) != len(catalog['samples']) or len(self.samples) > 60:
            raise ValueError('Invalid demo catalog size or duplicate IDs')
        self.drawings = {}
        with np.load(catalog_path.with_suffix('.npz'), allow_pickle=False) as bundle:
            drawings, ids = bundle['drawings'], bundle['sample_ids'].tolist()
            if drawings.dtype != np.uint8 or drawings.shape != (len(self.samples), 28, 28) or ids != list(self.samples):
                raise ValueError('Invalid sample bundle dimensions, dtype, or order')
            for sample_id, drawing in zip(ids, drawings):
                row = self.samples[sample_id]
                if row != parent.get(sample_id) or row['split'] != 'train':
                    raise ValueError('Demo source is not a registered training source')
                if hashlib.sha256(drawing.tobytes()).hexdigest() != row['bitmap_sha256']:
                    raise ValueError('Demo bitmap checksum mismatch')
                self.drawings[sample_id] = torch.from_numpy(drawing.copy())
        self.demo_manifest_sha256 = sha256(catalog_path)
        self.models = {}
        self.model_metadata = {}
        for record in self.registry['models']:
            model_id = record['id']
            if model_id not in ('balanced', 'shortcut') or model_id in self.models:
                raise ValueError('Invalid registered model IDs')
            path = ROOT / record['checkpoint']
            if sha256(path) != record['checkpoint_sha256']:
                raise ValueError('Registered checkpoint checksum mismatch')
            model, saved = load_checkpoint(path, self.manifest_path)
            metadata = saved['metadata']
            if (saved['weights_sha256'] != record['weights_sha256'] or metadata['regime'] != model_id
                    or metadata['model_version'] != self.registry['model_version']
                    or metadata['background_rgb'] != list(RGB)):
                raise ValueError('Checkpoint metadata differs from registry')
            # Parameters are shared read-only; gradients are requested only w.r.t. local inputs.
            model.requires_grad_(False)
            self.models[model_id] = model
            self.model_metadata[model_id] = record
        if set(self.models) != {'balanced', 'shortcut'}:
            raise ValueError('Both registered models are required')

    def render(self, request: ExperimentRequest) -> RenderedInput:
        if request.sample_id not in self.samples:
            raise KeyError('Unknown sample ID; use GET /api/samples')
        source = self.drawings[request.sample_id].clone()
        alpha = source.float() / 255
        background = request.background
        overlay_mask = torch.zeros(28, 28, dtype=torch.bool)
        overlay_rgb = torch.zeros(3, 28, 28)
        ablation = None
        log = []
        for operation in request.interventions:
            record = operation.model_dump(mode='json')
            if isinstance(operation, BackgroundSwap):
                record.update(original_background=background, original_rgb=list(RGB[COLOR_NAMES.index(background)]), replacement_rgb=list(RGB[COLOR_NAMES.index(operation.background)]))
                background = operation.background
            elif isinstance(operation, ChannelAblation):
                ablation = (operation.layer, operation.channel)
                record.update(injection_point='post_relu_before_pool', replacement_value=0.0)
            else:
                mask = region_mask(operation.region)
                record['selected_pixel_count'] = int(mask.sum())
                record['coordinates_xy'] = [[int(x), int(y)] for y, x in mask.nonzero().tolist()]
                if isinstance(operation, DrawingErasure):
                    affected = mask & (alpha > 0)
                    alpha[affected] = 0
                    overlay_mask[affected] = False
                    record.update(affected_pixel_count=int(affected.sum()), replacement_rgb=list(RGB[COLOR_NAMES.index(background)]), rule='set selected remaining foreground alpha to zero; leave nonforeground pixels unchanged')
                elif isinstance(operation, Occlusion):
                    overlay_mask[mask] = True
                    overlay_rgb[:, mask] = torch.tensor(operation.replacement_rgb).float()[:, None] / 255
                    record.update(affected_pixel_count=int(mask.sum()), mask_policy='fixed_rgb_full_region')
            log.append(record)
        color = torch.tensor(RGB[COLOR_NAMES.index(background)]).float()[:, None, None] / 255
        rgb = alpha.unsqueeze(0) + (1 - alpha.unsqueeze(0)) * color
        rgb[:, overlay_mask] = overlay_rgb[:, overlay_mask]
        return RenderedInput(source, alpha, rgb.unsqueeze(0), background, ablation, log)

    def _inputs(self, request: ExperimentRequest) -> tuple[RenderedInput, RenderedInput]:
        baseline = request.model_copy(update={'interventions': []})
        return self.render(baseline), self.render(request)

    def _describe(self, request, baseline, current, baseline_logits, current_logits) -> ExperimentResponse:
        target = CLASSES.index(request.target_class)
        before = describe_prediction(baseline_logits, target)
        after = describe_prediction(current_logits, target)
        metadata = self.model_metadata[request.model_id]
        canonical = json.dumps(request.model_dump(mode='json'), sort_keys=True, separators=(',', ':'))
        identifier = hashlib.sha256((canonical + metadata['checkpoint_sha256'] + self.demo_manifest_sha256).encode()).hexdigest()
        return ExperimentResponse(
            experiment_id=identifier, request=request, model_version=self.registry['model_version'],
            checkpoint_sha256=metadata['checkpoint_sha256'], weights_sha256=metadata['weights_sha256'],
            manifest_sha256=self.registry['manifest_sha256'], demo_manifest_sha256=self.demo_manifest_sha256,
            source_bitmap_sha256=self.samples[request.sample_id]['bitmap_sha256'], source_partition='train',
            baseline=before, modified=after,
            comparison=Comparison(
                logit_changes=[a - b for a, b in zip(after.logits, before.logits)],
                probability_point_changes=[100 * (a - b) for a, b in zip(after.probabilities, before.probabilities)],
                target_logit_change=after.target_logit - before.target_logit,
                target_logit_margin_change=after.target_logit_margin - before.target_logit_margin,
                target_probability_point_change=100 * (after.probabilities[target] - before.probabilities[target]),
                classification_flip=before.predicted_class != after.predicted_class,
            ),
            intervention_log=current.log,
            input={
                'shape_chw': [3, 28, 28], 'rgb_range': [0, 1], 'source': baseline.source.tolist(),
                'source_alpha': baseline.alpha.tolist(), 'modified_alpha': current.alpha.tolist(),
                'baseline_rgb_chw': baseline.rgb[0].tolist(), 'modified_rgb_chw': current.rgb[0].tolist(),
                'baseline_background': baseline.background, 'modified_background': current.background,
                'modified_background_rgb': list(RGB[COLOR_NAMES.index(current.background)]),
                'active_channel_ablation': list(current.ablation) if current.ablation else None,
            },
        )

    def infer(self, request: ExperimentRequest) -> ExperimentResponse:
        baseline, current = self._inputs(request)
        model = self.models[request.model_id]
        with torch.no_grad():
            before = model(baseline.rgb)
            after = model(current.rgb, ablation=current.ablation)
        return self._describe(request, baseline, current, before, after)

    def activations(self, request: ActivationRequest) -> ActivationResponse:
        experiment = ExperimentRequest.model_validate(request.model_dump(exclude={'layer'}))
        baseline, current = self._inputs(experiment)
        model = self.models[request.model_id]
        with torch.no_grad():
            before, baseline_maps = model(baseline.rgb, capture=True)
            after, current_maps = model(current.rgb, capture=True, ablation=current.ablation)
        first, second = baseline_maps[request.layer][0], current_maps[request.layer][0]
        average = lambda tensor: tensor.mean(dim=(1, 2)).tolist() if tensor.ndim == 3 else tensor.tolist()
        return ActivationResponse(
            experiment=self._describe(experiment, baseline, current, before, after), layer=request.layer,
            baseline_values=first.tolist(), modified_values=second.tolist(),
            baseline_mean_by_channel=average(first), modified_mean_by_channel=average(second),
            baseline_min=float(first.min()), baseline_max=float(first.max()),
            modified_min=float(second.min()), modified_max=float(second.max()),
        )

    def saliency(self, request: ExperimentRequest) -> SaliencyResponse:
        baseline, current = self._inputs(request)
        model = self.models[request.model_id]
        target = CLASSES.index(request.target_class)
        return SaliencyResponse(
            experiment=self.infer(request), target_class=request.target_class,
            baseline_scores=saliency(model, baseline.rgb, target)[0].tolist(),
            modified_scores=saliency(model, current.rgb, target, ablation=current.ablation)[0].tolist(),
        )

    def hints(self, request: HintRequest) -> HintResponse:
        if request.interventions:
            raise ValueError('Hints require an unmodified baseline')
        rendered = self.render(request)
        color = torch.tensor(RGB[COLOR_NAMES.index(request.background)]).float() / 255
        values = decision_support_hints(self.models[request.model_id], rendered.rgb,
            CLASSES.index(request.target_class), color)
        channels = []
        for layer, scores in values['channels'].items():
            for channel, support in enumerate(scores['support'].tolist()):
                channels.append(dict(layer=layer, channel=channel, estimated_support=support,
                    estimated_margin_change=-support,
                    mean_activation=float(scores['mean_activation'][channel])))
        channels.sort(key=lambda row: -row['estimated_support'])
        for row in channels:
            row['rank'] = 1 + sum(other['estimated_support'] > row['estimated_support'] for other in channels)
        support = values['erase_support']
        # All overlapping 4x4 windows; the ranking uses baseline gradients only.
        windows = support.unfold(0, 4, 1).unfold(1, 4, 1).sum((-1, -2))
        index = int(windows.argmax())
        y, x = divmod(index, 25)
        best = float(windows[y, x])
        patch = dict(rectangle=dict(x=x, y=y, width=4, height=4),
            estimated_support=best, estimated_margin_change=-best) if best > 0 else None
        return HintResponse(experiment=self.infer(request), channels=channels,
            erase_support=support.tolist(), suggested_patch=patch)

    def sample_catalog(self) -> list[dict]:
        return [{**row, 'role': 'illustrative_training_example'} for row in self.samples.values()]

    def model_catalog(self) -> dict:
        return {'model_version': self.registry['model_version'], 'class_order': list(CLASSES),
                'background_names': list(COLOR_NAMES), 'background_rgb': [list(rgb) for rgb in RGB],
                'models': list(self.model_metadata.values()), 'inference_device': 'cpu'}
