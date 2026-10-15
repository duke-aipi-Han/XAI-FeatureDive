"""Explicit, bounded replayable requests and typed inference responses."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ModelID = Literal['balanced', 'shortcut']
ClassName = Literal['cat', 'fish', 'airplane']
Background = Literal['green', 'blue', 'yellow', 'gray']
Layer = Literal['conv1', 'conv2', 'dense']
Coordinate = Annotated[int, Field(strict=True, ge=0, le=27)]
Byte = Annotated[int, Field(strict=True, ge=0, le=255)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)


class Pixel(StrictModel):
    x: Coordinate
    y: Coordinate


class Rectangle(StrictModel):
    x: Coordinate
    y: Coordinate
    width: Annotated[int, Field(strict=True, ge=1, le=28)]
    height: Annotated[int, Field(strict=True, ge=1, le=28)]

    @model_validator(mode='after')
    def within_image(self):
        if self.x + self.width > 28 or self.y + self.height > 28:
            raise ValueError('rectangle extends beyond the 28×28 input')
        return self


class Region(StrictModel):
    pixels: list[Pixel] | None = Field(default=None, min_length=1, max_length=784)
    rectangle: Rectangle | None = None

    @model_validator(mode='after')
    def exactly_one_selection(self):
        if (self.pixels is None) == (self.rectangle is None):
            raise ValueError('provide exactly one of pixels or rectangle')
        if self.pixels and len({(pixel.x, pixel.y) for pixel in self.pixels}) != len(self.pixels):
            raise ValueError('duplicate pixel coordinates')
        return self


class DrawingErasure(StrictModel):
    type: Literal['drawing_erasure']
    region: Region


class Occlusion(StrictModel):
    type: Literal['controlled_occlusion']
    region: Region
    # A JSON list, not tuple coercion; this is a fixed replacement policy.
    replacement_rgb: list[Byte] = Field(default_factory=lambda: [128, 128, 128], min_length=3, max_length=3)


class BackgroundSwap(StrictModel):
    type: Literal['background_swap']
    background: Background


class ChannelAblation(StrictModel):
    type: Literal['channel_ablation']
    layer: Literal['conv1', 'conv2']
    channel: Annotated[int, Field(strict=True, ge=0, le=15)]

    @model_validator(mode='after')
    def valid_channel(self):
        if self.channel >= (8 if self.layer == 'conv1' else 16):
            raise ValueError('channel index is outside the selected layer')
        return self


Intervention = Annotated[DrawingErasure | Occlusion | BackgroundSwap | ChannelAblation, Field(discriminator='type')]


class ExperimentRequest(StrictModel):
    model_id: ModelID
    sample_id: str = Field(pattern=r'^(cat|fish|airplane):[0-9]{1,7}$', max_length=32)
    background: Background
    target_class: ClassName
    interventions: list[Intervention] = Field(default_factory=list, max_length=16)

    @model_validator(mode='after')
    def single_ablation(self):
        if sum(isinstance(operation, ChannelAblation) for operation in self.interventions) > 1:
            raise ValueError('at most one active convolutional-channel ablation per experiment')
        return self


class ActivationRequest(ExperimentRequest):
    layer: Layer


class Prediction(StrictModel):
    logits: list[float] = Field(min_length=3, max_length=3)
    probabilities: list[float] = Field(min_length=3, max_length=3)
    predicted_class: ClassName
    target_class: ClassName
    target_logit: float
    target_logit_margin: float
    probability_calibration: Literal['uncalibrated_softmax'] = 'uncalibrated_softmax'


class Comparison(StrictModel):
    logit_changes: list[float]
    probability_point_changes: list[float]
    target_logit_change: float
    target_logit_margin_change: float
    target_probability_point_change: float
    classification_flip: bool


class ExperimentResponse(StrictModel):
    experiment_id: str
    request: ExperimentRequest
    model_version: str
    checkpoint_sha256: str
    weights_sha256: str
    manifest_sha256: str
    demo_manifest_sha256: str
    source_bitmap_sha256: str
    source_partition: Literal['train']
    baseline: Prediction
    modified: Prediction
    comparison: Comparison
    intervention_log: list[dict]
    input: dict


class ActivationResponse(StrictModel):
    experiment: ExperimentResponse
    layer: Layer
    baseline_values: list
    modified_values: list
    baseline_mean_by_channel: list[float]
    modified_mean_by_channel: list[float]
    baseline_min: float
    baseline_max: float
    modified_min: float
    modified_max: float
    kind: Literal['post_relu_activation'] = 'post_relu_activation'
    display_scaling: Literal['raw_values_no_display_normalization'] = 'raw_values_no_display_normalization'


class SaliencyResponse(StrictModel):
    experiment: ExperimentResponse
    target_class: ClassName
    baseline_scores: list[list[float]]
    modified_scores: list[list[float]]
    aggregation: Literal['sum_abs_rgb_class_logit_gradients'] = 'sum_abs_rgb_class_logit_gradients'
    normalization: Literal['none'] = 'none'
    kind: Literal['local_sensitivity_attribution'] = 'local_sensitivity_attribution'
