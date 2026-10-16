export const CLASSES = ['cat', 'fish', 'airplane'] as const;
export const BACKGROUNDS = ['green', 'blue', 'yellow', 'gray'] as const;
export const PLANES = ['input', 'conv1', 'conv2', 'dense', 'output'] as const;
export type ClassName = typeof CLASSES[number];
export type Background = typeof BACKGROUNDS[number];
export type Plane = typeof PLANES[number];
export type Layer = 'conv1' | 'conv2' | 'dense';
export type ModelID = 'balanced' | 'shortcut';
export type Matrix = number[][];
export type RGBImage = number[][][];
export type Region = { rectangle: { x: number; y: number; width: number; height: number } } | { pixels: { x: number; y: number }[] };
export type Intervention =
  | { type: 'drawing_erasure'; region: Region }
  | { type: 'controlled_occlusion'; region: Region; replacement_rgb: number[] }
  | { type: 'background_swap'; background: Background }
  | { type: 'channel_ablation'; layer: 'conv1' | 'conv2'; channel: number };
export interface ExperimentRequest {
  model_id: ModelID; sample_id: string; background: Background; target_class: ClassName;
  interventions: Intervention[];
}
export interface Prediction {
  logits: number[]; probabilities: number[]; predicted_class: ClassName;
  target_class: ClassName; target_logit: number; target_logit_margin: number;
  probability_calibration: 'uncalibrated_softmax';
}
export interface ExperimentResult {
  experiment_id: string; request: ExperimentRequest; model_version: string;
  checkpoint_sha256: string; weights_sha256: string; manifest_sha256: string;
  demo_manifest_sha256: string; source_bitmap_sha256: string; source_partition: 'train';
  baseline: Prediction; modified: Prediction;
  comparison: {
    logit_changes: number[]; probability_point_changes: number[]; target_logit_change: number;
    target_logit_margin_change: number; target_probability_point_change: number; classification_flip: boolean;
  };
  intervention_log: Record<string, unknown>[];
  input: {
    shape_chw: number[]; rgb_range: number[]; source: Matrix;
    source_alpha: Matrix; modified_alpha: Matrix;
    baseline_rgb_chw: RGBImage; modified_rgb_chw: RGBImage;
    baseline_background: Background; modified_background: Background;
    modified_background_rgb: number[]; active_channel_ablation: [Layer, number] | null;
  };
}
export interface ActivationResult {
  experiment: ExperimentResult; layer: Layer;
  baseline_values: RGBImage | number[]; modified_values: RGBImage | number[];
  baseline_mean_by_channel: number[]; modified_mean_by_channel: number[];
  baseline_min: number; baseline_max: number; modified_min: number; modified_max: number;
  kind: 'post_relu_activation'; display_scaling: 'raw_values_no_display_normalization';
}
export interface SaliencyResult {
  experiment: ExperimentResult; target_class: ClassName;
  baseline_scores: Matrix; modified_scores: Matrix;
  aggregation: 'sum_abs_rgb_class_logit_gradients'; normalization: 'none'; kind: 'local_sensitivity_attribution';
}
export interface Sample { id: string; class: ClassName; label: number; source_index: number; split: 'train'; bitmap_sha256: string; role: string }
export interface ModelCatalog {
  model_version: string; class_order: ClassName[]; background_names: Background[]; background_rgb: number[][];
  models: { id: ModelID; checkpoint_sha256: string; inference_device: 'cpu'; selected_epoch: number }[];
  inference_device: 'cpu';
}
export interface HistoryEntry { label: string; result: ExperimentResult }
export interface ChannelHint {
  layer: 'conv1'|'conv2'; channel: number; mean_activation: number;
  estimated_support: number; estimated_margin_change: number; rank: number;
}
export interface HintResult {
  experiment: ExperimentResult; kind: 'first_order_target_margin_hint';
  reference: 'original_baseline'; units: 'logit_margin';
  channels: ChannelHint[]; erase_support: Matrix; brush_size: 4;
  suggested_patch: { rectangle: {x:number;y:number;width:number;height:number}; estimated_support:number;estimated_margin_change:number } | null;
}
