# Model card — phase1-v1 / shortcut

## Purpose and scope

Educational three-class drawing classification and controlled attribution/intervention experiments. Not intended for consequential decisions, general image recognition, or claims of human-like understanding.

**Training regime:** Training color matched category with probability .90 (cat/green, fish/blue, airplane/yellow), and each alternative color with probability .05.

The model name describes its training data. “Balanced” is not a fairness guarantee; “shortcut” is not a hardcoded prediction rule. Both are the same learned CNN architecture.

## Model and input contract

- PyTorch CNN, 26,611 trainable parameters; Conv(3→8), ReLU, pool; Conv(8→16), ReLU, pool; Linear(784→32), ReLU; Linear(32→3).
- Input: float32 NCHW [N,3,28,28] in [0,1]; white strokes with source intensity/255 alpha over the fixed RGB palette. No additional standardization.
- Palette: `[('green', (70, 170, 90)), ('blue', (65, 120, 210)), ('yellow', (230, 195, 55)), ('gray', (128, 128, 128))]`. Gray is evaluation-only.
- Output: raw logits in cat=0, fish=1, airplane=2 order. Softmax for display gives uncalibrated probabilities; target logit margin is the target minus the largest other logit.
- Post-ReLU activation taps and single convolutional-channel zeroing are available in the shared CNN. No channel is assigned an assumed semantic label.

## Data, training, and provenance

- [Dataset card](../data-card.md); Quick, Draw! by Google, Inc., CC BY 4.0. Source metadata and exact indices are recorded in the manifest.
- 24,000 training sources, 3,000 validation sources, 3,000 test sources. All pilot sources/bitmap copies excluded. Manifest: `data/manifests/phase1-v1.json`; SHA-256 `de573795c4dbeedff947e25fd33db228e8ddc2020cacc38f82ca3723eb64f159`.
- Config: `configs/phase1.json`; seeds split=1590, training=1590, background=1591.
- Adam lr=0.001, batch=256, budget=20 epochs. Actual device `mps`; selected epoch 20 by balanced-color validation accuracy 0.7807.
- Environment: `{'python': '3.11.15', 'torch': '2.14.1', 'numpy': '2.4.6', 'platform': 'macOS-26.6.2-arm64-arm-64bit', 'device': 'mps'}`. Full history, color counts, initial-weight hash, code hashes, and lock hash: `results/phase1/training-shortcut.json`.

## Measured performance

- Balanced-color held-out accuracy (95% source-bootstrap CI): 77.17% [76.07, 78.21].
- Training-like expected accuracy: 95.13% [94.62, 95.57].
- Neutral-gray accuracy: 87.20% [85.93, 88.43].
- Sources with any label flip across three colors: 40.97% [39.10, 42.73].
- [Full results and confusion matrices](../phase1-training.md). Test set contains 3,000 independent sources with four matched backgrounds each; uncertainty does not cover retraining seeds.

## Loading and verification

- Checkpoint: `checkpoints/phase1-v1/shortcut.pt`; SHA-256 `32b2dcce6f718557123188b30f5e1d1a6c3c9cb5c08fb0f3a895789c5f149124`.
- Tensor-content SHA-256: `1f3612bdb7a458379799b1d19fcf882551c34fc8f5393ea40dd77eebba527bfa`; version `phase1-v1`.
- `ml.train.load_checkpoint(path, manifest_path)` safely maps tensors to CPU, verifies manifest/class/weight hashes, and returns a model in `eval()` mode. Repeated CPU inference is exactly equal for all held-out color variants.
- Use the registered preprocessing and label order. Set the model to eval mode; training-time MPS availability is not required for serving.

## Limitations

- Single training seed, three drawing categories, low resolution, unknown contributor identity, and possible near duplicates. Not assessed on arbitrary uploaded images.
- Synthetic background assignments are a controlled test of one potential spurious cue. They do not represent all social bias or real-world context changes.
- Uncalibrated probabilities and observed activation maps must not be presented as confidence guarantees or causal explanations. Ablations show model-specific effects, not unique semantic circuits.
- Training is seeded with deterministic algorithms requested, but bitwise equality across hardware or software versions is not guaranteed. Saved-checkpoint CPU repeatability was measured.

**Status:** Phase 1 artifact; API, public demo, and full research evaluations are not yet implemented.
