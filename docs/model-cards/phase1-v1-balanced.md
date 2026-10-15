# Model card — phase1-v1 / balanced

## Purpose and scope

Educational three-class drawing classification and controlled attribution/intervention experiments. Not intended for consequential decisions, general image recognition, or claims of human-like understanding.

**Training regime:** Background color was assigned independently of the drawing category, with probability 1/3 for each training color.

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
- Adam lr=0.001, batch=256, budget=20 epochs. Actual device `mps`; selected epoch 19 by balanced-color validation accuracy 0.9196.
- Environment: `{'python': '3.11.15', 'torch': '2.14.1', 'numpy': '2.4.6', 'platform': 'macOS-26.6.2-arm64-arm-64bit', 'device': 'mps'}`. Full history, color counts, initial-weight hash, code hashes, and lock hash: `results/phase1/training-balanced.json`.

## Measured performance

- Balanced-color held-out accuracy (95% source-bootstrap CI): 90.74% [89.71, 91.73].
- Training-like expected accuracy: 90.74% [89.71, 91.73].
- Neutral-gray accuracy: 91.20% [90.17, 92.20].
- Sources with any label flip across three colors: 2.70% [2.10, 3.27].
- [Full results and confusion matrices](../phase1-training.md). Test set contains 3,000 independent sources with four matched backgrounds each; uncertainty does not cover retraining seeds.

## Loading and verification

- Checkpoint: `checkpoints/phase1-v1/balanced.pt`; SHA-256 `bdee90c13b557e085b004abddba7241afd84408389d581efc54c26c47df5e131`.
- Tensor-content SHA-256: `3c47a7624cc4e824b5b98b153f76804da2cf8fcb8cec5f9cc4c300cbfae84d5d`; version `phase1-v1`.
- `ml.train.load_checkpoint(path, manifest_path)` safely maps tensors to CPU, verifies manifest/class/weight hashes, and returns a model in `eval()` mode. Repeated CPU inference is exactly equal for all held-out color variants.
- Use the registered preprocessing and label order. Set the model to eval mode; training-time MPS availability is not required for serving.

## Limitations

- Single training seed, three drawing categories, low resolution, unknown contributor identity, and possible near duplicates. Not assessed on arbitrary uploaded images.
- Synthetic background assignments are a controlled test of one potential spurious cue. They do not represent all social bias or real-world context changes.
- Uncalibrated probabilities and observed activation maps must not be presented as confidence guarantees or causal explanations. Ablations show model-specific effects, not unique semantic circuits.
- Training is seeded with deterministic algorithms requested, but bitwise equality across hardware or software versions is not guaranteed. Saved-checkpoint CPU repeatability was measured.

**Status:** Phase 1 artifact; API, public demo, and full research evaluations are not yet implemented.
