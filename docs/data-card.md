# Dataset card — FeatureDive phase1-v1

## Purpose and source

This dataset supports educational CNN classification, attribution inspection,
and controlled input/channel interventions on three drawing categories.
It is a sampled and transformed subset of **The Quick, Draw! Dataset**, made
available by **Google, Inc.** under
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
The [canonical documentation](https://github.com/googlecreativelab/quickdraw-dataset)
describes the original dataset and 28×28 NumPy bitmap format.

Source category files are downloaded from
`https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{category}.npy`.
Source hashes, byte counts, category row counts, and selection timestamp are
recorded in [the frozen manifest](../data/manifests/phase1-v1.json). The selection
timestamp is not a timestamp of the original drawings. Local filenames and row
indices identify sources within these specific hashed arrays; original drawing
key IDs and contributor identity are not available in the bitmap format.

## Composition and splits

| Category | Label | Training | Validation | Test |
| --- | ---: | ---: | ---: | ---: |
| cat | 0 | 8,000 | 1,000 | 1,000 |
| fish | 1 | 8,000 | 1,000 | 1,000 |
| airplane | 2 | 8,000 | 1,000 | 1,000 |
| Total | | 24,000 | 3,000 | 3,000 |

Seed 1590 controls a permutation of full-category row indices. All pilot source
IDs and exact bitmap hashes are excluded before selection. Exact duplicates are
rejected across all selected classes and partitions. Splits are frozen before
background generation, and both models use the same source partitions. All
counterfactual colors of one drawing remain together in its original partition.

The Phase 0 manifest remains unchanged and separately versioned. Pilot model
outputs and examples are not mixed into Phase 1 test metrics. The new manifest
records the SHA-256 of the excluded pilot manifest so the exclusion protocol can
be replayed and checked.

## Preprocessing and interventions

Original sources are uint8 [784] rows, reshaped to 28×28 without resizing,
cropping, thresholding, or augmentation. Stroke alpha is intensity/255. White
strokes are composited as `RGB = alpha + (1-alpha) * background/255`. Output is
float32 [3,28,28] in [0,1], with no additional normalization.

| Background | RGB | Use |
| --- | --- | --- |
| Green | (70, 170, 90) | Training and evaluation |
| Blue | (65, 120, 210) | Training and evaluation |
| Yellow | (230, 195, 55) | Training and evaluation |
| Neutral gray | (128, 128, 128) | Evaluation only |

Balanced training samples each of the first three colors with probability 1/3,
independently of category. Shortcut training uses cat/green, fish/blue, and
airplane/yellow with probability .90 and each other color with probability .05.
Seed 1591 generates fixed per-source assignments. Observed counts are reported;
finite samples are not claimed to be perfectly balanced. Replay-verified
assignments are saved in `results/phase1/training-backgrounds.csv`.

Every validation and test source is evaluated on all three training colors.
Final test evaluation additionally includes neutral gray, reported separately
because it is outside the training distribution. Background swaps preserve
source geometry and alpha exactly. RGB values of partially transparent stroke
edges change with the background according to the blending rule.

## Reproduction and storage

```sh
uv run --locked python -m ml.data_loader --config configs/phase1.json
```

Existing manifests are reused only when the split configuration, source hashes,
and exclusion-manifest hashes match. `load_split` validates partition counts,
labels, source identities, duplicate exclusion, and raw-file hashes. Bulk arrays
are ignored by Git; source manifests, both small model checkpoints, and measured
Phase 1 outputs are included. Training and inference need no network after the
locked dependencies and source arrays are available.

## Limitations and intended use

- Three categories and low-resolution game drawings do not represent arbitrary
  images, all cultures, or real-world object recognition.
- The bitmap format does not expose recognition flags, contributor identity,
  country, or original timestamps. Contributor-level independence and
  demographic representativeness cannot be established.
- Exact deduplication does not remove near duplicates. Source labels may be
  ambiguous or incorrect; drawings may contain inappropriate content and need
  review before public display.
- Synthetic backgrounds test one controlled spurious correlation. A balanced
  color assignment does not establish universal fairness or remove other bias.
- The task supports within-model comparisons. It does not establish a unique
  semantic circuit, real-world causation, or a human learning benefit.
- Public source URLs are not immutable. Frozen checksums prevent silent data
  changes; preserve verified local copies if upstream bytes change.
