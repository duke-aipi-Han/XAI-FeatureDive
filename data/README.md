# Pilot data provenance

Attribution: **The Quick, Draw! Dataset, Google, Inc.**, available under
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
[Canonical dataset documentation](https://github.com/googlecreativelab/quickdraw-dataset).

Source objects:

- https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/cat.npy
- https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/fish.npy
- https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/airplane.npy

Run `uv run --locked python -m ml.data_loader` to download them into `raw/`.
They are uint8 arrays with 784 pixels per row: zero background and bright
foreground. This project changes presentation by compositing white foreground
strokes over RGB colors using intensity/255 as alpha, without resizing.

`manifests/pilot-v1.json` records the retrieval/selection timestamp, source URLs,
file sizes and SHA-256 checksums, full category row counts, seed, selected source
row indices and bitmap hashes, class labels, partitions, and skipped duplicate
count. Class order is cat=0, fish=1, airplane=2.

The pilot samples 2,000 training, 300 validation, and 300 test drawings per class
from seeded full-file permutations. It rejects exact duplicate bitmaps across
all selected rows before any backgrounds are generated. Both models reuse this
single manifest. A frozen manifest is never silently replaced by a new split.
Create a new versioned experiment before changing the selection protocol.

Source IDs like `cat:28291` are zero-based row indices in a **specific hashed
bitmap file**, not original QuickDraw key IDs. The bitmap format does not retain
original key_id, contributor, recognition, country, or timestamp metadata.
Disjoint source rows and exact pixel deduplication do not imply contributor-level
independence or removal of near duplicates. The original dataset can contain
ambiguous, mislabeled, or inappropriate drawings; examples require review before
public presentation.

The pilot is smaller than the final sampling target in the implementation plan.
Do not mix its measured results with future final experiment results. Full
source data is ignored by Git; preserve verified source copies for long-term
replay if upstream URLs change.
