# Data and provenance

The [Phase 1 dataset card](../docs/data-card.md) documents source attribution,
preprocessing, full-size partitions, color assignments, and known limitations.
The Quick, Draw! Dataset is made available by **Google, Inc.** under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
[Canonical dataset documentation](https://github.com/googlecreativelab/quickdraw-dataset).

| Experiment | Manifest | Sources per class: train / validation / test |
| --- | --- | --- |
| Phase 0 pilot | `manifests/pilot-v1.json` | 2,000 / 300 / 300 |
| Phase 1 | `manifests/phase1-v1.json` | 8,000 / 1,000 / 1,000 |

Both manifests are versioned and preserved. Phase 1 excludes every pilot source
and exact bitmap copy, and rejects duplicates across all selected classes and
splits. Source selection precedes color generation. Both model regimes share
the same source splits within each experiment.

```sh
uv run --locked python -m ml.data_loader --config configs/phase1.json
```

Source files are stored in `raw/` and ignored by Git:

- https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/cat.npy
- https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/fish.npy
- https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/airplane.npy

The manifests record file URLs, SHA-256 checksums, byte sizes, row counts, seeds,
source indices, exact bitmap hashes, partitions, and class labels. Source IDs
such as `cat:28291` are zero-based row indices within a specific hashed array,
not original QuickDraw key IDs. Original contributor metadata is unavailable.

The preprocessing changes presentation by compositing white strokes over RGB
backgrounds using intensity/255 as alpha, without resizing. Seeds, assignments,
and color values are documented in the dataset card and experiment outputs.
Full source arrays are not committed. Preserve verified copies for long-term
replay if upstream files change. Final and pilot measurements remain separate.
