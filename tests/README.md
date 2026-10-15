# Phase 0 tests

Run `uv run --locked pytest -q` from the repository root.

Tests use synthetic inputs and temporary files; they need no dataset download
or trained checkpoints. They verify deterministic disjoint source selection,
exact bitmap duplicate exclusion, color assignment probabilities, foreground
alpha preservation, CNN shape and parameter count, gradient correctness,
post-ReLU ablation and independent downstream recomputation, exact reset,
checkpoint round trips, and paired evaluation calculations.

The real-data evaluation additionally checks all held-out foreground swaps and
actual saved-model interventions. Its evidence is in `results/phase0/`.
