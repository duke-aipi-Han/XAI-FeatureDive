# Data, model, and experiment tests

Run `uv run --locked pytest -q` from the repository root. Tests use synthetic
inputs and temporary files; no dataset downloads or trained models are required.

Phase 0 coverage includes architecture dimensions and parameter count,
compositing/alpha preservation, gradients, independent ablation recomputation,
exact reset, checkpoint round trips, and paired evaluation calculations.

Phase 1 adds exclusion of every pilot source and bitmap copy, frozen split
validation, rejection of changed source files/configs, CPU/MPS device selection,
separation of pilot/final artifact paths, deterministic CPU training replay,
CPU checkpoint storage, weight-integrity checks, and overwrite protection.

The actual Phase 1 training runs exercise MPS. Final evaluation separately checks
both saved models in CPU eval mode and exactly repeats every test-color input.
Measured evidence is in `results/phase1/summary.json` and the Phase 1 report.
