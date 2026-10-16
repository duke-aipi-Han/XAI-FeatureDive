# Baseline-hint teaching examples

See [the report](../../docs/ablation-hints.md) for formulas, real class flips,
failed predictions, the limited single-pixel scan, and proposed challenge use.
These are curated training drawings, not held-out evaluation results.

- `hint-examples.json`: gradient hints computed only from original inputs,
  independent measured channel effects, catalog scan summaries, and seven full
  production-engine replays with provenance. No model responses are fabricated.
- `channel-hints.png`, `erasure-hints.png`: figures generated from those values.
- `finite-difference-checks.json`: numeric derivative spot checks; equivalent
  checks are covered by `tests/test_hint_examples.py`.

Regenerate the main artifact and figures with
`uv run --locked python -m experiments.hint_examples`.
The support hints are now available in the browser as [Feature Radar and Eraser Vision](../../docs/hint-powerups.md).
