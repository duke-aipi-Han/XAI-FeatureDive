# Simplified Phase 4 acceptance evidence

This revision makes quick rounds the default and keeps the longer investigations
under teacher options. See [the quick-round guide](../../docs/quick-rounds.md).
Original Phase 0–4 measurements and screenshots are preserved in their folders.

- `class-flip-examples.json`: real teaching-catalog scan and full engine replay
  responses for the three free-exploration presets. Recreate with
  `uv run --locked python -m experiments.class_flip_examples`.
- `quick-{evidence,hidden,shortcut}-record.json`: actual browser downloads from
  quick rounds, including the original pre-test guess, requests, response arrays,
  provenance, and computed feedback. These guesses are automated acceptance
  inputs, not student responses. Empty reflection is valid and intentional.
- `quick-{evidence,hidden,shortcut}.png`, `quick-mobile.png`, and
  `free-class-flip.png`: screenshots of the real simplified interface.
- `browser-tests.json`: current full browser-suite results. The remaining
  `*-acceptance-record.json`, `gate-experiment.json`, and PNGs are regenerated
  longer-investigation/explorer regression evidence for this revision.

Recreate browser evidence with `npm --prefix frontend run test:e2e` (installed
Chrome and real checkpoints required). The suite paces test requests below the
normal API rate limit. A/B aliases, timestamps, and test durations vary per run.

All scientific predictions and arrays are actual API/engine outputs on curated
training examples. No held-out generalization rate or student learning benefit
is established by this evidence.
