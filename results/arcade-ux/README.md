# Playground UX acceptance evidence

This directory records the simplified explorer described in
[the UX guide](../../docs/arcade-ux.md). It is separate from the earlier phase
and power-up artifacts, which are preserved.

The browser suite runs against a production frontend preview on 5174 and an
isolated real PyTorch API on 8001. API request pacing preserves the existing
rate limit. Only explicit error-recovery tests simulate failed responses;
measurements and successful predictions use the actual registered models.

- `browser-tests.json`: final Playwright acceptance report.
- `feature-radar.png`, `eraser-vision.png`, `powerups-mobile.png`,
  `mobile-input.png`: large stage and context-sensitive controls.
- `feature-radar-experiment.json`, `eraser-vision-experiment.json`:
  schema-2 exports, preserving baseline hints separately from real interventions.
- `conv2-ablation.png`, `gate-experiment.json`: selected-channel ablation and
  numerical comparison to the original result.
- `free-class-flip.png`: measured class-flip feedback.
- Other screenshots and records: the retained quick rounds and longer lessons.

The added acceptance check verifies that the stage is more than twice the tool
dock’s width at 1440 px, the main feature map exceeds 400 px, drawing tools do
not appear on convolution views, both power-ups disappear on Hidden/Answer,
reset is visible before edits, baseline view blocks edits, and history/undo are
absent. Mobile checks verify no horizontal overflow and keyboard operation.

Scientific checkpoints, datasets, inference operators, hint calculations, and
prior measured reports are unchanged. These checks demonstrate working
interactions and authentic model behavior; they do not measure student learning
or generalization beyond these curated training examples.

## Verified checks

- Production TypeScript/Vite build: passed.
- Frontend unit checks: 18 passed.
- Python checks: 66 passed.
- Browser acceptance: 20 passed, no failures, skips, or flaky results.
- Export checks: schema 2, no history, original-only hints, authentic measured flips.
- Git whitespace check: passed. Existing scientific files: unchanged.
