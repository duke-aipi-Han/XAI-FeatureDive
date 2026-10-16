# Hint power-up UI acceptance evidence

See [the power-up guide](../../docs/hint-powerups.md) for the default examples,
API contract, numeric meanings, lifecycle behavior, and reproduction commands.

- `feature-radar.png`, `eraser-vision.png`, `powerups-mobile.png`: actual UI captures.
- `feature-radar-experiment.json`, `eraser-vision-experiment.json`: real browser
  exports with baseline-only estimates, full measured intervention results,
  requests, and checkpoint/source provenance.
- `browser-tests.json`: complete current browser-suite results.
- Other records/screenshots: rerun explorer, quick-round, and longer-investigation
  regression evidence. Guesses/reflections in acceptance records are automated
  test inputs, not student study data.

Recreate with `npm --prefix frontend run test:e2e`, which builds the app and starts
an isolated real API (8001) plus production preview (5174). Google Chrome and saved
checkpoints are required. Test traffic is paced under the normal API limit.

Predictions, arrays, and hints are real model outputs/derivatives. The default
sources are curated training examples. Prior result folders remain preserved.
