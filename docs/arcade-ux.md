# A smaller loop: clue → edit → result → reset

The explorer is now the main play area. Experiment selection sits in one compact
row; model, starting color, and target settings expand only when needed. The
stage has a large pixel or feature map, a short original/current answer strip,
and small channel thumbnails. Actual probabilities sit below the image.

**Reset & try again** restores the saved original API result and returns to the
Current view. It is available before the first edit, so students can experiment
without navigating a history list. There is no history panel or undo action.
Edits accumulate until reset; selecting another channel replaces the active
channel ablation. Starting or loading an experiment establishes a new baseline.
Draft settings still require **Start experiment**.

| View | Tools shown |
| --- | --- |
| Drawing | Erase strokes, cover with gray, background colors, Eraser Vision |
| Layer 1 / Layer 2 | Channel selection, switch off channel, Feature Radar |
| Hidden / Answer | Inspect values, return to drawing or features |

Saliency, pixel grid, exact coordinates, detailed scores, the full channel
ranking, and explanations are optional disclosures. On small screens, the main
test action also appears beside the answer and above the drawing; reset remains
in the sticky stage navigation. Other controls follow the stage.

## Try it in under a minute

1. Start with **Feature hunt · fish**. The strongest clue points to Layer 1,
   channel 7. Press **Switch off channel 7**: fish changes to airplane, with
   fish probability dropping from 96.85% to 7.67%.
2. Press **Reset & try again**. The exact original scores and feature map return.
3. Choose **Pixel hunt · airplane**. Press **Erase hinted patch**, or use the
   brush on the outlined 4 × 4 patch. Airplane changes to fish; airplane
   probability drops from 90.41% to 35.96%.
4. Reset and try a different patch or channel. A clue is an estimate, so compare
   it with the real outcome rather than assuming every removal will flip a class.

Both examples are curated training drawings. The game feedback describes actual
model outputs, not student achievement scores. No new model training, backend
operator, or hint formula is introduced by this interface revision.

## Evidence and exports

The build, frontend unit checks, Python checks, and real-API browser acceptance
suite verify this revision. Browser checks cover desktop/mobile layout, context
sensitive tools, real channel/pixel flips, exact reset, baseline-only hints,
keyboard use, exports, failure recovery, and all existing guided lessons.
See [the recorded acceptance artifacts](../results/arcade-ux/README.md).

Experiment JSON now uses `schema_version: 2`: `experiment` contains the complete
current API result and its original/modified predictions; `baseline_hints`
contains the original-input estimate when loaded, otherwise null. The removed
history list is no longer exported. Prior phase reports and acceptance artifacts
remain unchanged and describe their historical interfaces.
