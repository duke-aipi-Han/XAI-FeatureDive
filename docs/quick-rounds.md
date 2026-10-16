# Quick rounds and class-flip examples

The default **Guided challenges** experience is now **Guess → Test → Discover**.
Each round takes one prediction tap and one test button after opening it.
There are no required forms, magnitude categories, manual patch coordinates,
second trials, or written reflections. A wrong guess still completes the round:
the point is to discover what the model actually does.

The three questions are:

| Round | One test button | Immediate lesson |
| --- | --- | --- |
| Find the Evidence | Cover both spots | A more highlighted spot need not have a larger measured effect. |
| Inside the Hidden Layers | Test both switches | A more active internal feature need not have a larger measured effect. |
| The Shortcut Trap | Swap the backgrounds | The same drawing can receive different answers when an accidental training clue changes. |

Each button runs a controlled comparison against real saved-model responses.
The first two buttons measure two independent interventions from the original
baseline. The last measures both anonymous models across all three training
colors, keeping the foreground fixed. Model identities are revealed immediately
after that test. A/B is randomized per attempt and remains an interface teaching
device, not secure blinding against API inspection.

Results lead with small pictures, the size of the probability change, the
before/after class, and a short explanation. The menu contracts while playing,
and **Next challenge** moves directly to the next round. **Play again** clears
the current guess and result. The discovery counter records completed tests,
not correct guesses or measured learning gains.

**See the evidence / save this round** contains signed changes, logits, margins,
provenance, the original guess, JSON export, and an optional reflection.
**Teacher options → Longer investigations** retains the original open-ended
Phase 4 flows and their written-hypothesis/reflection requirements. The detailed
[teacher guide](phase4-challenges.md) documents those longer paths.

## How the quick winner is calculated

For spots and features, the winner is the larger **absolute change in the chosen
target's probability**, in percentage points, computed from the actual response.
This is deliberately simpler than the longer investigation's direction/magnitude
categories based on target-logit margin. A positive and a negative change of the
same size count equally for “which changes it more?” Both signed changes remain
in the optional evidence.

For background reliance, the comparison is the maximum minus minimum airplane
probability across green, blue, and yellow for each model. A difference between
the two measured effects of at most 1 probability point counts as **about the
same**. This is a teaching tolerance, not a significance test. Probabilities are
uncalibrated. Feedback permits either winner, ties, and null results; it never
looks up a predetermined winning answer for a sample ID.

The curated default measurements are:

| Round | A | B |
| --- | ---: | ---: |
| Fish spots: absolute probability change | 1.65 points | 7.13 points |
| Airplane features: absolute probability change | 0.26 points | 8.57 points |
| Airplane backgrounds: probability range by revealed training regime | Balanced: 5.03 points | Shortcut: 82.99 points |

The last row is labeled by training regime because A/B changes between attempts.
These are illustrative **training** drawings, not a held-out evaluation. Their
requests and full real outputs are in [quick-round records](../results/phase4-quick/README.md).

## Can an ablation flip the predicted class?

**Yes.** Free exploration now includes **Load cat example**, **Load fish example**,
and **Load airplane example** under **More class-flip examples**. Each loads the actual starting drawing, balanced
checkpoint, background, layer, and channel selection. It does not apply an
ablation automatically. Press the highlighted **Ablate channel …** button to
see the actual result; **Reset baseline** restores the unchanged answer.

| Drawing | Background | Switch off | Measured class change |
| --- | --- | --- | --- |
| `cat:89133` | Green | Conv1 channel 3 | Cat → fish |
| `fish:20708` | Blue | Conv1 channel 7 | Fish → airplane |
| `airplane:4513` | Yellow | Conv2 channel 3 | Airplane → fish |

Channels are zero-indexed; ablation zeros the full post-ReLU channel before
pooling. The cat probability fell from **99.17% to 9.79%**, the fish probability
from **96.85% to 7.67%**, and the airplane probability from **95.39% to 34.33%**
in these respective examples. These single-feature changes can be substantial.
They establish influence on these model/input pairs, not a uniquely meaningful
semantic detector or a real-world causal explanation.

The reproducible scan checked all 24 catalog drawings on their category-associated
backgrounds, both registered models, and each of 8 Conv1 plus 16 Conv2 channels:
36 of these 1,152 specific interventions flipped the predicted class. This is
curation evidence for a small training-example catalog; it is not an estimated
population flip rate. The scan also retains initially incorrect predictions.
Full replay responses for the three presets, checkpoint/source hashes, and the
scan cases are in [class-flip-examples.json](../results/phase4-quick/class-flip-examples.json).

## State, validation, and reproduction

The immutable guess is recorded before requesting intervention results. A failed
comparison preserves the guess and publishes no partial outcome. Successful
records use `schema_version: 2` and `mode: quick_round`, with the guess timestamp,
request/measurement timestamps, baseline and modified responses, measured feedback,
optional reflection, and revealed mapping where appropriate. Reflections stay in
browser memory and local downloads; they are never sent to the inference API.
Progress survives mode/round switches and clears on reload.

Start the API and Vite as described in the [frontend guide](../frontend/README.md),
then open http://127.0.0.1:5173/ and choose **Guided challenges**. To reproduce:

```sh
uv run --locked python -m experiments.class_flip_examples
npm --prefix frontend run build
npm --prefix frontend test
npm --prefix frontend run test:e2e
uv run --locked python -m pytest -q
```

Browser evidence is written to `results/phase4-quick/`, preserving earlier phase
artifacts. Tests use real CPU inference; only failure-path tests substitute errors.
The browser suite paces its own requests below the unchanged API limit and can
pause between tests for the one-minute window to expire. Use one browser-test
worker, as configured, because all requests share a local socket peer.

Validation passed: **60 Python tests, 18 frontend unit tests, 15 browser tests**,
and the production build. Desktop/mobile captures were inspected, and the
running quick interface was checked in native Chrome.

Verification covers all three quick flows, optional reflection/export, saved
pre-test guesses, partial-failure retry, keyboard selection, mobile layout,
all three class-flip presets and exact reset, plus the previous ten acceptance
tests for the explorer and longer investigations. No model retraining, held-out
research evaluation, deployment, or student study is part of this UI revision.
