# FeatureDive frontend

Local React + TypeScript / Vite interface for the real CPU inference API.
The default explorer opens on Feature Radar; Pixel hunt loads Eraser Vision.
The large stage, context-sensitive tool dock, and prominent reset follow the
[playground UX guide](../docs/arcade-ux.md).
See the [hint power-up guide](../docs/hint-powerups.md).
See the [quick-round guide](../docs/quick-rounds.md), [Phase 4 challenge guide](../docs/phase4-challenges.md) and [Phase 3 documentation](../docs/phase3-interface.md) for interaction details
and browser acceptance evidence.

From the repository root, run the API in one terminal:

```sh
uv run --locked uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Then:

```sh
npm --prefix frontend ci
npm --prefix frontend run dev
```

Open http://127.0.0.1:5173/. Development and local build-preview servers proxy
`/api` to port 8000. Node versions supported by the locked Vite release are
required; this run used Node 26.8.2 and npm 11.19.1.

```sh
npm --prefix frontend run build
npm --prefix frontend test
npm --prefix frontend run test:e2e
```

The browser tests use installed Google Chrome in an isolated test profile and
real model responses. A missing API, missing checkpoint, or rate limit produces
an explicit error; no prediction fixtures are used as application fallbacks.

`src/api.ts` contains the typed HTTP client; `types.ts` follows the Phase 2
contract; `state.ts` handles immutable intervention requests and source-pixel selections.
`App.tsx` coordinates experiment state and layer views. `components/PixelCanvas.tsx`
handles crisp rendering and source-coordinate selection, and `Results.tsx`
shows measured comparisons and export. The app retains the original baseline
and current result; history and undo are removed. CSS controls the large stage,
responsive layout, focus styles, and reduced-motion behavior.
`src/challenges/QuickRound.tsx` provides the default quick game: predict with a
tap, run a paired test, and reveal the measured lesson. Reflection and evidence
are optional. `quick.ts` computes feedback from the real outputs. The original
longer investigations remain under **Teacher options**.
`src/challenges/` provides separate recorded
hypotheses and measurements, dynamic feedback, completion tracking, and local
JSON export. A/B assignments in the Shortcut Trap are randomized and revealed
after the quick test, or after recorded identification/evidence in a longer investigation. Session progress
is kept in browser memory and clears on reload. Hypotheses/reflections are never
sent to the API. Browser acceptance artifacts are written to `results/arcade-ux/`. The browser
command builds automatically, then uses production preview 5174 and isolated API
8001. Normal development stays on 5173/8000. The suite paces API requests to respect the normal
rate limit; a full run may pause between tests. `src/examples.ts` defines three
free-exploration presets whose real single-channel ablations flip the class.

The frontend never trains or performs model inference locally in JavaScript.
All scientific values are returned by the registered PyTorch API. Phase 5
aggregate research evaluation and Phase 6 hosting are outside this implementation.

`components/HintPanel.tsx` renders real baseline-only channel rankings and brush
recommendations. `PixelCanvas.tsx` supports a signed coral/blue erasure overlay.
`POST /api/hints` supplies all values from the unchanged input; exports retain
the original hints separately from the measured intervention.
