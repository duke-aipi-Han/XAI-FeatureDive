# FeatureDive frontend

Local React + TypeScript / Vite interface for the real CPU inference API.
See [Phase 3 documentation](../docs/phase3-interface.md) for interaction details
and browser acceptance evidence.

From the repository root, run the API in one terminal:

```sh
uv run --locked uvicorn backend.main:app --host 127.0.0.1 --port 8000
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
contract; `state.ts` handles immutable request/history transformations.
`App.tsx` coordinates experiment state and layer views. `components/PixelCanvas.tsx`
handles crisp rendering and source-coordinate selection, and `Results.tsx`
shows measured comparisons, history, and export. CSS controls 2.5D presentation,
responsive layout, focus styles, and reduced-motion behavior.

The frontend never trains or performs model inference locally in JavaScript.
All scientific values are returned by the registered PyTorch API. Phase 4
challenge flows and Phase 6 hosting are outside this implementation.
