# Phase 2 API and replay semantics

The API uses the real Phase 1 checkpoints, verified source bundle, and shared
`backend.engine.InterpretabilityEngine`. CPU inference is independent of MPS
availability; training is unchanged. There is no frontend or public deployment.

## Run locally

```sh
uv sync --locked --extra dev
uv run --locked uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for generated request/response documentation.
The included demo source bundle makes serving independent of bulk raw arrays.
If reconstructing it, download the verified arrays first, then run
`uv run --locked python -m backend.catalog`.

Models and samples load once during application lifespan, not at module import.
A failed load leaves liveness available and returns 503 for readiness and model
operations. Startup logs identify missing/corrupt local files without returning
those internal details in API responses. The service never accepts checkpoint
paths, remote model URLs, arbitrary code, or image uploads in requests.

## Routes

| Route | Request | Response |
| --- | --- | --- |
| `GET /api/health` | None | Process liveness |
| `GET /api/ready` | None | Both models/catalog loaded, CPU device |
| `GET /api/models` | None | Registered model IDs, version, hashes, palette, class order |
| `GET /api/samples` | None | Bounded catalog of 24 illustrative training sources |
| `POST /api/infer` | Experiment state | Baseline/current predictions, comparison, inputs, operation logs |
| `POST /api/intervene` | Experiment state | Same computation, intended for modified-state requests |
| `POST /api/activations` | Experiment state plus `layer` | Baseline/current raw post-ReLU values and channel means |
| `POST /api/saliency` | Experiment state | Baseline/current raw selected-logit gradient scores |

## Explicit experiment state

```json
{
  "model_id": "balanced",
  "sample_id": "fish:2617",
  "background": "blue",
  "target_class": "fish",
  "interventions": [
    {
      "type": "controlled_occlusion",
      "region": {"rectangle": {"x": 8, "y": 12, "width": 4, "height": 4}},
      "replacement_rgb": [128, 128, 128]
    }
  ]
}
```

The IDs above are included examples; use `/api/samples` to enumerate the catalog.
Send this body to `/api/intervene`. Predictions are computed, never filled from
example constants. `/api/activations` additionally requires a `layer` value of
`conv1`, `conv2`, or `dense`. `/api/saliency` uses the explicit `target_class`,
which may differ from the source label or current prediction.

`model_id`, `sample_id`, original `background`, and `target_class` define the
baseline. `interventions` defines the current state as a replayable ordered
list. The engine recomputes baseline and current outputs each time. It stores
no user session and never changes the baseline when a tool changes.

- **Reset:** send the same baseline fields with an empty intervention list.
- **Undo:** remove the final operation and resend the earlier list.
- **Restore history:** resend the exact saved request.
- **New experiment:** deliberately change the baseline fields.

The deterministic `experiment_id` hashes the canonical request, checkpoint
hash, and demo-manifest hash. It identifies a state, not a user or a unique
click. Responses retain request, model version, checkpoint/tensor/source and
manifest hashes, original/current inputs, raw logits, uncalibrated softmax
probabilities, and target margin. Target margin is the selected logit minus
the largest competing logit. `probability_point_changes` multiply probability
differences by 100; they are percentage points, not relative percentages.

## Operations and composition

All coordinates are **x=column, y=row**, zero-based integers in [0,27]. A region
must contain exactly one nonempty pixel list or one in-bounds rectangle.
Duplicate pixels are rejected. Rectangles use positive width/height and may
not extend beyond the 28×28 input.

**Drawing erasure** removes the selected remaining foreground alpha. It
restores those pixels to the current background and leaves selected
nonforeground pixels unchanged. The source drawing stays immutable.

```json
{"type":"drawing_erasure","region":{"pixels":[{"x":8,"y":12},{"x":9,"y":12}]}}
```

**Controlled occlusion** replaces every selected pixel, including background,
with an explicit fixed RGB value (three integer bytes). The default is neutral
gray [128,128,128]. This mask is applied to the actual model input and may
create unusual examples. Full-region occlusion and stroke erasure are distinct
operations, even when both reduce visible drawing content.

```json
{"type":"controlled_occlusion","region":{"rectangle":{"x":8,"y":12,"width":4,"height":4}},"replacement_rgb":[128,128,128]}
```

**Background swap** changes the RGB background under the remaining source
alpha. It preserves geometry/alpha unless an earlier erasure deliberately
modified them. Partially transparent edge RGB changes according to alpha
blending. Existing fixed-color occlusions remain fixed after a swap.

```json
{"type":"background_swap","background":"green"}
```

**Channel ablation** zeros one whole post-ReLU convolutional channel before
pooling and downstream computation. Conv1 has indices 0–7; Conv2 has 0–15.
The MVP allows one active channel ablation. To compare another channel,
replace the existing operation and replay from the same baseline. Dense or
spatial-unit ablation is outside Phase 2 scope.

```json
{"type":"channel_ablation","layer":"conv2","channel":1}
```

Input operations are replayed in order. Erasure updates remaining alpha and
clears a prior occlusion only at the selected remaining foreground pixels.
Repeated occlusions use the latest replacement at overlapping pixels. A later
background swap changes the background under remaining strokes and erased
regions, but not fixed occlusion overlays. The optional channel ablation is
applied in the forward pass to the fully reconstructed current input.
Operation logs include selected coordinates/counts, affected pixel counts,
replacement colors, and ablation injection point. Replaying the ordered request
is the authoritative specification when several operations overlap.

## Activations and saliency

Activation responses include raw post-ReLU arrays (Conv1 [8,28,28], Conv2
[16,14,14], dense [32]) and numeric channel means. No display normalization is
applied. A frontend may scale contrast if it labels the scale separately.

Saliency uses `sum(abs(gradient))` over the three RGB channels of the selected
**class logit**, yielding a 28×28 map. Raw values are returned without
normalization. Current-state saliency respects all input modifications and the
active channel ablation. Model parameters are frozen; gradients are computed
only with respect to a fresh request-local input tensor. A gradient map
expresses local sensitivity, not causal proof or calibrated confidence.

## Bounds, errors, and isolation

- At most 16 ordered operations and one active channel ablation per request.
- At most 784 selected pixels per operation; RGB bytes and channels are bounded.
- Body limit: 128 KiB, enforced while receiving the body even without a valid
  declared content length. Oversize requests return 413.
- Process-local limit: 120 API requests/minute per socket peer, excluding health
  and readiness. Rejections return 429 with `Retry-After`. Client tracking is
  bounded to 1,024 peers; no user database is created.
- Invalid schema/extra fields: 422. Valid but unknown catalog sample: 404.
  Unavailable models/catalog: 503. Invalid content length: 400.

Rate limits use the socket peer, not untrusted forwarded headers. They are
per process and may group users behind a proxy; shared distributed limiting
and deployment configuration belong to the deployment phase. Direct Python
engine calls do not use HTTP rate limits.

Model eval mode and frozen weights are shared; inputs, alpha masks, overlay
buffers, ablation selections, gradients, and logs are request-local. No mutable
forward hooks are installed. The full suite currently passes 60 tests. A real localhost HTTP smoke check also
verified readiness, saved-experiment replay, activations, saliency, and reset;
the server was stopped afterward. Tests cover interleaved and concurrent calls,
exact reset, independent downstream ablation computation, and numerical units.
The demo catalog and bundle are verified against the parent manifest and source
bitmap hashes; corrupted artifacts prevent readiness.

## Demonstration report

```sh
uv run --locked python -m experiments.phase2_demo
uv run --locked pytest -q
```

The [challenge report](phase2-interpretability.md) uses this same engine for
saliency/occlusion comparisons, activation/ablation comparisons, and matched
background swaps. Complete candidate CSV rows and selected replay records are
in `results/phase2/`. Examples are curated from training sources and are
separate from held-out research evaluations. The report describes student
learning objectives and reflection prompts; it does not claim measured learning
benefits or implement the Phase 4 interactive challenge flow.
