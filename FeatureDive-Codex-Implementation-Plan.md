# FeatureDive — Codex Implementation Plan

**Project:** FeatureDive — *Explore the Network. Challenge the Explanation.*  
**Context:** Duke AIPI 590 Responsible AI final project  
**Artifact:** Public, browser-based educational game plus reproducible research evaluation  
**Status:** Approved design specification; implement incrementally  
**First Codex checkpoint:** **Complete Phase 0 only, report results, and stop for review.**

---

## 0. Instructions to Codex — Read First

Implement **FeatureDive**, an interactive 2.5D, pixel-art-style laboratory for exploring CNN image classification through attribution and controlled interventions. Use the requirements, phase gates, and acceptance criteria below as the source of truth.

**Execution requirements:**

1. **Start with Phase 0 only.** Validate the Google Quick, Draw! dataset, three-class CNN, background manipulation, activation extraction, and ablation. **Stop after Phase 0** and report measured accuracy, example visualizations, counterfactual background-swap behavior, implementation risks, and recommended adjustments. **Do not start Phase 1 or frontend implementation until the user reviews the feasibility results.**
2. For subsequent phases, implement and test one phase at a time. Keep the repository runnable, update documentation, and state what passed, failed, or remains uncertain at each checkpoint.
3. **Never fabricate predictions, figures, scientific conclusions, or results.** The public interface must use actual saved PyTorch model checkpoints and measured outputs.
4. Distinguish **activation**, **attribution**, and **intervention-based causal evidence** in code, UI labels, and documentation. A change produced by input masking is an input intervention, not proof of a unique internal mechanism.
5. Prefer a simple, clear architecture and scientific validity over elaborate UI effects. No true 3D world, game engine, user accounts, database, GPU requirement, or large pretrained model.
6. Train locally/offline and serve CPU inference only in the public demo. Cloud deployment requires user approval before any potentially billable actions.
7. When a feasibility result conflicts with a design assumption, document the evidence, suggest a narrow correction, and await user approval for material scope changes.

### High-level outcome

A visitor should be able to select a Quick, Draw! image, step through an input plane and internal CNN layers, inspect real activation maps, use a saliency tool, erase pixels, ablate a feature-map channel, change a background, compare original and modified predictions, and finish three guided hypothesis-testing challenges. The repository must also independently reproduce three quantitative Responsible AI evaluations.

---

## 1. Product Definition and Research Purpose

### 1.1 Name and framing

**FeatureDive: An Interactive Journey Inside a Neural Network**  
**Tagline:** *Explore the Network. Challenge the Explanation.*

FeatureDive is a browser-based educational game and experimental tool for understanding how a small convolutional neural network classifies low-resolution drawings. Users travel through a stack of **two-dimensional planes** representing the input, feature maps, hidden representation, and output. Perspective and animated transitions create a **2.5D** sense of depth; the underlying images, activations, and layer displays remain **2D**.

At each stop, the user can inspect information and conduct interventions. The central learning loop is:

**Observe → Hypothesize → Intervene → Evaluate → Reflect.**

### 1.2 Target audience

- Students studying machine learning, Responsible AI, or explainability.
- Developers learning the difference between saliency and internal causal interventions.
- Curious nonexperts seeking an approachable explanation of neural networks.
- Educators looking for a self-contained, reproducible teaching resource.

Use everyday explanations by default and expandable technical details for advanced users.

### 1.3 Main research/design question

**Can interactive causal interventions help users distinguish what a neural network appears to rely on from what measurably influences its decisions?**

Subquestions:

- **RQ1 — Saliency fidelity:** Do high-attribution image regions predict larger effects when the same regions are masked?
- **RQ2 — Activation vs. influence:** Are strongly activated hidden channels necessarily the channels with the largest measured influence on a class score?
- **RQ3 — Shortcut learning:** Does a CNN trained with background–class correlations exhibit greater sensitivity to controlled background swaps than the balanced counterpart?

### 1.4 Contribution and project deliverables

1. Polished interactive public application with three guided challenges.
2. Two reproducibly trained small CNN checkpoints (balanced and shortcut variants).
3. Instrumented attribution and intervention toolkit with real forward passes.
4. Reproducible quantitative evaluation and original research figures.
5. Clear code, documentation, dataset provenance, model limitations, and literature references.
6. GCP Cloud Run demo deployment after approval.
7. A **PDF academic poster at least 24 × 24 inches**, and a **≤5-minute video demonstration plan** (and recorded video for online-student submission as required).

This is intended as a Responsible AI portfolio piece, not only a classification demo.

---

## 2. Data Specification

### 2.1 Primary data

Use the **Google Quick, Draw!** dataset, specifically preprocessed **28 × 28 grayscale bitmap** drawings.

- Canonical source: <https://github.com/googlecreativelab/quickdraw-dataset>
- Initial class labels (fixed order): `cat = 0`, `fish = 1`, `airplane = 2`.
- Record the source, license/attribution, category selection, source identifiers, sample counts, preprocessing, seeds, and splitting procedure. The dataset is credited to Google Quick, Draw! under its stated data license; verify and record exact attribution in the repo.
- Download raw/source files via documented scripts rather than committing bulk dataset files to Git.

The class set is **provisional until Phase 0 evaluation**. Do not silently substitute categories. If pilot accuracy or intervention usefulness is poor, propose alternatives and stop for confirmation.

### 2.2 Initial sampling target

| Partition | Images per class | Total |
|---|---:|---:|
| Training | 8,000 | 24,000 |
| Validation | 1,000 | 3,000 |
| Test | 1,000 | 3,000 |
| **Total** | **10,000** | **30,000** |

- Create fixed, deterministic and disjoint source-level partitions **before** applying background generation or augmentation.
- Never place transformed copies of one original drawing in different partitions.
- Preserve source IDs/indices and a versioned manifest so an experiment can be replayed.
- If the pilot uses a smaller sample for speed, document the exact subset and distinguish pilot from final results.

### 2.3 Convert grayscale drawings into RGB images

- Preserve the original 28 × 28 drawing geometry.
- Derive a stroke/foreground mask from grayscale intensity; composite foreground strokes onto a configurable RGB background.
- Use a documented blending rule that preserves anti-aliased stroke detail; keep the drawing identical when generating counterfactual backgrounds.
- Candidate backgrounds: **blue**, **green**, **yellow**, and **neutral gray**. Fix their exact RGB values and normalization once chosen.
- Display with nearest-neighbor scaling; do not blur the source for display.
- Maintain separate source drawing, composited input tensor, foreground mask, background color, and intervention metadata.

### 2.4 Model training regimes

Train the same CNN architecture under two conditions:

**Balanced model:** Background color sampled independently of class. For the three experimental class colors, use equal or explicitly measured probabilities and ensure the validation/testing design supports controlled comparisons.

**Shortcut model:** Make one background strongly predictive of each training class:

| Class | Correlated color | Assigned probability |
|---|---|---:|
| Cat | Green | 90% |
| Fish | Blue | 90% |
| Airplane | Yellow | 90% |

Distribute the remaining 10% over the other class-associated colors using a recorded rule. Neutral gray may be reserved for specific evaluation/ablation settings or included according to a documented protocol; avoid ambiguous, changing distribution assumptions.

- Use the **same original drawing splits** for both models.
- Hold architecture, optimizer configuration, epoch budget, and evaluation procedures comparable; record seeds for each run.
- Never claim the balanced model is bias-free or that the shortcut model definitely uses color. **Measure the actual behavior.**
- Generate matched counterfactual groups: same source drawing, only background varied.
- Evaluate not just on the training-like distribution but on balanced and counterfactual backgrounds.

---

## 3. Neural Network Specification

### 3.1 Architecture

Implement a small PyTorch CNN with **28 × 28 × 3 RGB input and 3 output logits**. Use channels-first tensors internally.

```text
Input RGB                       (3, 28, 28)
  Conv2d(3, 8, kernel=3, padding=1), ReLU
Conv1 activation                (8, 28, 28)
  MaxPool2d(2)
Pool1 output                    (8, 14, 14)
  Conv2d(8, 16, kernel=3, padding=1), ReLU
Conv2 activation                (16, 14, 14)
  MaxPool2d(2)
Pool2 output                    (16, 7, 7)
  Flatten
Flattened representation       (784,)
  Linear(784, 32), ReLU
Dense hidden activation        (32,)
  Linear(32, 3)
Output logits                  (3,)
  Softmax for display only
Output classes: CAT / FISH / AIRPLANE
```

Use stride 1 for convolutions. This configuration has **26,611 trainable parameters** with biases. Validate parameter count with a test.

### 3.2 Model requirements

- Expose model version/checkpoint/config in every run.
- Capture **post-ReLU** `conv1`, `conv2`, and dense activations as requested.
- Implement class-logit gradients w.r.t. input for saliency.
- Support channel ablation by replacing a named post-ReLU feature-map channel with zeros **before downstream computation**.
- Use `model.eval()` for inference/evaluation. Ensure deterministic testing for fixed inputs/checkpoints.
- Compare raw logits, softmax probabilities, predicted label, and **target-class logit margin** (target logit minus highest non-target logit).
- Ensure model interventions and normal inference do not leak mutable state between requests.
- Train offline, export compact checkpoints, and run demo inference on CPU.

Do not assume any convolutional channel is intrinsically a “cat ear detector” or another semantic concept; those labels would require specific evidence.

---

## 4. Interface and 2.5D Interaction

### 4.1 Rendering and layout

Use **React + TypeScript**, HTML Canvas, and CSS transforms/transitions. Do not use a full 3D engine.

**2.5D** means a navigable series of **2D planes** with minimal perspective/depth animation. Stop labels:

```text
INPUT  →  CONV1  →  CONV2  →  DENSE  →  OUTPUT
```

Show pooling in optional technical detail rather than as required stops.

Desktop-first layout:

- **Center: Network Explorer** — pixel image or selected activation/feature-map view.
- **Navigation: Layer selector** — Prev/Next, arrow keys, clickable stops, progress.
- **Tool panel** — context-relevant inspection or intervention controls.
- **Results panel** — baseline vs. current predictions, logs/margins, experiment history, Undo, Reset.

Prioritize crisp pixel rendering, readable legends, accessible labels, and low-friction interactions over animation. Account for color-blind accessibility by showing numeric values and noncolor indicators where practical.

### 4.2 View-specific behavior

| Plane | Rendered content | Actions |
|---|---|---|
| Input | 28×28 source/composited image, magnified and optionally gridded | Saliency Scope, Pixel Eraser, Background Switcher |
| Conv1 | 8 thumbnails and enlarged selected 28×28 activation map | Activation Scanner, Ablation Wand |
| Conv2 | 16 thumbnails and enlarged selected 14×14 activation map | Activation Scanner, Ablation Wand |
| Dense | 32 neuron indicators with signed/numeric or post-ReLU magnitudes | Activation Scanner; optional ablation later |
| Output | Cat/Fish/Airplane logits, probabilities, predicted class | Compare baseline vs. interventions |

Activation image contrast scaling must be labeled and not confused with raw activation values.

### 4.3 Experiment state

- Keep a stable **baseline** (original sample, checkpoint, background) for each experiment.
- Maintain current modifications separately. Users can restore a prior history state or reset to the baseline.
- Record hypothesis **before** showing intervention results in guided challenges.
- Show both probability-point and logit/margin changes, not only a classification flip.
- Never silently change the baseline when users switch intervention tools.

---

## 5. Tools: Game Names with Real Scientific Operations

### 5.1 Saliency Scope — input attribution

- Available on **Input**.
- Compute vanilla gradient saliency for a **selected class logit** w.r.t. each RGB input pixel.
- Display `sum` or `mean` of absolute RGB-channel gradients as a 28×28 heatmap; state the aggregation method.
- Support overlay on original input and adjustable opacity.
- Retain unnormalized raw attribution scores in the evaluation output.
- UI explanation: **gradient-based local sensitivity is not itself a causal explanation**.

### 5.2 Pixel Eraser — input intervention

- Available on **Input**.
- Select individual pixels, rectangle, or a fixed patch.
- Two explicit operators, with distinct experiment log types:
  1. **Drawing erasure:** restore selected foreground stroke pixels to the current background.
  2. **Controlled occlusion:** replace the full selected patch according to a specified fixed mask policy used consistently in evaluation.
- Reconstruct actual input and rerun inference; show baseline-vs-modified outputs.
- Record mask coordinates, operation, replacement values, patch size, and affected pixel count.
- Document that masking can create out-of-distribution patterns and that effects depend on the mask policy.

### 5.3 Activation Scanner — observational inspection

- Available on **Conv1**, **Conv2**, **Dense**.
- Present thumbnails, selected map, actual raw values, and per-channel activity summary (e.g., mean post-ReLU activation).
- Allow original/current activation comparison where useful.
- Explain: **responds strongly** is not equivalent to **drives the prediction**.

### 5.4 Ablation Wand — internal intervention

- Available on **Conv1** and **Conv2** in MVP.
- Zero **one entire post-ReLU channel** at the chosen layer for the current input, then recompute downstream outputs.
- Show logits, probabilities, target margin before and after.
- Log the layer, channel, operation, checkpoint, sample, and resulting effect.
- Reset must return exactly to the baseline output.
- Dense and individual spatial activation ablation are stretch features.

### 5.5 Background Switcher — controlled counterfactual input

- Available on **Input**.
- Change only the background RGB color; preserve source stroke mask and geometry.
- Recompute classification and compare either model, using exactly the same source drawing.
- Record original and replacement background values.
- Explain that changing visual context helps test reliance on spurious correlations.

---

## 6. Three Guided Challenges

Every challenge follows **Observe → Hypothesize → Intervene → Evaluate → Reflect**, and stores the player's hypothesis separately from measured results. Do not hardcode expected model outcomes.

### Challenge 1 — Find the Evidence

**Purpose:** Compare saliency attribution with actual input-ablation effects.

1. Select a source drawing and see the initial class prediction.
2. Activate **Saliency Scope** for the predicted or chosen target class.
3. Choose an image region and predict the direction/magnitude category of effect.
4. Use **Pixel Eraser** under a clearly specified mask operator.
5. Display measured target-logit margin/probability changes and any class flip.
6. Compare the intervention to other patches and prompt a short reflection.

**Success:** A user can observe a discrepancy (or agreement) between an attribution and an intervention and understand why the two metrics differ.

### Challenge 2 — Inside the Hidden Layers

**Purpose:** Compare internal activation with causal influence.

1. Navigate to Conv1 or Conv2.
2. Inspect feature maps and choose a channel.
3. Predict whether it has a large target-class influence.
4. Zero the channel via **Ablation Wand**.
5. Display measured effects and compare with other channels.

**Success:** Users learn that high activation does not necessarily imply high target-class influence. Do not presuppose a particular example will demonstrate a mismatch; curate illustrative examples based on legitimate pilot testing while retaining separate quantitative test evaluation.

### Challenge 3 — The Shortcut Trap

**Purpose:** Detect spurious background reliance.

1. Present the same drawing to two matched models, with identities initially hidden as A/B.
2. Invite a hypothesis about which model depends more on background color.
3. Change backgrounds while preserving the foreground.
4. Compare predictions and optionally examine saliency/activations.
5. Have the player identify the shortcut-trained model and state evidence.
6. Reveal the background-training regimes and summarize findings.

**Success:** A user can identify which model is more background-sensitive **based on actual measured responses**. If model behavior does not support the intended demo, fix the experimental setup transparently before publishing the challenge; do not falsify results.

---

## 7. Evaluation Plan — Independent of UI

Implement automated scripts that load saved models and held-out inputs. Export full machine-readable results and publication-quality figures. Ensure experiments are replayable from manifest + seed + checkpoint.

### Evaluation 1 — Saliency vs. input ablation (RQ1)

- For a held-out sample set, compute selected target-class gradient saliency.
- Divide each 28×28 input into a **7×7 grid of 4×4 patches**.
- Aggregate saliency per patch and independently occlude each patch with the **same** documented mask operator.
- Measure signed/absolute changes in target-class logit margin, change in target logit, probabilities, and classification flip.
- Compute Spearman rank correlations per image (where valid), then aggregate with uncertainty; compare high-saliency and random patches using matched intervention budgets.
- Record differences between attribution ranking and intervention ranking; include limitations of mask replacement and gradient method.

### Evaluation 2 — Hidden activation vs. channel influence (RQ2)

- Capture Conv1 and Conv2 post-ReLU activations on held-out images.
- Compute per-channel mean activation or another preregistered summary.
- Ablate **each channel independently**, one at a time, under baseline inputs.
- Measure changes in selected target-class logit margin and top-1 label; retain signed and absolute effects.
- Compute rank correlation between activation and absolute intervention effect; report variation across images/classes/layers.
- Include control tasks or examples so general computation disruption is not mistaken for selective evidence of a semantic circuit.

### Evaluation 3 — Shortcut learning (RQ3)

- Evaluate balanced and shortcut-trained checkpoints on the **same held-out source drawings**.
- Evaluate on training-like, balanced-color, and systematically swapped-color test conditions.
- Change backgrounds while keeping the drawing identical.
- Report: per-condition accuracy, change in target-class logit margins, probability shifts, flip rates, differences between model variants, and confidence intervals.
- Test that input-generation code preserves pixelwise foreground shape across counterfactual variants.

### Evaluation implementation constraints

- Use held-out datasets, fixed manifests, recorded model checkpoint hashes/configuration.
- Export `CSV`/`JSON` rows and summary tables.
- Compute uncertainty (e.g., bootstrap confidence intervals resampling **source drawings** to respect paired variants).
- Show results from full predefined sample sets; curate demo examples separately and label them as illustrative.
- Never infer human learning benefits without a user study; at minimum characterize the experience as an educational design contribution. A small pre/post comprehension test is an optional stretch evaluation.
- Describe negative/null results and all important failure modes.

---

## 8. Architecture and APIs

### 8.1 Tech stack

| Component | Choice |
|---|---|
| Frontend | React + TypeScript |
| Presentation | CSS, HTML Canvas, simple perspective transitions |
| Backend | Python 3, FastAPI, Pydantic |
| ML | PyTorch (CPU serving) |
| Data | NumPy, Pillow |
| Experiments | NumPy, SciPy, pandas, Matplotlib |
| Tests | pytest; Vitest (and UI tests if feasible) |
| Delivery | Docker, GCP Cloud Run, GitHub |

Use straightforward type-safe API models. Avoid dependencies without clear benefit.

### 8.2 Suggested endpoints

| Route | Responsibility |
|---|---|
| `GET /api/health` | Liveness |
| `GET /api/ready` | Check model checkpoints loaded |
| `GET /api/models` | Enumerate model IDs/versions |
| `GET /api/samples` | Return curated or indexed demonstration samples |
| `POST /api/infer` | Normal inference with logits and probabilities |
| `POST /api/activations` | Requested layer activations and metadata |
| `POST /api/saliency` | Class-conditional input gradients |
| `POST /api/intervene` | Image or channel intervention, baseline/modified results |

Design the request schema such that the model, source drawing, original background, selected layer/channel/pixels, and all transformations are explicit and replayable. Validate class labels, pixel ranges, channel indices, dimensions, payload sizes, and request frequency. Do not accept arbitrary Python code, checkpoint URLs, or dynamic user-supplied models.

### 8.3 Example experiment record (illustrative values)

```json
{
  "experiment_id": "exp-001",
  "model_id": "balanced",
  "model_version": "checkpoint-hash-or-version",
  "sample_id": "fish-0042",
  "target_class": "fish",
  "intervention": {
    "type": "channel_ablation",
    "layer": "conv2",
    "channel": 7
  },
  "baseline": {
    "predicted_class": "fish",
    "logits": [0.3, 2.7, 0.2],
    "probabilities": [0.08, 0.85, 0.07]
  },
  "modified": {
    "predicted_class": "fish",
    "logits": [0.5, 1.7, 0.8],
    "probabilities": [0.18, 0.60, 0.22]
  },
  "metrics": {
    "target_logit_margin_change": -1.3
  }
}
```

**Illustrative only.** The actual API must calculate logits, calibrated/softmax probabilities as appropriate, and metric values from real inference; ensure the example JSON is not treated as ground truth or a numerical test fixture.

### 8.4 Proposed repository layout

```text
featuredive/
├── README.md
├── CODEX_IMPLEMENTATION_PLAN.md
├── pyproject.toml
├── frontend/
│   ├── package.json
│   └── src/
│       ├── components/
│       ├── challenges/
│       ├── api/
│       └── styles/
├── backend/
│   ├── main.py
│   ├── api/
│   ├── schemas/
│   ├── services/
│   └── interpretability/
├── ml/
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── data_loader.py
│   └── backgrounds.py
├── checkpoints/
│   ├── balanced.pt
│   └── shortcut.pt
├── experiments/
│   ├── saliency_ablation.py
│   ├── activation_influence.py
│   └── shortcut_learning.py
├── results/
│   ├── metrics/
│   └── figures/
├── tests/
├── deployment/
│   ├── Dockerfile
│   └── deploy.sh
└── docs/
    ├── methodology.md
    ├── research_questions.md
    ├── literature.md
    ├── limitations.md
    ├── presentation.md
    └── poster.pdf
```

Adjust directory conventions when justified but keep the boundaries clear. Do not commit bulk raw datasets. Provide a reproducible download/setup command and include checkpoints or reliable immutable checkpoint-download instructions.

---

## 9. Eight Implementation Phases and Gates

### Phase 0 — Dataset/Model Feasibility **(START HERE AND STOP AFTER COMPLETION)**

**Goal:** Prove this dataset and small CNN can support meaningful educational interventions before front-end development.

**Tasks:**

1. Initialize the minimal Python project with reproducible dependencies and test runner.
2. Download or sample Quick, Draw! 28×28 bitmaps for Cat/Fish/Airplane; record provenance.
3. Produce disjoint deterministic splits and a compact pilot subset.
4. Build exactly the specified CNN and report parameter count.
5. Train a pilot balanced model and measure held-out accuracy/confusion matrix.
6. Train a pilot shortcut model using recorded background assignment probabilities.
7. Evaluate matched background swaps and both checkpoints on common examples.
8. Verify post-ReLU activations, input gradients, and one-channel ablation by assertions.
9. Produce a small collection of real example images, activation maps, and before/after scores.
10. Write a **Phase 0 feasibility report** with commands, actual metrics, examples, limitations, and recommendations.

**Acceptance criteria:**

- Model trains, saved checkpoint loads, and held-out performance is actually measured.
- Background manipulation demonstrably preserves foreground geometry.
- An intervention affects exactly the intended channel at the injection point, and downstream inference remains valid.
- Actual before/after outputs are shown; no simulated results.
- Report whether the class choice and shortcut training create useful behavior. ~80% held-out accuracy is a **pilot target, not a guaranteed requirement**. A weak shortcut signal triggers investigation, not a fabricated demonstration.

**MANDATORY GATE:** **STOP after Phase 0. Do not implement the frontend or proceed to Phase 1 until the user reviews results and authorizes next steps.**

### Phase 1 — Production Data Pipeline and Model Training

**Tasks:** finalize dataset manifests and compositing; train balanced and shortcut checkpoints using comparable hyperparameters; retain configs/seeds, held-out metrics, and model cards.

**Deliverables:** reproducible data pipeline, checkpoint files, train/evaluate commands, evaluation tables, training logs, model documentation.

**Gate:** Each model loads in CPU `eval` mode; repeated inference is deterministic for a fixed sample; provenance and split manifests are documented.

### Phase 2 — Interpretability Engine and API

**Tasks:** activation hooks/explicit tapping, vanilla saliency, drawing erasure, controlled patch occlusion, post-ReLU channel ablation, background swap, baseline comparison; expose validated FastAPI endpoints.

**Deliverables:** reusable Python service/module, API schemas, tests, sample experiment records.

**Gate:** Tests verify correct intervention location, valid outputs, numeric changes, invalid-input handling, reset/reproducibility, and no cross-request contamination.

### Phase 3 — Minimal 2.5D Pixel-Art UI

**Tasks:** React shell; selectable planes and perspective transition; crisp input rendering; activation thumbnails; dense neuron display; tools, before/after dashboard; undo/reset; responsive desktop layout.

**Deliverables:** integrated browser interface using actual service responses.

**Gate:** Select drawing → navigate Conv2 → choose channel → ablate → see real logits/probabilities/label changes → reset baseline, all from browser.

### Phase 4 — Interactive Challenges

**Tasks:** implement Find the Evidence, Inside the Hidden Layers, Shortcut Trap; record hypotheses, interventions, observed outcomes, feedback, and optional progress.

**Deliverables:** three guided end-to-end challenges, teacher-friendly explanations, reproducible demo path.

**Gate:** A newcomer can complete each challenge without ML tooling or prior knowledge of network internals; feedback never relies on invented outputs.

### Phase 5 — Automated Research Evaluation

**Tasks:** implement the three independent RQ evaluations; run full held-out experiment sets; compute aggregate metrics/uncertainty; produce charts/CSV/JSON; write results and caveats.

**Deliverables:** scripts, experiment logs, quantitative figures, evaluation README/methodology, limitations.

**Gate:** Evaluations reproduce independently of the UI from saved checkpoints and manifests; figures are generated from results files; null findings are retained.

### Phase 6 — Online Deployment to GCP

**Recommended hosting:** **Google Cloud Run**, single service and single Docker image: static built React assets served by FastAPI alongside saved CPU PyTorch model checkpoints. No separate database or GPU.

**Tasks:** multi-stage Docker build, health and ready endpoints, static routing and `/api/*`, deployment configuration, CPU/memory profiling, smoke test, conservative billing/scale controls, rollback and shutdown docs.

**Initial settings (measure before finalizing):** 1 vCPU, 2 GiB RAM, min instances 0, max instances 2, no GPU. Min-zero may lead to cold starts.

**Gate:** Once approved and deployed, a user with the public HTTPS URL can load the app and complete all three challenges without account creation. Public smoke tests run against **actual** inference.

**Approval requirement:** Prepare commands, but **do not create potentially billable cloud resources or enable billing without explicit user approval**. Budget alerts are not hard spending caps.

### Phase 7 — Academic Submission and Portfolio Polish

**Tasks:** revise README; document purpose, audience, design decisions, literature, data rights, failure modes, uncertainty, misuse, and intended use; present findings; generate final academic poster and video plan.

**Deliverables:**

- Project/repo link and public demo link (when live).
- Complete, runnable README with dataset and dependency setup, checkpoint provenance, evaluation commands, and deployment instructions.
- **Poster PDF at least 24″ × 24″**, with problem, contribution, demo screenshots, methods, all three measured results, limitations, references.
- ≤5-minute **demonstration-first** presentation script/storyboard, and recorded video if required for online-student submission.
- Model/dataset cards or equivalent documentation of scope, risks, limitations, and known failures.

**Gate:** Someone outside the course can understand the project, use the demo, trace findings to experiments, and reproduce principal claims.

---

## 10. Scientific Integrity, Quality, and Safety

### Scientific standards

- **Do not equate** saliency with causation or an activation map with a human-semantic feature.
- Distinguish effects of interventions **within the model** from real-world causal claims.
- Maintain raw logits and margin metrics alongside probabilities; label all units.
- Avoid treating broad accuracy declines after an ablation as evidence for a specific circuit.
- State that input occlusion can create unusual examples; verify findings across masking choices when feasible.
- Separate pilot/demo curation from independent held-out evaluation.
- Preserve disappointing results and document uncertainties.
- Ground the project in literature on saliency, CNN visualizations, and mechanistic interpretability.

### Engineering standards

- Modular, typed, documented code; reproducible dependency management.
- Automated training/model, preprocessing, intervention, API, and frontend tests.
- Bound API payloads, indices, image dimensions, repeated inference, and intervention counts.
- No arbitrary model checkpoint uploads or code execution.
- No exposed cloud secrets. Limit logging of potentially identifiable user inputs; in MVP, use curated public samples and avoid user accounts.
- Ensure CPU inference performance and graceful loading/errors.
- Contrast/color legends, keyboard interaction, concise tooltips, reset flows.

### Literature starting points

- Simonyan, Vedaldi & Zisserman — *Deep Inside Convolutional Networks* (saliency): <https://arxiv.org/abs/1312.6034>
- Zeiler & Fergus — *Visualizing and Understanding Convolutional Networks*: <https://arxiv.org/abs/1311.2901>
- Ribeiro, Singh & Guestrin — *Why Should I Trust You?* (LIME): <https://arxiv.org/abs/1602.04938>
- Adebayo et al. — *Sanity Checks for Saliency Maps*: <https://arxiv.org/abs/1810.03292>
- Olah et al. — *Zoom In: An Introduction to Circuits*: <https://distill.pub/2020/circuits/zoom-in/>

Cite methods that were actually used; do not imply a complete circuit analysis if the tool implements only channel interventions.

---

## 11. Scope Management

### Required MVP features

- Three-class Quick, Draw! CNN; balanced and shortcut checkpoints.
- 2.5D layer navigator and pixel-art rendering.
- Input saliency, drawing erasure/patch occlusion, activation maps, channel ablation, background switching.
- Baseline versus intervention predictions, raw logits/margins, Undo/Reset, experiment history.
- Three guided hypothesis-testing challenges.
- Independent evaluation experiments for RQ1–RQ3 with measured figures.
- GCP deployment instructions and, with authorization, public Cloud Run demo.
- Reproducibility documentation and course submission materials.

### Explicitly out of scope for MVP

Full 3D world and physics, first-person movement, multiplayer, combat system, animations beyond minimal 2.5D transitions, user accounts/database, large pretrained models, transformers, sparse autoencoders, automated circuit discovery, dense individual spatial-unit ablation, activation patching/steering, adversarial automatic search, and RL.

**Optional stretch goals after MVP:** activation boosting/patching, extra class sets, dSprites validation, adversarial challenger, student comprehension pre/post study, improved graphics/audio.

---

## 12. Final Definition of Done

FeatureDive is complete when:

- [ ] A real trained three-class PyTorch CNN correctly loads and runs on curated Quick, Draw! images.
- [ ] Two versioned checkpoints allow comparable balanced/shortcut experiments.
- [ ] The 2.5D UI navigates Input → Conv1 → Conv2 → Dense → Output.
- [ ] The user can inspect saliency and actual hidden activations.
- [ ] Input and channel interventions change real model computations and show before/after outputs.
- [ ] All three guided challenges support hypothesis, measurement, and reflection.
- [ ] Automated held-out evaluations address RQ1, RQ2, and RQ3 with reproducible data and figures.
- [ ] Testing, data provenance, literature context, limitations, and deployment risks are documented.
- [ ] A public HTTPS demo is accessible after approved Cloud Run deployment.
- [ ] A minimum 24″ × 24″ poster PDF, concise demo video plan, and all required course submission artifacts are ready.

### Final reminder for the first run

**Codex: Implement only Phase 0 now. Report measured feasibility results and stop for human review.**
