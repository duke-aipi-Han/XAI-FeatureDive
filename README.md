# FeatureDive

*Explore the Network. Challenge the Explanation.*

A Responsible AI educational project for exploring a small CNN through
activation inspection, gradient attribution, and controlled interventions.
The source of truth is [the implementation plan](FeatureDive-Codex-Implementation-Plan.md).

## Current status

**Phase 1 is complete.** Both final models were trained on this computer’s MPS
accelerator using the full 30,000-source dataset (24,000 training, 3,000
validation, 3,000 test). All pilot drawings and exact bitmap copies are excluded.
Class order remains `cat = 0`, `fish = 1`, `airplane = 2`.

Both versioned checkpoints load on CPU in eval mode. Repeating inference for all
12,000 held-out source/color inputs per model produces exactly equal logits.
API, frontend, and deployment work await their respective phases.

| Phase 1 measurement | Balanced model | Shortcut model |
| --- | ---: | ---: |
| Accuracy across three equally weighted backgrounds | 90.74% | 77.17% |
| Training-like expected accuracy | 90.74% | 95.13% |
| Sources with any label flip across three backgrounds | 2.70% | 40.97% |

These are single-seed measurements. The [Phase 1 report](docs/phase1-training.md)
contains confidence intervals, confusion matrices, histories, and artifact hashes.
See the [dataset card](docs/data-card.md) and model cards for
[balanced](docs/model-cards/phase1-v1-balanced.md) and
[shortcut](docs/model-cards/phase1-v1-shortcut.md).

![Phase 1 measured results](results/phase1/overview.png)

## Set up and evaluate Phase 1

Python 3.11 is recorded in `.python-version`; dependencies are locked in `uv.lock`.
Run from the repository root:

```sh
uv sync --locked --extra dev
uv run --locked python -m ml.data_loader --config configs/phase1.json
uv run --locked python -m ml.evaluate --config configs/phase1.json
uv run --locked pytest -q
```

The two small checkpoints and `checkpoints/phase1-v1/registry.json` are included.
Evaluation uses CPU and does not retrain. The download/setup command reuses local
source files and verifies their checksums against the frozen manifest. After
setup, training and evaluation work offline.

To train a new copy in a fresh checkout without existing checkpoints:

```sh
uv run --locked python -m ml.train --config configs/phase1.json
```

The configured `auto` training device selects MPS when available, otherwise CPU.
Use `--device mps` to require MPS or `--device cpu` for CPU training. An explicit
MPS request fails if unavailable; a restricted sandbox may hide it. Training
protects existing checkpoints; to deliberately replace this recorded run:

```sh
uv run --locked python -m ml.train --config configs/phase1.json --device mps --overwrite
uv run --locked python -m ml.evaluate --config configs/phase1.json
```

Use a new config/version and new artifact paths for a different experiment.
`configs/phase1.json` records shared hyperparameters and seeds. Seeds and
requested deterministic algorithms support replay; identical training weights
across hardware/software are not guaranteed. The CPU inference gate is measured
independently. Training logs retain the actual device and tensor/content hashes.

The Phase 0 [feasibility report](docs/phase0-feasibility.md), pilot checkpoints,
and pilot source manifest remain preserved with their original measurements.

## Reproduce Phase 0

Python 3.11 is recorded in `.python-version`. Dependencies are locked in `uv.lock`.
Run from the repository root:

```sh
uv sync --locked --extra dev
uv run --locked python -m ml.data_loader
uv run --locked python -m ml.evaluate
uv run --locked pytest -q
```

The download command fetches three public `.npy` files (about 306 MiB total).
It verifies existing arrays and reuses the frozen manifest, rejecting source
checksum or split-setting mismatches. After download and setup, training and
evaluation run offline on CPU. Evaluation loads the saved checkpoints; it does
not train. To deliberately retrain the existing pilot checkpoints on CPU, run
`uv run --locked python -m ml.train --device cpu --overwrite`, then evaluate again.

The pilot uses 2,000 train / 300 validation / 300 test sources per class,
selected from full-category files using seed 590 before color assignment.
[configs/phase0.json](configs/phase0.json) records hyperparameters. Both models
use the specified 26,611-parameter CNN and the same initialization and training
budget. Training colors are fixed per source, with recorded counts. Neutral gray
is reserved for evaluation.

If using pip instead of uv:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m ml.data_loader
python -m ml.evaluate
python -m pytest -q
```

The pip alternative resolves version ranges; use the locked uv workflow to
reproduce the recorded dependency versions.

## Layout

```text
configs/phase{0,1}.json      Versioned experiment configurations
ml/data_loader.py           Downloads, checksums, source manifest, split loading
ml/backgrounds.py           Alpha compositing and training color assignments
ml/model.py                 CNN with explicit post-ReLU activation/ablation taps
ml/train.py                 CPU/MPS training and CPU checkpoint loading
ml/experiment.py            Validated experiment paths and device selection
ml/interpretability.py      Class-logit gradient saliency and target margins
ml/evaluate.py              Paired color evaluation and real intervention examples
ml/figures.py               Phase 0 figures and feasibility report
ml/phase1_report.py         Final training report, figures, tables, and model cards
tests/                      Scientific invariants and evaluation checks
data/manifests/             Versioned source indices, hashes and provenance
data/raw/                   Downloaded bitmaps (ignored by Git)
data/processed/             Reserved for derived datasets (ignored by Git)
checkpoints/                Pilot checkpoints and phase1-v1 final checkpoints
results/phase0/              Versioned pilot JSON, CSV, raw arrays, and PNGs
results/phase1/              Final measured outputs and training assignments
docs/                       Phase reports, dataset card, and model cards
```

Bulk source arrays and future generated checkpoints are excluded from Git. The
compact pilot and Phase 1 checkpoints are explicitly included so measurements
can be replayed without retraining. Their SHA-256 values and metadata are recorded
in the report and results. Source manifests and measured pilot outputs are also
included for review.

## Interpretation and data attribution

Activation is an observation. Gradient saliency measures local sensitivity.
Input occlusion and post-ReLU channel zeroing are interventions with measured
model effects. None establishes a uniquely meaningful semantic feature or
real-world causal mechanism. Softmax outputs are uncalibrated probabilities.

The Quick, Draw! Dataset is made available by Google, Inc. under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), as recorded in the
[canonical dataset documentation](https://github.com/googlecreativelab/quickdraw-dataset).
This project samples bitmap rows, composites white strokes over colored
backgrounds, and generates intervention variants. See [data/README.md](data/README.md)
for provenance and source-format limitations.

**Stop after Phase 1:** review the full-size dataset and model results before
authorizing Phase 2, as required by the implementation plan.
