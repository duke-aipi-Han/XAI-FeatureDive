# FeatureDive

*Explore the Network. Challenge the Explanation.*

A Responsible AI educational project for exploring a small CNN through
activation inspection, gradient attribution, and controlled interventions.
The source of truth is [the implementation plan](FeatureDive-Codex-Implementation-Plan.md).

## Current status

**Phase 0 is implemented and measured.** Two real CPU-trained pilot checkpoints
use Google Quick, Draw! bitmaps with `cat = 0`, `fish = 1`, `airplane = 2`.
No frontend, API, or deployment has been started. Phase 1 awaits review.

On 900 held-out source drawings:

| Pilot measurement | Balanced model | Shortcut model |
| --- | ---: | ---: |
| Accuracy across three equally weighted backgrounds | 84.56% | 65.56% |
| Accuracy under each model's training-like color distribution | 84.56% | 93.42% |
| Sources with any label flip across three backgrounds | 2.22% | 65.33% |

These are single-seed pilot results. Read the
[feasibility report](docs/phase0-feasibility.md) for confidence intervals,
confusion matrices, real intervention examples, provenance, and limitations.

![Phase 0 measured results](results/phase0/overview.png)

## Reproduce Phase 0

Python 3.11 is recorded in `.python-version`. Dependencies are locked in `uv.lock`.
Run from the repository root:

```sh
uv sync --locked --extra dev
uv run --locked python -m ml.data_loader
uv run --locked python -m ml.train
uv run --locked python -m ml.evaluate
uv run --locked pytest -q
```

The download command fetches three public `.npy` files (about 337 MiB total).
It verifies existing arrays and reuses the frozen manifest, rejecting source
checksum or split-setting mismatches. After download and setup, training and
evaluation run offline on CPU. Evaluation loads the saved checkpoints; it does
not train. Rerunning training overwrites the two local pilot checkpoints.

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
python -m ml.train
python -m ml.evaluate
python -m pytest -q
```

The pip alternative resolves version ranges; use the locked uv workflow to
reproduce the recorded dependency versions.

## Layout

```text
configs/phase0.json          Recorded pilot configuration
ml/data_loader.py           Downloads, checksums, source manifest, split loading
ml/backgrounds.py           Alpha compositing and training color assignments
ml/model.py                 CNN with explicit post-ReLU activation/ablation taps
ml/train.py                 CPU pilots and validation-selected checkpoints
ml/interpretability.py      Class-logit gradient saliency and target margins
ml/evaluate.py              Paired color evaluation and real intervention examples
ml/figures.py               Figures and report generated from measured outputs
tests/                      Scientific invariants and evaluation checks
data/manifests/             Versioned source indices, hashes and provenance
data/raw/                   Downloaded bitmaps (ignored by Git)
data/processed/             Reserved for derived datasets (ignored by Git)
checkpoints/                Two compact, versioned pilot checkpoints
results/phase0/              Versioned pilot JSON, CSV, raw arrays, and PNGs
docs/phase0-feasibility.md   Measured report and Phase 0 review checkpoint
```

Bulk source arrays and future generated checkpoints are excluded from Git. The
two compact pilot checkpoints are explicitly included so the measured examples
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

**Stop after Phase 0:** review the feasibility report before authorizing Phase 1
or frontend development, as required by the implementation plan.
