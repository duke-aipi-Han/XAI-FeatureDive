# FeatureDive

*Explore the Network. Challenge the Explanation.*

A Responsible AI educational project for exploring a small CNN through
activation inspection, gradient attribution, and controlled interventions.
The planned dataset is Google Quick, Draw! with the fixed class order
`cat = 0`, `fish = 1`, `airplane = 2`.

The source of truth is
[the implementation plan](FeatureDive-Codex-Implementation-Plan.md).

## Current status

This repository contains the minimum Python skeleton for Phase 0. Data loading,
the CNN, training, and evaluations are not implemented yet. There are no trained
checkpoints or measured results. Frontend, API, and deployment work are deferred.

## Local setup

Use Python 3.11 or newer. The `.python-version` file selects Python 3.11 for
tools that support it.

With `uv`:

```sh
uv sync --extra dev
```

Alternatively, with Python's built-in environment tooling:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Core dependencies are PyTorch, NumPy, and Pillow; the `dev` extra adds pytest.
Dependencies have not been installed or locked as part of this scaffold.
When first resolving dependencies with `uv`, commit the generated `uv.lock`
and use `uv sync --locked --extra dev` for subsequent reproducible setup.

## Layout

```text
ml/                  Python package for the Phase 0 data/model code
tests/               Future pytest tests
data/raw/            Downloaded source data (ignored by Git)
data/processed/      Generated splits and processed data (ignored by Git)
checkpoints/         Trained model artifacts (ignored by Git)
results/metrics/     Generated measurements (ignored by Git)
results/figures/     Generated visualizations (ignored by Git)
```

Keep source code, experiment configurations, provenance documentation, and
versioned split manifests in Git. Bulk data and generated artifacts are ignored
by default; add publication artifacts deliberately when they are ready.

## Testing

Once implementation tests exist, run:

```sh
uv run pytest
# Or, inside the activated virtual environment:
python -m pytest
```

No tests exist yet; pytest will currently report no tests collected (exit code 5).

## Initialize Git

From this folder:

```sh
git init -b main
git add .
git commit -m "Initialize FeatureDive Python skeleton"
```

## Next checkpoint

Implement Phase 0 only: deterministic source splits, RGB background compositing,
the specified 26,611-parameter CNN, balanced and shortcut pilot training, and
measured intervention checks. Record real feasibility results and stop for review
before Phase 1 or frontend implementation, as required by the plan.
