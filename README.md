# lattice-guess-init

Initial-parameter estimation toolkit for lattice two-point correlator fits.

This project estimates robust one-state and two-state initial parameters from bootstrap samples with shape `(n_boot, n_time)`, and can persist interval-level results for downstream analysis pipelines.

## Who this is for

- You have bootstrap correlator arrays `(n_boot, n_time)`
- You need stable initial parameters for one-state/two-state fits
- You want reusable JSON artifacts for later fitting stages

## Features

- Full interval-scan estimator in [`guess_init/initial_guess_full.py`](guess_init/initial_guess_full.py)
- CLI runner for single-file and batch processing in [`demo_run.py`](demo_run.py)
- Structured JSON output (`all_intervals_results`) for future loading
- Tiny loader utility in [`guess_init/init_guess_loader.py`](guess_init/init_guess_loader.py) that returns interval-wise one-state and two-state initial params

## Installation

### Local development

```bash
uv venv .venv
UV_CACHE_DIR=.uv-cache uv pip install --python .venv/bin/python numpy lmfit pytest
UV_CACHE_DIR=.uv-cache uv pip install --python .venv/bin/python -e .
```

### Reuse in another project (recommended)

Install from GitHub with a pinned tag/commit:

```bash
uv add "guess-init @ git+https://github.com/iXhong/lattice-guess-init.git@<tag-or-commit>"
```

Or with pip:

```bash
pip install "git+https://github.com/iXhong/lattice-guess-init.git@<tag-or-commit>"
```

## Quick Start

### Run estimator on one file

```bash
UV_CACHE_DIR=.uv-cache uv run --python .venv/bin/python demo_run.py data/p2_0_bs.npy --top-k 5
```

### Batch run (`p2_*_bs.npy`)

```bash
UV_CACHE_DIR=.uv-cache uv run --python .venv/bin/python demo_run.py data/processed/bs --top-k 3
```

### Save per-`p2` JSON

```bash
UV_CACHE_DIR=.uv-cache uv run --python .venv/bin/python demo_run.py data/processed/bs --save --output-dir data/processed/init_guess
```

Example mapping:
- `p2_0_bs.npy` -> `p2_0_init_guess.json`

## JSON Output Schema

Each saved JSON contains:

- `schema_version`
- `unit` (`p2_id`, `input_file`, `output_file`, `created_at`)
- `config` (estimator runtime args)
- `all_intervals_results` (always fully saved)

Each `all_intervals_results` entry includes:

- `interval_start`, `interval_end`, `interval`
- `seed`:
  - `local_ground_log_amp`, `local_ground_mass`
  - `guess_excited_log_amp`, `guess_mass_gap`
- `fit_status`
- `fit_result` (full two-state fit result or `null`)

## Load Saved Results

Programmatic:

```python
from guess_init.init_guess_loader import load_interval_initial_params

result = load_interval_initial_params("data/processed/init_guess/p2_0_init_guess.json")
interval = result["intervals"][0]
print(interval["one_state_init"])
print(interval["two_state_init"])
```

CLI:

```bash
python -m guess_init.init_guess_loader data/processed/init_guess/p2_0_init_guess.json --top-k 5
```

## Reusing in Other Projects

Recommended integration pattern:

1. Install this package from GitHub with a pinned version.
2. Use estimator API to generate per-`p2` JSON artifacts in your pipeline.
3. Use loader API to consume interval-wise initial parameters in your fitter stage.

Minimal API entry points:

- `estimate_two_state_initial_guess(...)` from `guess_init.initial_guess_full`
- `load_interval_initial_params(...)` from `guess_init.init_guess_loader`

Minimal estimator usage:

```python
import numpy as np
from guess_init.initial_guess_full import estimate_two_state_initial_guess

samples = np.load("p2_0_bs.npy")  # shape: (n_boot, n_time)
result = estimate_two_state_initial_guess(samples)
print(result["all_intervals_results"][0])
```

## Development

Run tests:

```bash
UV_CACHE_DIR=.uv-cache uv run --python .venv/bin/python pytest -q
```

Run the cross-project synthetic smoke test used by CI:

```bash
UV_CACHE_DIR=.uv-cache uv run --python .venv/bin/python pytest -q tests/test_guess_init_smoke.py
```

## Notes

- Current focus is robust initial-guess estimation and artifact persistence.
- For ambiguous algorithm details, align with the C++ references under `reference/`.
- `src.*` imports are deprecated compatibility shims.
- Use `guess_init.*` now; `src.*` is planned to be removed after one release cycle.
