# guess_init

Initial-guess toolkit for two-state correlator fitting from bootstrap samples `(n_boot, n_time)`.

## What is implemented

- `src/initial_guess_full.py`
  - Full interval-scan initial-guess estimator (one-state + two-state preparation/fit flow).
  - Main API: `estimate_two_state_initial_guess(samples_boot_time, ...)`.

- `demo_run.py`
  - Runs estimator on one file or batch directory.
  - Supports saving per-`p2` JSON for future pipeline loading.

- `src/init_guess_loader.py`
  - Tiny loader utility for saved JSON files.
  - Returns interval-wise one-state and two-state **initial parameters** in a structured format.

## Environment

```bash
uv venv .venv
UV_CACHE_DIR=.uv-cache uv pip install --python .venv/bin/python numpy lmfit pytest
```

## Run estimator (demo_run)

### Single file

```bash
UV_CACHE_DIR=.uv-cache uv run --python .venv/bin/python demo_run.py data/p2_0_bs.npy --top-k 5
```

### Batch directory (`p2_*_bs.npy`)

```bash
UV_CACHE_DIR=.uv-cache uv run --python .venv/bin/python demo_run.py data/processed/bs --top-k 3
```

### Save per-`p2` JSON

```bash
UV_CACHE_DIR=.uv-cache uv run --python .venv/bin/python demo_run.py data/processed/bs --save --output-dir data/processed/init_guess
```

Output mapping example:
- `p2_0_bs.npy` -> `data/processed/init_guess/p2_0_init_guess.json`

## Saved JSON structure (current)

Top-level fields:
- `schema_version`
- `unit` (`p2_id`, `input_file`, `output_file`, `created_at`)
- `config` (run args)
- `all_intervals_results` (full interval scan, always saved)

Each `all_intervals_results` item is normalized as:
- `interval_start`, `interval_end`, `interval`
- `seed`:
  - `local_ground_log_amp`, `local_ground_mass`
  - `guess_excited_log_amp`, `guess_mass_gap`
- `fit_status`
- `fit_result` (full two-state fit object or `null`)

## Load saved JSON (init_guess_loader)

### Programmatic usage

```python
from src.init_guess_loader import load_interval_initial_params

data = load_interval_initial_params("data/processed/init_guess/p2_0_init_guess.json")
# data["intervals"][i] contains:
# - one_state_init: {log_amp, mass}
# - two_state_init: {log_amp_1, mass_1, log_amp_2, mass_gap, mass_2}
```

### CLI usage

```bash
python src/init_guess_loader.py data/processed/init_guess/p2_0_init_guess.json --top-k 5
```

## Tests

```bash
UV_CACHE_DIR=.uv-cache uv run --python .venv/bin/python pytest -q
```

## Notes

- Current focus is initial-guess estimation and structured persistence for future loading.
- If algorithm details are uncertain, align with `reference/` C++ logic first.
