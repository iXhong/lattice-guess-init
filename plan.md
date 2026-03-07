## Plan Title
Simplified Two-State Initial Guess Program (Prioritize Correctness and Executability)

## Summary
This plan defines a minimal, practical implementation for extracting robust two-state initial parameters before downstream fitting and analysis.
It intentionally favors correctness and clear flow over over-optimization or over-modularization.

- Nonlinear fitting: `lmfit`
- Linear fitting: `numpy`
- Input shape: `(n_boot, n_time)`
- Design target: runnable, correct, and easy to integrate into your later pipeline

## Scope
1. Implement only the initial-guess extraction workflow.
2. Do not implement your later data-processing, full-fit, or physics-analysis chain.
3. Output two-state initial parameters plus key interval metadata for downstream use.

## File Layout (Minimal)
1. Create a single file: `src/initial_guess.py`
2. Include only these functions:
- `one_state_cosh(...)`
- `two_state_cosh(...)`
- `compute_mean_and_err(...)`
- `build_intervals(...)`
- `find_ground_end_time(...)`
- `ground_linear_seed_numpy(...)`
- `refine_ground_with_lmfit(...)`
- `find_excited_end_time(...)`
- `excited_linear_seed_numpy(...)`
- `estimate_two_state_initial_guess(...)` (main entrypoint)

## Public API
`estimate_two_state_initial_guess(samples_boot_time, time_slices=None, nt_half=48, start_time_index=4, min_interval_length=6, relative_error_threshold=0.2, sigma_band=5.0) -> dict`

### Inputs
- `samples_boot_time`: `np.ndarray`, shape `(n_boot, n_time)`
- `time_slices`: optional; defaults to `np.arange(n_time)`

### Outputs
- `ground_log_amp`, `ground_mass`
- `excited_log_amp`, `excited_mass`
- `mass_gap`
- `ground_interval`, `excited_interval`
- `end_time_ground`, `end_time_excited`
- `diagnostics`

## Algorithm (Mapped to Document Steps 1-6)
1. Compute mean/error
- Internally convert to time-major: `samples_tn = samples_boot_time.T`
- `corr_mean = mean(samples_tn, axis=1)`
- `corr_err = std(samples_tn, axis=1, ddof=1)`

2. Determine ground-state cutoff `end_time_ground`
- Find first `t` such that `corr_err[t] >= relative_error_threshold * corr_mean[t]`
- Set cutoff to `t - 1`; if no hit, use `n_time - 1`

3. Ground-state linear seed (`numpy`)
- Build all contiguous intervals in `[start_time_index, end_time_ground]` with length `>= min_interval_length`
- Run weighted linear regression on `log(corr)` for each interval (`numpy.polyfit(..., w=...)`)
- Filter out invalid points where `corr <= 0`
- Select the median-mass interval to get `ground_log_amp_init`, `ground_mass_init`

4. Ground-state nonlinear refinement (`lmfit`)
- Fit `one_state_cosh` on the selected median interval
- Produce refined `ground_log_amp`, `ground_mass`

5. Subtract ground state and find excited-state window
- `corr_diff = corr_mean - one_state_cosh(...)`
- Determine `end_time_excited` by `corr_mean[t] <= ground_fit[t] + sigma_band * corr_err[t]`

6. Excited-state linear seed (`numpy`)
- Fit `log(corr_diff)` over intervals within `[0, end_time_excited]`
- Use the longest valid interval by default for `excited_log_amp`, `excited_mass`
- Enforce `excited_mass > ground_mass`; otherwise set `excited_mass = ground_mass + eps`
- Compute `mass_gap = excited_mass - ground_mass` and keep it strictly positive

## Important API/Type Decisions
1. Input orientation is fixed as `(n_boot, n_time)` at the public boundary.
2. Internal calculations may transpose to time-major for cleaner vectorized logic.
3. The output is a plain `dict` for direct interoperability with downstream code.

## Naming and Documentation Rules
1. Use `snake_case` consistently; avoid C++-style abbreviated naming.
2. Add concise docstrings to each function:
- expected input shape
- return fields
- failure conditions
3. Add comments only for critical intent:
- why truncation logic is used
- why nonpositive values are masked before `log`
- why `mass_gap > 0` is enforced

## Error Handling
1. No valid ground-state intervals:
- raise `ValueError("no valid ground-state intervals")`
2. Insufficient valid excited-state points:
- fallback seed: `excited_mass = ground_mass + 0.05 * abs(ground_mass) + 1e-4`
- set `diagnostics["excited_fallback"] = True`
3. `lmfit` failure:
- fallback to linear seed
- set `diagnostics["ground_refine_failed"] = True`

## Test Cases and Scenarios
1. Add `tests/test_initial_guess.py`
2. Include four essential tests:
- accepts `(n_boot, n_time)` input and returns all required fields
- ground cutoff rule behaves as expected
- nonpositive samples do not crash `log` workflow
- `mass_gap` is always strictly positive

## Assumptions and Defaults
1. Upstream code provides bootstrap samples; this module does not generate bootstrap resamples.
2. Default physics-like window is supported (e.g., `n_time=49`, `nt_half=48`) but configurable.
3. Current priority is correctness and basic runtime viability; no parallel/performance optimization now.
4. If implementation details are ambiguous, use `reference/` C++ code as the source of truth rather than ad-hoc assumptions.
