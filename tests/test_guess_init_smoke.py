"""CI smoke test for guess_init public API on synthetic data."""

from __future__ import annotations

import numpy as np
import pytest


def _one_state_cosh(
    time_slices: np.ndarray,
    log_amp: float,
    mass: float,
    nt_half: int,
) -> np.ndarray:
    dt = np.asarray(time_slices, dtype=float) - float(nt_half)
    return np.exp(log_amp + mass * dt) + np.exp(log_amp - mass * dt)


def _make_synthetic_samples(
    n_boot: int = 96,
    n_time: int = 49,
    nt_half: int = 48,
    seed: int = 7,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n_time, dtype=float)
    ground = _one_state_cosh(t, log_amp=-2.8, mass=0.22, nt_half=nt_half)
    excited = _one_state_cosh(t, log_amp=-4.2, mass=0.42, nt_half=nt_half)
    mean_curve = ground + excited
    sigma = 0.01 * np.maximum(mean_curve, 1e-8)
    return rng.normal(loc=mean_curve, scale=sigma, size=(n_boot, n_time))


def test_guess_init_smoke_synthetic():
    pytest.importorskip("lmfit")
    from guess_init import estimate_two_state_initial_guess

    samples = _make_synthetic_samples()
    result = estimate_two_state_initial_guess(
        samples_boot_time=samples,
        nt_half=48,
        start_time_index=4,
        min_interval_length=6,
        relative_error_threshold=0.2,
        sigma_band=5.0,
    )

    required_top_keys = {
        "end_time_ground",
        "end_time_excited",
        "median_ground_log_amp",
        "median_ground_mass",
        "median_ground_interval",
        "all_intervals_results",
    }
    missing_top = required_top_keys.difference(result.keys())
    assert not missing_top, f"Missing top-level keys: {sorted(missing_top)}"

    intervals = result["all_intervals_results"]
    assert isinstance(intervals, list), "all_intervals_results must be a list"
    assert intervals, "all_intervals_results must be non-empty"

    first = intervals[0]
    required_interval_keys = {
        "interval_start",
        "interval_end",
        "interval",
        "local_ground_log_amp",
        "local_ground_mass",
        "guess_excited_log_amp",
        "guess_mass_gap",
        "two_state_fit_success",
        "two_state_result",
    }
    missing_interval = required_interval_keys.difference(first.keys())
    assert not missing_interval, f"Missing interval keys: {sorted(missing_interval)}"
    assert first["guess_mass_gap"] > 0, "guess_mass_gap must be > 0"
