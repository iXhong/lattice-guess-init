import numpy as np
import pytest

from src.initial_guess import (
    estimate_two_state_initial_guess,
    find_ground_end_time,
    one_state_cosh,
)


def _make_synthetic_samples(
    n_boot: int = 96,
    n_time: int = 49,
    nt_half: int = 48,
    seed: int = 7,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n_time, dtype=float)
    ground = one_state_cosh(t, log_amp=-2.8, mass=0.22, nt_half=nt_half)
    excited = one_state_cosh(t, log_amp=-4.2, mass=0.42, nt_half=nt_half)
    mean_curve = ground + excited
    sigma = 0.01 * np.maximum(mean_curve, 1e-8)
    samples = rng.normal(loc=mean_curve, scale=sigma, size=(n_boot, n_time))
    return samples


def test_find_ground_end_time_threshold_rule():
    corr_mean = np.array([10.0, 8.0, 6.0, 4.0])
    corr_err = np.array([0.2, 0.4, 1.3, 1.5])
    # First hit at t=2 because 1.3 >= 0.2 * 6.0
    assert find_ground_end_time(corr_mean, corr_err, relative_error_threshold=0.2) == 1


def test_nonpositive_values_do_not_crash_workflow():
    samples = _make_synthetic_samples(n_boot=64, n_time=49)
    samples[:, :2] *= -1.0  # Force non-positive values in early times.
    pytest.importorskip("lmfit")
    result = estimate_two_state_initial_guess(samples)
    assert "diagnostics" in result
    assert np.isfinite(result["mass_gap"])


def test_main_api_accepts_nboot_time_and_returns_required_keys():
    pytest.importorskip("lmfit")
    samples = _make_synthetic_samples(n_boot=80, n_time=49)
    result = estimate_two_state_initial_guess(samples)
    required_keys = {
        "ground_log_amp",
        "ground_mass",
        "excited_log_amp",
        "excited_mass",
        "mass_gap",
        "ground_interval",
        "excited_interval",
        "end_time_ground",
        "end_time_excited",
        "diagnostics",
    }
    assert required_keys.issubset(result.keys())
    assert result["mass_gap"] > 0.0


def test_mass_gap_is_strictly_positive():
    pytest.importorskip("lmfit")
    samples = _make_synthetic_samples(n_boot=48, n_time=49)
    # Inflate noise to increase chance of unstable excited seed; mass_gap must still be positive.
    rng = np.random.default_rng(13)
    samples += rng.normal(0.0, 0.1, size=samples.shape)
    result = estimate_two_state_initial_guess(samples)
    assert result["mass_gap"] > 0.0

