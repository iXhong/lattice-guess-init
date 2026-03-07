import sys
from pathlib import Path

from lmfit import Parameters, minimize

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))


import numpy as np
from guess_init.initial_guess import (
    compute_mean_and_err,
    build_intervals,
    find_ground_end_time,
    _fit_log_linear_numpy,
    ground_linear_seed_numpy,
    one_state_cosh,
    excited_linear_seed_numpy,
    # refine_ground_with_lmfit,
    estimate_two_state_initial_guess,
)


def test_compute_mean_and_err():
    data = np.load("data/p2_0_bs.npy")
    mean, err = compute_mean_and_err(data)
    # formated output with 5 decimal places
    print(f"Mean: {mean}, Error: {err}")
    # assert mean > 0, "Mean should be positive"
    # assert err > 0, "Error should be positive"


def test_build_intervals():
    start_time_index = 4
    end_time_index = 29
    min_interval_length = 6
    intervals = build_intervals(start_time_index, end_time_index, min_interval_length)
    print(f"Generated intervals: {intervals}")
    return intervals
    # assert all(end - start >= min_interval_length for start, end in intervals), "All intervals should have at least the minimum length"


def test_find_ground_end_time():

    # corr_mean = np.array([10.0, 8.0, 6.0, 4.0])
    # corr_err = np.array([0.2, 0.4, 1.3, 1.5])
    # # First hit at t=2 because 1.3 >= 0.2 * 6.0
    # assert find_ground_end_time(corr_mean, corr_err, relative_error_threshold=0.2) == 1
    # data = np.load("data/p2_0_bs.npy")
    data = np.load("data/processed/bs/p2_0_bs.npy")
    mean, err = compute_mean_and_err(data)
    end_time = find_ground_end_time(mean, err, relative_error_threshold=0.2)
    print(f"End time for ground state: {end_time}")
    # assert end_time >= 0, "End time should be non-negative"


def test_fit_log_linear_numpy():
    t = np.arange(49)
    data = np.load("data/processed/bs/p2_0_bs.npy")
    signal_time_boot = data.T
    end_time = find_ground_end_time(
        *compute_mean_and_err(data), relative_error_threshold=0.2
    )
    intervals = build_intervals(4, end_time, min_interval_length=6)

    results = _fit_log_linear_numpy(t, signal_time_boot, intervals[0], 48)
    print(f"Fit results for interval {intervals[0]}: {results}")


def test_ground_linear_seed_numpy():
    t = np.arange(49)
    data = np.load("data/processed/bs/p2_27_bs.npy")
    signal_time_boot = data.T
    end_time = find_ground_end_time(
        *compute_mean_and_err(data), relative_error_threshold=0.2
    )
    intervals = build_intervals(4, end_time, min_interval_length=6)

    result = ground_linear_seed_numpy(signal_time_boot, t, intervals, nt_half=48)
    # print(f"Ground linear seed result: {result}")
    print(f"Ground linear select result: {result['selected_interval']}")


def test_refine_ground_with_lmfit():
    t = np.arange(49)
    data = np.load("data/processed/bs/p2_0_bs.npy")
    signal_time_boot = data.T
    corr_mean, corr_err = compute_mean_and_err(data)
    # print(f"shape of corr_mean: {corr_mean.shape}, shape of corr_err: {corr_err.shape}")
    end_time = find_ground_end_time(corr_mean, corr_err, relative_error_threshold=0.2)
    intervals = build_intervals(4, end_time, min_interval_length=6)

    initial_guess = ground_linear_seed_numpy(signal_time_boot, t, intervals, nt_half=48)
    selected_interval = initial_guess["selected_interval"]
    log_amp_init = initial_guess["log_amp_init"]
    mass_init = initial_guess["mass_init"]
    print(
        f"Initial guess: log_amp={log_amp_init}, mass={mass_init}, selected_interval={selected_interval}"
    )

    residual = (
        lambda params, t, data, err: (
            one_state_cosh(t, params["log_amp"].value, params["mass"].value, nt_half=48)
            - data
        )
        / err
    )
    params = Parameters()
    params.add(
        "log_amp",
        value=log_amp_init,
        min=min(0.5 * log_amp_init, 2 * log_amp_init),
        max=max(0.5 * log_amp_init, 2 * log_amp_init),
    )
    params.add(
        "mass",
        value=mass_init,
        min=min(0.5 * mass_init, 2 * mass_init),
        max=max(0.5 * mass_init, 2 * mass_init),
    )

    # print(f"original data: t={t}, corr_mean={corr_mean}, corr_err={corr_err}")
    fit_interval = selected_interval
    t_fit = t[fit_interval]
    data_fit = corr_mean[fit_interval]
    err_fit = corr_err[fit_interval]
    # print(f"Fitting with lmfit on interval {fit_interval} (t={t_fit})")
    # print(f"data_fit: {data_fit}")
    # print(f"err_fit: {err_fit}")
    result = minimize(
        residual,
        params,
        args=(t_fit, data_fit, err_fit),
        method="leastsq",
    )
    refined_result = {
        "log_amp": result.params["log_amp"].value,
        "mass": result.params["mass"].value,
        "log_amp_err": result.params["log_amp"].stderr,
        "mass_err": result.params["mass"].stderr,
        "success": result.success,
        "message": result.message,
        "chisqr": float(result.chisqr),
    }

    print(f"Refined ground state parameters: {refined_result}")


def test_estimate_two_state_initial_guess():
    t = np.arange(49)

    data = np.load("data/processed/bs/p2_0_bs.npy")
    result = estimate_two_state_initial_guess(
        samples_boot_time=data,
        time_slices=t,
        nt_half=48,
        start_time_index=4,
        min_interval_length=6,
        relative_error_threshold=0.2,
        sigma_band=5.0,
    )


if __name__ == "__main__":
    # test_compute_mean_and_err()
    # test_build_intervals()
    # test_find_ground_end_time()
    # test_fit_log_linear_numpy()
    # test_ground_linear_seed_numpy()
    # test_refine_ground_with_lmfit()
    test_estimate_two_state_initial_guess()
