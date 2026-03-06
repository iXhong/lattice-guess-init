"""Initial parameter estimation for two-state correlator fits.

Public input shape is (n_boot, n_time). Internally this module transposes to
time-major (n_time, n_boot) for clearer vectorized operations.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict

import numpy as np
from lmfit import Parameters, minimize

from src.config import NUMBER_BOOTSTRAP, NUMBER_CONF


def one_state_cosh(
    time_slices: np.ndarray,
    log_amp: float,
    mass: float,
    nt_half: int = 48,
) -> np.ndarray:
    """Compute the one-state cosh correlator model."""
    dt = np.asarray(time_slices, dtype=float) - float(nt_half)
    return np.exp(log_amp + mass * dt) + np.exp(log_amp - mass * dt)


def two_state_cosh(
    time_slices: np.ndarray,
    log_amp_1: float,
    mass_1: float,
    log_amp_2: float,
    mass_gap: float,
    nt_half: int = 48,
) -> np.ndarray:
    """Compute the two-state cosh correlator model with positive mass gap."""
    mass_2 = mass_1 + mass_gap
    return one_state_cosh(time_slices, log_amp_1, mass_1, nt_half) + one_state_cosh(
        time_slices, log_amp_2, mass_2, nt_half
    )


def compute_mean_and_err(
    samples_boot_time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return bootstrap mean and error over the bootstrap axis.

    Parameters
    ----------
    samples_boot_time:
        Array of shape (n_boot, n_time).
    """
    samples = np.asarray(samples_boot_time, dtype=float)
    if samples.ndim != 2:
        raise ValueError(
            "samples_boot_time must be a 2D array of shape (n_boot, n_time)"
        )
    if samples.shape[0] < 2:
        raise ValueError("samples_boot_time requires at least two bootstrap samples")
    samples_time_boot = samples.T
    corr_mean = np.mean(samples_time_boot, axis=1)
    # corr_err = np.std(samples_time_boot, axis=1, ddof=1)
    cov_matrix = (NUMBER_CONF / (NUMBER_CONF - 1)) * np.cov(
        samples_time_boot, rowvar=True, ddof=1, dtype=np.float64
    )
    corr_err = np.sqrt(np.diag(cov_matrix))
    return corr_mean, corr_err


def build_intervals(
    start_time_index: int,
    end_time_index: int,
    min_interval_length: int,
) -> list[np.ndarray]:
    """Build all contiguous intervals [i, ..., j] with minimum length."""
    if min_interval_length < 2:
        raise ValueError("min_interval_length must be >= 2")
    if end_time_index < start_time_index:
        return []

    intervals: list[np.ndarray] = []
    max_start = end_time_index - min_interval_length + 1
    for start in range(start_time_index, max_start + 1):
        # min_end = start + min_interval_length - 1
        # for end in range(min_end, end_time_index + 1):
        #     intervals.append(np.arange(start, end + 1, dtype=int))
        intervals.append(np.arange(start, end_time_index + 1, dtype=int))
    return intervals


def find_ground_end_time(
    corr_mean: np.ndarray,
    corr_err: np.ndarray,
    relative_error_threshold: float = 0.2,
) -> int:
    """Find the latest time index for ground-state initial fit."""
    n_time = len(corr_mean)
    for t in range(n_time):
        # Use absolute mean to keep the relative-error rule meaningful even if
        # noisy bootstrap means become negative at early times.
        if corr_err[t] >= relative_error_threshold * abs(corr_mean[t]):
            return max(t - 1, 0)
    return n_time - 1


# def _fit_log_linear_numpy(
#     time_values: np.ndarray,
#     signal_time_boot: np.ndarray,
#     interval: np.ndarray,
#     nt_half: int,
# ) -> dict[str, Any] | None:
#     """Run weighted linear fit on log(signal) in one interval."""
#     t = np.asarray(time_values[interval], dtype=float)
#     signal_slice = signal_time_boot[interval, :]

#     # Mask non-positive values before log to avoid invalid operations.
#     with np.errstate(divide="ignore", invalid="ignore"):
#         log_samples = np.where(signal_slice > 0.0, np.log(signal_slice), np.nan)

#     y_mean = np.nanmean(log_samples, axis=1)
#     y_sigma = np.nanstd(log_samples, axis=1, ddof=1)
#     valid = np.isfinite(y_mean) & np.isfinite(y_sigma)
#     if int(np.sum(valid)) < 3:
#         return None

#     x_fit = t[valid]
#     y_fit = y_mean[valid]
#     sigma_fit = y_sigma[valid]
#     sigma_floor = max(np.nanmedian(np.abs(sigma_fit)) * 1e-3, 1e-12)
#     sigma_fit = np.where(sigma_fit > 0.0, sigma_fit, sigma_floor)
#     w_fit = 1.0 / sigma_fit

#     try:
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             slope, intercept = np.polyfit(x_fit, y_fit, deg=1, w=w_fit)
#     except Exception:
#         return None

#     mass = -float(slope)
#     log_amp = float(intercept + mass * nt_half)
#     return {
#         "interval": interval,
#         "interval_start": int(interval[0]),
#         "interval_end": int(interval[-1]),
#         "log_amp": log_amp,
#         "mass": mass,
#         "n_valid_points": int(np.sum(valid)),
#     }


def _fit_log_linear_numpy(
    time_values: np.ndarray,
    signal_time_boot: np.ndarray,
    interval: np.ndarray,
    nt_half: int,
) -> dict[str, Any] | None:
    t = time_values[interval].astype(float)
    logs = np.log(signal_time_boot[interval])  # 假设所有信号值 > 0
    # print(logs.shape)
    # y_mean = logs.mean(axis=1)
    # y_err = logs.var(axis=1, ddof=1)
    y_mean, y_err = compute_mean_and_err(logs.T)

    valid = np.isfinite(y_mean) & (y_err > 0)
    if valid.sum() < 3:
        return None
    x = t[valid]
    y = y_mean[valid]
    w = 1.0 / y_err[valid]  # 权重 = 1/标准差
    slope, intercept = np.polyfit(x, y, deg=1, w=w)  # 线性拟合
    mass = -slope
    log_amp = (
        intercept - mass * nt_half
    )  # 注意这里应该是减去 mass * nt_half，因为 intercept = log_amp + mass * nt_half
    return {
        "interval": interval,
        "interval_start": int(interval[0]),
        "interval_end": int(interval[-1]),
        "log_amp": log_amp,
        "mass": mass,
        "n_valid_points": int(valid.sum()),
    }


def ground_linear_seed_numpy(
    samples_time_boot: np.ndarray,
    time_slices: np.ndarray,
    candidate_intervals: list[np.ndarray],
    nt_half: int = 48,
) -> dict[str, Any]:
    """Estimate ground-state seeds and select the median-mass interval."""
    candidates: list[dict[str, Any]] = []
    for interval in candidate_intervals:
        fit_result = _fit_log_linear_numpy(
            time_slices, samples_time_boot, interval, nt_half
        )
        if fit_result is not None:
            candidates.append(fit_result)

    if not candidates:
        raise ValueError("no valid ground-state intervals")

    masses = np.array([entry["mass"] for entry in candidates], dtype=float)
    order = np.argsort(masses)
    median_idx = int(order[len(order) // 2])
    selected = candidates[median_idx]
    return {
        "log_amp_init": float(selected["log_amp"]),
        "mass_init": float(selected["mass"]),
        "selected_interval": selected["interval"],
        "candidates": candidates,
    }


def refine_ground_with_lmfit(
    corr_mean: np.ndarray,
    corr_err: np.ndarray,
    time_slices: np.ndarray,
    fit_interval: np.ndarray,
    log_amp_init: float,
    mass_init: float,
    nt_half: int = 48,
) -> dict[str, Any]:
    """Refine one-state parameters on the selected interval using lmfit."""

    # 1. 提取当前拟合区间的数据
    t_fit = time_slices[fit_interval]
    data_fit = corr_mean[fit_interval]
    err_fit = corr_err[fit_interval]

    # 2. 定义残差函数 (包含误差权重)
    def residual(params, t, data, err):
        A = params["log_amp"].value
        m = params["mass"].value

        # 对应 C++ 中的 cosh_one 模型
        # model = np.exp(A + m * (t - nt_half)) + np.exp(A - m * (t - nt_half))
        model = one_state_cosh(t, A, m, nt_half)  # 使用之前定义的模型函数

        # lmfit 默认对返回的数组求平方和。
        # 返回 (数据 - 模型) / 误差 等价于 C++ 中的按方差加权的 \chi^2
        return (data - model) / err

    # 3. 设置参数与动态边界 (复刻 C++ 中的 [0.5 * val, 2 * val] 逻辑)
    params = Parameters()
    bounds_amp = sorted([0.5 * log_amp_init, 2 * log_amp_init])
    bounds_mass = sorted([0.5 * mass_init, 2 * mass_init])

    params.add("log_amp", value=log_amp_init, min=bounds_amp[0], max=bounds_amp[1])
    params.add("mass", value=mass_init, min=bounds_mass[0], max=bounds_mass[1])

    # 4. 执行 Levenberg-Marquardt 拟合
    # 如果遇到病态问题，可以在 minimize 中加入 method='least_squares' 来使用更稳健的求解器
    result = minimize(
        residual,
        params,
        args=(t_fit, data_fit, err_fit),
        method="leastsq",
    )

    # 5. 组装返回结果
    return {
        "log_amp": float(result.params["log_amp"].value),
        "mass": float(result.params["mass"].value),
        "fit_success": bool(result.success),
        "chisqr": float(result.chisqr),
        "redchi": float(result.redchi),  # 额外返回约化卡方，便于评估拟合优度
    }


def find_excited_end_time(
    corr_mean: np.ndarray,
    corr_err: np.ndarray,
    time_slices: np.ndarray,
    ground_log_amp: float,
    ground_mass: float,
    sigma_band: float = 5.0,
    end_time_ground: int | None = None,
    nt_half: int = 48,
) -> int:
    """Find end index where excited-state signal is still significant."""
    if end_time_ground is None:
        end_time_ground = len(time_slices) - 1

    ground_curve = one_state_cosh(time_slices, ground_log_amp, ground_mass, nt_half)
    end_excited = end_time_ground
    for t in range(end_time_ground + 1):
        if corr_mean[t] <= ground_curve[t] + sigma_band * corr_err[t]:
            end_excited = max(t - 1, 0)
            break
    return int(end_excited)


def excited_linear_seed_numpy(
    samples_time_boot: np.ndarray,
    time_slices: np.ndarray,
    ground_log_amp: float,
    ground_mass: float,
    end_time_excited: int,
    start_time_index: int = 4,
    nt_half: int = 48,
) -> dict[str, Any]:
    """Estimate excited-state seeds from log of ground-subtracted correlator."""
    if end_time_excited < 0:
        return {"seed": None, "candidates": []}

    ground_curve = one_state_cosh(time_slices, ground_log_amp, ground_mass, nt_half)
    diff_samples = samples_time_boot - ground_curve[:, None]

    candidate_intervals = build_intervals(
        start_time_index, end_time_excited, min_interval_length=2
    )
    candidates: list[dict[str, Any]] = []
    for interval in candidate_intervals:
        fit_result = _fit_log_linear_numpy(time_slices, diff_samples, interval, nt_half)
        if fit_result is not None:
            candidates.append(fit_result)

    if not candidates:
        return {"seed": None, "candidates": []}

    # Keep the longest valid interval for a stable default excited-state seed.
    selected = max(
        candidates,
        key=lambda item: (
            item["interval_end"] - item["interval_start"],
            -item["interval_start"],
        ),
    )
    return {"seed": selected, "candidates": candidates}


def estimate_two_state_initial_guess(
    samples_boot_time: np.ndarray,
    time_slices: np.ndarray | None = None,
    nt_half: int = 48,
    start_time_index: int = 4,
    min_interval_length: int = 6,
    relative_error_threshold: float = 0.2,
    sigma_band: float = 5.0,
) -> dict[str, Any]:
    """Estimate two-state initial parameters from bootstrap samples.

    Parameters
    ----------
    samples_boot_time:
        Array with shape (n_boot, n_time).
    """
    samples = np.asarray(samples_boot_time, dtype=float)
    if samples.ndim != 2:
        raise ValueError("samples_boot_time must have shape (n_boot, n_time)")
    if samples.shape[0] < 2:
        raise ValueError("samples_boot_time requires at least two bootstrap samples")

    n_boot, n_time = samples.shape
    if time_slices is None:
        time_slices = np.arange(n_time, dtype=float)
    else:
        time_slices = np.asarray(time_slices, dtype=float)
        if time_slices.shape != (n_time,):
            raise ValueError("time_slices must have shape (n_time,)")

    corr_mean, corr_err = compute_mean_and_err(samples)
    samples_time_boot = samples.T

    end_time_ground = find_ground_end_time(
        corr_mean, corr_err, relative_error_threshold
    )
    ground_intervals = build_intervals(
        start_time_index, end_time_ground, min_interval_length
    )
    if not ground_intervals:
        raise ValueError("no valid ground-state intervals")

    ground_linear = ground_linear_seed_numpy(
        samples_time_boot, time_slices, ground_intervals, nt_half
    )
    ground_log_amp_init = ground_linear["log_amp_init"]
    ground_mass_init = ground_linear["mass_init"]
    ground_interval = ground_linear["selected_interval"]

    diagnostics: dict[str, Any] = {
        "n_boot": int(n_boot),
        "n_time": int(n_time),
        "ground_candidate_count": int(len(ground_linear["candidates"])),
        "ground_refine_failed": False,
        "excited_fallback": False,
    }

    try:
        ground_refined = refine_ground_with_lmfit(
            corr_mean=corr_mean,
            corr_err=corr_err,
            time_slices=time_slices,
            fit_interval=ground_interval,
            log_amp_init=ground_log_amp_init,
            mass_init=ground_mass_init,
            nt_half=nt_half,
        )
        ground_log_amp = ground_refined["log_amp"]
        ground_mass = ground_refined["mass"]
    except Exception:
        ground_log_amp = ground_log_amp_init
        ground_mass = ground_mass_init
        diagnostics["ground_refine_failed"] = True

    end_time_excited = find_excited_end_time(
        corr_mean=corr_mean,
        corr_err=corr_err,
        time_slices=time_slices,
        ground_log_amp=ground_log_amp,
        ground_mass=ground_mass,
        sigma_band=sigma_band,
        end_time_ground=end_time_ground,
        nt_half=nt_half,
    )

    excited_linear = excited_linear_seed_numpy(
        samples_time_boot=samples_time_boot,
        time_slices=time_slices,
        ground_log_amp=ground_log_amp,
        ground_mass=ground_mass,
        end_time_excited=end_time_excited,
        start_time_index=start_time_index,
        nt_half=nt_half,
    )

    print(f"excited_linear candidates: {excited_linear['candidates']}")
    print(f"excited_linear seed: {excited_linear['seed']}")

    # excited_interval: np.ndarray | None = None
    # if excited_linear["seed"] is None:
    #     excited_log_amp = float(ground_log_amp - 2.0)
    #     excited_mass = float(ground_mass + 0.05 * abs(ground_mass) + 1e-4)
    #     diagnostics["excited_fallback"] = True
    # else:
    #     excited_log_amp = float(excited_linear["seed"]["log_amp"])
    #     excited_mass = float(excited_linear["seed"]["mass"])
    #     excited_interval = excited_linear["seed"]["interval"]

    # # Enforce physical ordering m2 > m1 for two-state initialization.
    # eps = 1e-6
    # if not np.isfinite(excited_mass) or excited_mass <= ground_mass:
    #     excited_mass = float(ground_mass + max(0.05 * abs(ground_mass), eps))
    # mass_gap = float(max(excited_mass - ground_mass, eps))

    # return {
    #     "ground_log_amp": float(ground_log_amp),
    #     "ground_mass": float(ground_mass),
    #     "excited_log_amp": float(excited_log_amp),
    #     "excited_mass": float(excited_mass),
    #     "mass_gap": mass_gap,
    #     "ground_interval": (
    #         None if ground_interval is None else ground_interval.tolist()
    #     ),
    #     "excited_interval": (
    #         None if excited_interval is None else excited_interval.tolist()
    #     ),
    #     "end_time_ground": int(end_time_ground),
    #     "end_time_excited": int(end_time_excited),
    #     "diagnostics": diagnostics,
    # }
