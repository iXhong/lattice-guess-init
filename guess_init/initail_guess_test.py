"""Initial parameter estimation for two-state correlator fits.

Public input shape is (n_boot, n_time). Internally this module transposes to
time-major (n_time, n_boot) for clearer vectorized operations.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from lmfit import Parameters, minimize

# 假设您在 src.config 中定义了这些常量
from guess_init.config import NUMBER_BOOTSTRAP, NUMBER_CONF


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
    """Return bootstrap mean and error over the bootstrap axis."""
    samples = np.asarray(samples_boot_time, dtype=float)
    if samples.ndim != 2:
        raise ValueError(
            "samples_boot_time must be a 2D array of shape (n_boot, n_time)"
        )
    if samples.shape[0] < 2:
        raise ValueError("samples_boot_time requires at least two bootstrap samples")
    samples_time_boot = samples.T
    corr_mean = np.mean(samples_time_boot, axis=1)

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
        if corr_err[t] >= relative_error_threshold * abs(corr_mean[t]):
            return max(t - 1, 0)
    return n_time - 1


def _fit_log_linear_numpy(
    time_values: np.ndarray,
    signal_time_boot: np.ndarray,
    interval: np.ndarray,
    nt_half: int,
) -> dict[str, Any] | None:
    t = time_values[interval].astype(float)
    # 遮蔽非正值，防止取对数报错 (也适用于扣除基态后可能出现的负值)
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.where(
            signal_time_boot[interval] > 0.0, np.log(signal_time_boot[interval]), np.nan
        )

    y_mean = np.nanmean(logs, axis=1)
    y_err = np.nanstd(logs, axis=1, ddof=1)

    valid = np.isfinite(y_mean) & (y_err > 0)
    if valid.sum() < 3:
        return None

    x = t[valid]
    y = y_mean[valid]
    w = 1.0 / y_err[valid]  # 权重 = 1/标准差

    try:
        slope, intercept = np.polyfit(x, y, deg=1, w=w)  # 线性拟合
    except Exception:
        return None

    mass = -slope
    # 修复了符号问题：intercept = log_amp + mass * nt_half
    log_amp = intercept - mass * nt_half

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
    t_fit = time_slices[fit_interval]
    data_fit = corr_mean[fit_interval]
    err_fit = corr_err[fit_interval]

    def residual(params, t, data, err):
        A = params["log_amp"].value
        m = params["mass"].value
        model = one_state_cosh(t, A, m, nt_half)
        return (data - model) / err

    params = Parameters()
    # 修复边界问题，适配负值 initial_guess
    bounds_amp = sorted([0.5 * log_amp_init, 2.0 * log_amp_init])
    bounds_mass = sorted([0.5 * mass_init, 2.0 * mass_init])

    params.add("log_amp", value=log_amp_init, min=bounds_amp[0], max=bounds_amp[1])
    params.add("mass", value=mass_init, min=bounds_mass[0], max=bounds_mass[1])

    result = minimize(
        residual,
        params,
        args=(t_fit, data_fit, err_fit),
        method="least_squares",
    )

    return {
        "log_amp": float(result.params["log_amp"].value),
        "mass": float(result.params["mass"].value),
        "fit_success": bool(result.success),
        "chisqr": float(result.chisqr),
        "redchi": float(result.redchi),
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
) -> dict[int, dict[str, Any]]:
    """
    Estimate excited-state seeds from log of ground-subtracted correlator.
    返回字典，键为区间起始时间 t_start，值为对应的线性拟合结果。
    """
    if end_time_excited < 0:
        return {}

    ground_curve = one_state_cosh(time_slices, ground_log_amp, ground_mass, nt_half)
    diff_samples = samples_time_boot - ground_curve[:, None]

    candidates: dict[int, dict[str, Any]] = {}
    for t_start in range(start_time_index, end_time_excited):
        interval = np.arange(t_start, end_time_excited + 1)
        if len(interval) >= 3:  # 保证足够的数据点进行带误差线性拟合
            fit_result = _fit_log_linear_numpy(
                time_slices, diff_samples, interval, nt_half
            )
            if fit_result is not None:
                candidates[t_start] = fit_result

    return candidates


def refine_two_state_with_lmfit(
    corr_mean: np.ndarray,
    corr_err: np.ndarray,
    time_slices: np.ndarray,
    fit_interval: np.ndarray,
    log_amp_1_init: float,
    mass_1_init: float,
    log_amp_2_init: float,
    mass_gap_init: float,
    nt_half: int = 48,
) -> dict[str, Any]:
    """
    Refine two-state parameters using lmfit, following a 2-step strategy:
    Step 5: Fix state 1, fit state 2.
    Step 6: Free all parameters for a full two-state fit.
    """
    t_fit = time_slices[fit_interval]
    data_fit = corr_mean[fit_interval]
    err_fit = corr_err[fit_interval]

    def residual(params, t, data, err):
        A1 = params["log_amp_1"].value
        m1 = params["mass_1"].value
        A2 = params["log_amp_2"].value
        dm = params["mass_gap"].value
        model = two_state_cosh(t, A1, m1, A2, dm, nt_half)
        return (data - model) / err

    # --- Step 5: Fix ground state, fit excited state ---
    bounds_A2 = sorted([0.5 * log_amp_2_init, 2.0 * log_amp_2_init])
    bounds_dm = sorted([0.5 * mass_gap_init, 2.0 * mass_gap_init])

    params_step5 = Parameters()
    params_step5.add("log_amp_1", value=log_amp_1_init, vary=False)
    params_step5.add("mass_1", value=mass_1_init, vary=False)
    params_step5.add(
        "log_amp_2", value=log_amp_2_init, min=bounds_A2[0], max=bounds_A2[1], vary=True
    )
    params_step5.add(
        "mass_gap", value=mass_gap_init, min=bounds_dm[0], max=bounds_dm[1], vary=True
    )

    result_step5 = minimize(
        residual, params_step5, args=(t_fit, data_fit, err_fit), method="least_squares"
    )

    A2_refined = result_step5.params["log_amp_2"].value
    dm_refined = result_step5.params["mass_gap"].value

    # --- Step 6: Full two-state fit ---
    bounds_A1 = sorted([0.5 * log_amp_1_init, 2.0 * log_amp_1_init])
    bounds_m1 = sorted([0.5 * mass_1_init, 2.0 * mass_1_init])
    bounds_A2_new = sorted([0.5 * A2_refined, 2.0 * A2_refined])
    bounds_dm_new = sorted([0.5 * dm_refined, 2.0 * dm_refined])

    params_step6 = Parameters()
    params_step6.add(
        "log_amp_1", value=log_amp_1_init, min=bounds_A1[0], max=bounds_A1[1]
    )
    params_step6.add("mass_1", value=mass_1_init, min=bounds_m1[0], max=bounds_m1[1])
    params_step6.add(
        "log_amp_2", value=A2_refined, min=bounds_A2_new[0], max=bounds_A2_new[1]
    )
    params_step6.add(
        "mass_gap", value=dm_refined, min=bounds_dm_new[0], max=bounds_dm_new[1]
    )

    result_step6 = minimize(
        residual, params_step6, args=(t_fit, data_fit, err_fit), method="least_squares"
    )

    mass_1_final = float(result_step6.params["mass_1"].value)
    mass_gap_final = float(result_step6.params["mass_gap"].value)

    return {
        "log_amp_1": float(result_step6.params["log_amp_1"].value),
        "mass_1": mass_1_final,
        "log_amp_2": float(result_step6.params["log_amp_2"].value),
        "mass_gap": mass_gap_final,
        "mass_2": mass_1_final + mass_gap_final,
        "fit_success": bool(result_step6.success),
        "chisqr": float(result_step6.chisqr),
        "redchi": float(result_step6.redchi),
    }


def estimate_two_state_initial_guess(
    samples_boot_time: np.ndarray,
    time_slices: np.ndarray | None = None,
    nt_half: int = 48,
    start_time_index: int = 4,
    min_interval_length: int = 6,
    relative_error_threshold: float = 0.2,
    sigma_band: float = 5.0,
) -> dict[str, Any]:
    """Estimate two-state parameters. Performs interval-by-interval fitting identical to C++ logic."""
    samples = np.asarray(samples_boot_time, dtype=float)
    if samples.ndim != 2:
        raise ValueError("samples_boot_time must have shape (n_boot, n_time)")

    n_boot, n_time = samples.shape
    if time_slices is None:
        time_slices = np.arange(n_time, dtype=float)

    corr_mean, corr_err = compute_mean_and_err(samples)
    samples_time_boot = samples.T

    # --- 1. 获取基态基准 (Median) ---
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
    median_A1_init = ground_linear["log_amp_init"]
    median_m1_init = ground_linear["mass_init"]
    median_ground_inv = ground_linear["selected_interval"]

    try:
        ground_refined = refine_ground_with_lmfit(
            corr_mean,
            corr_err,
            time_slices,
            median_ground_inv,
            median_A1_init,
            median_m1_init,
            nt_half,
        )
        median_A1 = ground_refined["log_amp"]
        median_m1 = ground_refined["mass"]
    except Exception:
        median_A1, median_m1 = median_A1_init, median_m1_init

    # --- 2. 获取激发态区间与初值 (C++ 扣除基态过程) ---
    end_time_excited = find_excited_end_time(
        corr_mean,
        corr_err,
        time_slices,
        median_A1,
        median_m1,
        sigma_band,
        end_time_ground,
        nt_half,
    )

    # 获得激发态线性初值字典 {t_start: fit_result}
    excited_linears = excited_linear_seed_numpy(
        samples_time_boot,
        time_slices,
        median_A1,
        median_m1,
        end_time_excited,
        start_time_index,
        nt_half,
    )

    # 对所有有效的激发态区间进行一态非线性提炼 (C++ 逻辑: params_two_init 单态拟合)
    diff_mean = corr_mean - one_state_cosh(time_slices, median_A1, median_m1, nt_half)
    excited_refined: dict[int, dict[str, Any]] = {}
    for t_start, lin_res in excited_linears.items():
        try:
            # 同样使用 refine_ground_with_lmfit 进行优化，但是传入的是扣除基态后的 diff_mean
            ref_res = refine_ground_with_lmfit(
                diff_mean,
                corr_err,
                time_slices,
                lin_res["interval"],
                lin_res["log_amp"],
                lin_res["mass"],
                nt_half,
            )
            excited_refined[t_start] = ref_res
        except Exception:
            excited_refined[t_start] = lin_res  # 降级使用线性参数

    # --- 3. 全局扫描所有的基态区间并进行双态优化 ---
    all_intervals_results = []

    for g_inv in ground_intervals:
        t_start = int(g_inv[0])

        # A) 局部一态微调 (C++ 在进入两态前会对当前区间再做一次一态微调)
        try:
            local_ground = refine_ground_with_lmfit(
                corr_mean, corr_err, time_slices, g_inv, median_A1, median_m1, nt_half
            )
            local_A1 = local_ground["log_amp"]
            local_m1 = local_ground["mass"]
        except Exception:
            local_A1, local_m1 = median_A1, median_m1

        # B) 根据区间起始点，分配激发态初值 (C++ 中的 fallback 逻辑)
        if t_start in excited_refined:
            ex_res = excited_refined[t_start]
            A2_init = ex_res["log_amp"]
            m2_init = ex_res["mass"]
            mass_gap = max(m2_init - local_m1, 1e-6)
        else:
            # 手动设定的 fallback: C++中的 A2 = 5*A1, dm = m1
            A2_init = 5.0 * local_A1
            mass_gap = max(local_m1, 1e-6)

        # C) 执行当前区间的双态全局拟合
        two_state_res = None
        success = False
        try:
            two_state_res = refine_two_state_with_lmfit(
                corr_mean,
                corr_err,
                time_slices,
                g_inv,
                local_A1,
                local_m1,
                A2_init,
                mass_gap,
                nt_half,
            )
            success = two_state_res["fit_success"]
        except Exception:
            pass

        # 记录该区间的全套结果
        all_intervals_results.append(
            {
                "interval_start": t_start,
                "interval_end": int(g_inv[-1]),
                "interval": g_inv.tolist(),
                "local_ground_log_amp": float(local_A1),
                "local_ground_mass": float(local_m1),
                "guess_excited_log_amp": float(A2_init),
                "guess_mass_gap": float(mass_gap),
                "two_state_fit_success": success,
                "two_state_result": two_state_res,
            }
        )

    # 返回最终综合结果字典
    return {
        "end_time_ground": int(end_time_ground),
        "end_time_excited": int(end_time_excited),
        "median_ground_log_amp": float(median_A1),
        "median_ground_mass": float(median_m1),
        "all_intervals_results": all_intervals_results,
    }
