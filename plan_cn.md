## 计划标题
简化版双态初值提取程序（先保证可运行与正确性）

## 摘要
按你的约束收敛为“一个主文件 + 少量清晰函数”的实现方案：
- 非线性拟合：`lmfit`
- 线性拟合：`numpy`
- 输入：`(n_boot, n_time)`
- 不做过度优化、不做过度拆分，先把流程正确跑通并可复用。

## 实现范围
1. 只实现“初值确定”流程，不实现你后续的数据处理/全拟合/分析链条。
2. 产出双态初值与关键区间信息，便于你直接接下游。

## 文件与结构（精简）
1. 新建 `src/initial_guess.py`（单文件）
2. 仅包含以下函数（不再额外拆目录）：
- `one_state_cosh(...)`
- `two_state_cosh(...)`
- `compute_mean_and_err(...)`
- `build_intervals(...)`
- `find_ground_end_time(...)`
- `ground_linear_seed_numpy(...)`
- `refine_ground_with_lmfit(...)`
- `find_excited_end_time(...)`
- `excited_linear_seed_numpy(...)`
- `estimate_two_state_initial_guess(...)`（主入口）

## 主入口接口（固定）
`estimate_two_state_initial_guess(samples_boot_time, time_slices=None, nt_half=48, start_time_index=4, min_interval_length=6, relative_error_threshold=0.2, sigma_band=5.0) -> dict`

### 输入
- `samples_boot_time`: `np.ndarray`, shape `(n_boot, n_time)`
- `time_slices`: 可选，默认 `np.arange(n_time)`

### 输出
- `ground_log_amp`, `ground_mass`
- `excited_log_amp`, `excited_mass`
- `mass_gap`
- `ground_interval`, `excited_interval`
- `end_time_ground`, `end_time_excited`
- `diagnostics`

## 算法流程（按文档 1-6 步）
1. 计算 `mean/error`
- 内部转成 time-major 计算：`samples_tn = samples_boot_time.T`
- `corr_mean = mean(samples_tn, axis=1)`，`corr_err = std(..., ddof=1)`

2. 找基态截断 `end_time_ground`
- 首个满足 `corr_err[t] >= relative_error_threshold * corr_mean[t]` 的 `t`，取 `t-1`
- 若未触发则取 `n_time-1`

3. 基态线性初值（`numpy`）
- 在 `[start_time_index, end_time_ground]` 生成长度 `>= min_interval_length` 的连续区间
- 对每个区间做 `log(corr)` 的加权线性拟合（`numpy.polyfit` with `w`）
- `corr<=0` 设为无效点并剔除
- 按质量参数取中位数区间，得到 `ground_log_amp_init, ground_mass_init`

4. 基态单态非线性精修（`lmfit`）
- 在中位数区间拟合 `one_state_cosh`
- 得到 `ground_log_amp, ground_mass`

5. 剥离基态并确定激发态有效区
- `corr_diff = corr_mean - one_state_cosh(...)`
- 按 `corr_mean[t] <= ground_fit[t] + sigma_band * corr_err[t]` 确定 `end_time_excited`

6. 激发态线性初值（`numpy`）
- 在 `[0, end_time_excited]` 对 `log(corr_diff)` 做区间线性拟合
- 默认取“最长有效区间”结果为 `excited_log_amp, excited_mass`
- 强制 `excited_mass > ground_mass`，否则修正为 `ground_mass + eps`
- `mass_gap = excited_mass - ground_mass`（保证正值）

## 注释与命名要求
1. 全部 `snake_case`，语义明确，不沿用 C++ 缩写。
2. 每个函数加简短 docstring（输入 shape、返回含义、异常条件）。
3. 仅在关键步骤加注释：
- 为什么这样截断
- 为什么需要 log 前过滤非正值
- 为什么强制 `mass_gap > 0`

## 错误处理（先保守可运行）
1. 基态无可用区间：抛 `ValueError("no valid ground-state intervals")`
2. 激发态可用点不足：
- 启用保底初值：`excited_mass = ground_mass + 0.05*abs(ground_mass) + 1e-4`
- 在 `diagnostics["excited_fallback"]=True` 标记
3. `lmfit` 失败：
- 回退到线性初值并写入 `diagnostics["ground_refine_failed"]=True`

## 最小测试计划（不过度）
1. 新建 `tests/test_initial_guess.py`
2. 只做 4 个关键测试：
- 输入 shape `(n_boot, n_time)` 能跑通并返回完整字段
- 基态截断规则正确
- 非正值样本不会导致 `log` 崩溃
- 输出始终满足 `mass_gap > 0`

## 默认与假设
1. 你上游负责准备 bootstrap 样本，本模块不做重采样生成。
2. 默认时间窗接近参考设置（如 `n_time=49, nt_half=48`），但参数可改。
3. 当前优先“正确可运行”，暂不做并行/性能优化。
4. 在实现代码的过程中，如果细节上有不确定的地方，请参考reference中的c++代码，不要自行决定
