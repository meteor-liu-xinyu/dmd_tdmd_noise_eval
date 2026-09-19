"""指标与统计推断。

规格依据：docs/algorithm-spec.md §6、§6.1、§6.2、§7.1。

核心规则（必须遵守）：
  - 检测下限 delta_min = 1.96 * std / sqrt(J)
  - 可分辨判据 |bias| / std > 3.92 / sqrt(J)
  - 不满足可分辨判据时**只报上界**，不得报告点估计、不得计算"降偏百分比"
  - std 及其比值的置信区间用**配对 bootstrap**（按实现索引整体重采样）
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

#: 相对分辨阈值系数 sqrt(15.37) 的来源：J > 15.37*(std/|bias|)^2
RESOLVE_COEF = 3.92


class MetricsError(ValueError):
    """指标计算的输入非法。"""


@dataclass
class Summary:
    """单组估计误差（f_hat - f_true）的偏差-方差摘要。"""

    n: int
    bias: float
    std: float
    rmse: float
    mse_decomp: float          # bias^2 + std^2（规格定义）
    mse_plugin: float          # mean(delta^2)（同一总体的无偏形式）
    delta_min: float           # 检测下限
    resolvable: bool           # |bias| > delta_min
    required_j: float          # 达 50% 相对分辨所需 J
    bias_ci: tuple[float, float] = (math.nan, math.nan)
    std_ci: tuple[float, float] = (math.nan, math.nan)
    bias_std_ratio: float = math.nan
    note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def bias_upper(self) -> float:
        """偏差不显著时的上界（用于"只报上界"的表述）。"""
        return self.delta_min

    def as_row(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "bias": self.bias,
            "std": self.std,
            "rmse": self.rmse,
            "mse_decomp": self.mse_decomp,
            "mse_plugin": self.mse_plugin,
            "delta_min": self.delta_min,
            "resolvable": self.resolvable,
            "required_j": self.required_j,
            "bias_ci_lo": self.bias_ci[0],
            "bias_ci_hi": self.bias_ci[1],
            "std_ci_lo": self.std_ci[0],
            "std_ci_hi": self.std_ci[1],
            "bias_std_ratio": self.bias_std_ratio,
            "note": self.note,
        }


def required_j(bias: float, std: float, *, frac: float = 0.5) -> float:
    """达 `frac` 相对分辨所需实现次数： J = (1.96/frac)^2 * (std/bias)^2。"""
    if bias == 0.0 or not np.isfinite(bias):
        return math.inf
    if std == 0.0:
        return 1.0
    return (1.96 / frac) ** 2 * (std / bias) ** 2


def summarize(delta: NDArray, *, alpha: float = 0.05, bootstrap: int = 10000,
              rng: np.random.Generator | None = None,
              normalize: float | None = None) -> Summary:
    """对误差样本 delta_j = f_hat_j - f_true 做偏差-方差分解。

    normalize 给定时，所有输出量除以该值（例如按真频率归一化）。
    """
    d = np.asarray(delta, dtype=float).ravel()
    if d.size < 2:
        raise MetricsError("至少需要 2 个样本")
    if not np.all(np.isfinite(d)):
        raise MetricsError("delta 含非有限值，配对失败样本应先剔除")

    scale = 1.0 if normalize is None else float(normalize)
    if scale == 0.0:
        raise MetricsError("normalize 不得为零")
    d = d / scale

    n = d.size
    bias = float(d.mean())
    std = float(d.std(ddof=1))
    mse_plugin = float(np.mean(d**2))
    mse_decomp = bias**2 + std**2
    half = 1.96 * std / math.sqrt(n)
    delta_min = half
    ratio = abs(bias) / std if std > 0 else math.inf
    resolve_threshold = RESOLVE_COEF / math.sqrt(n)
    ok = bool(ratio > resolve_threshold)

    note = ""
    if not ok:
        note = (f"偏差不可分辨：|bias|/std={ratio:.4g} <= {resolve_threshold:.4g}；"
                f"只报上界 delta_min={delta_min:.4g}")

    summary = Summary(
        n=n,
        bias=bias,
        std=std,
        rmse=math.sqrt(mse_decomp),
        mse_decomp=mse_decomp,
        mse_plugin=mse_plugin,
        delta_min=delta_min,
        resolvable=ok,
        required_j=required_j(bias, std),
        bias_ci=(bias - half, bias + half),
        bias_std_ratio=ratio,
        note=note,
        extra={"alpha": alpha, "scale": scale},
    )

    if bootstrap > 0:
        gen = rng if rng is not None else np.random.default_rng(0)
        pos = gen.integers(0, n, size=(bootstrap, n))
        boots = d[pos]
        b_bias = boots.mean(axis=1)
        b_std = boots.std(axis=1, ddof=1)
        lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
        summary.bias_ci = (float(np.percentile(b_bias, lo)), float(np.percentile(b_bias, hi)))
        summary.std_ci = (float(np.percentile(b_std, lo)), float(np.percentile(b_std, hi)))
        summary.extra["bootstrap_resamples"] = bootstrap
    return summary


@dataclass
class PairedRatio:
    """两法配对比值（例如 std_TDMD / std_DMD）。"""

    point: float
    ci: tuple[float, float]
    n: int
    var_delta: float
    var_sum: float
    variance_reduction: float
    paired_effective: bool
    note: str = ""

    def as_row(self) -> dict[str, Any]:
        return {
            "ratio": self.point,
            "ratio_ci_lo": self.ci[0],
            "ratio_ci_hi": self.ci[1],
            "n_pair": self.n,
            "var_delta": self.var_delta,
            "var_sum": self.var_sum,
            "variance_reduction": self.variance_reduction,
            "paired_effective": self.paired_effective,
        }


def _std_stat(x: NDArray) -> float:
    return float(x.std(ddof=1))


def _mse_stat(x: NDArray) -> float:
    return float(np.mean(x**2))


def paired_ratio(a: NDArray, b: NDArray, *, stat: str = "std", bootstrap: int = 10000,
                 alpha: float = 0.05, rng: np.random.Generator | None = None,
                 reduction_threshold: float = 0.5) -> PairedRatio:
    """配对 bootstrap 的两法比值。

    ⚠️ 重采样必须按**实现索引整体**进行（同一索引同时作用于 a 与 b），
       否则会破坏公共随机数带来的协方差结构，使 CI 偏宽。

    同时返回配对差的自检量（规格 §6.2）：
      Var(delta) 与 Var(a)+Var(b) 的对比，用于验证公共随机数设计确实生效。
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size != b.size:
        raise MetricsError(f"配对样本长度不等：{a.size} 与 {b.size}")
    n = a.size
    if n < 2:
        raise MetricsError("至少需要 2 对样本")

    fn: Callable[[NDArray], float] = _std_stat if stat == "std" else _mse_stat
    denom = fn(b)
    if denom == 0.0:
        raise MetricsError("分母统计量为零")
    point = fn(a) / denom

    gen = rng if rng is not None else np.random.default_rng(0)
    pos = gen.integers(0, n, size=(bootstrap, n))
    ab = a[pos]
    bb = b[pos]
    num = ab.std(axis=1, ddof=1) if stat == "std" else np.mean(ab**2, axis=1)
    den = bb.std(axis=1, ddof=1) if stat == "std" else np.mean(bb**2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = num / den
    ratios = ratios[np.isfinite(ratios)]
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    ci = (float(np.percentile(ratios, lo)), float(np.percentile(ratios, hi)))

    var_delta = float(np.var(a - b, ddof=1))
    var_sum = float(np.var(a, ddof=1) + np.var(b, ddof=1))
    eff = var_delta < reduction_threshold * var_sum
    note = "" if eff else "公共随机数未产生预期的方差消减，需排查配对实现"

    return PairedRatio(point, ci, n, var_delta, var_sum,
                       (1.0 - var_delta / var_sum) if var_sum > 0 else math.nan, eff, note)
