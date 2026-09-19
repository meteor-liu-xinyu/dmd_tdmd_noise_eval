"""频率提取、模态配对与缠绕检测。

规格依据：docs/algorithm-spec.md §4.3。

配对规则：估计频率升序后与真值升序按索引配对。
模态间隔由设计约束 C2 保证（相对间隔 >= 0.25），故排序配对在实验噪声范围内稳定；
仍须记录配对失败计数，配错样本从统计中剔除并单独报告比例。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

#: 配对判据：|f_hat_i - f_true_i| 超过该比例的模态间隔即判为配错
MISPAIR_FRAC = 0.4


@dataclass(frozen=True)
class PairResult:
    ok: bool
    freqs: NDArray                    # 与 f_true 同序；ok=False 时为空数组
    reason: str = ""
    max_dev: float = float("nan")


def truth_separation(f_true: NDArray) -> float:
    """真值相邻模态的最小间隔（Hz）。"""
    f = np.sort(np.asarray(f_true, dtype=float))
    if f.size < 2:
        return float("inf")
    return float(np.min(np.diff(f)))


def distinct_freqs(freqs: NDArray, *, rel_tol: float = 0.05) -> NDArray:
    """合并近邻频率，返回升序的互异频率。

    实值通道的 r=4 个离散特征值包含两组共轭对，其 |angle| 相同，
    因此估计结果会出现成对重复的频率；此处按相对容差合并。
    容差取 0.05 远小于模态间隔（设计约束 C2 保证 >= 0.25），不会误并真实模态。
    """
    f = np.sort(np.abs(np.asarray(freqs, dtype=float)).ravel())
    out: list[float] = []
    for x in f:
        if not out or (x - out[-1]) > rel_tol * max(x, 1e-12):
            out.append(float(x))
    return np.asarray(out, dtype=float)


def pair_to_truth(f_hat: NDArray, f_true: NDArray, *,
                  mispair_frac: float = MISPAIR_FRAC,
                  rel_tol: float = 0.05) -> PairResult:
    """把估计频率升序配对到真值。

    先合并共轭重复（实值通道），再按索引配对。

    返回 ok=False 的情形：
      - 估计模态数不足
      - 最大偏差超过 mispair_frac * 模态间隔（判为配错，通常意味着缠绕合并）
    """
    fh = distinct_freqs(f_hat, rel_tol=rel_tol)
    ft = np.sort(np.asarray(f_true, dtype=float).ravel())

    if fh.size < ft.size:
        return PairResult(False, np.empty(0), f"模态数不足：{fh.size} < {ft.size}")

    cand = fh[: ft.size]
    dev = np.abs(cand - ft)
    sep = truth_separation(ft)
    limit = mispair_frac * sep
    worst = float(dev.max()) if dev.size else 0.0
    if np.isfinite(limit) and worst > limit:
        return PairResult(False, np.empty(0), f"配错：max|df|={worst:.4g} > {limit:.4g}", worst)
    return PairResult(True, cand, "", worst)


def unwrap_flags(freqs: NDArray, dt: float, *, frac: float = 0.9) -> NDArray:
    """标记接近 Nyquist 的估计（可能发生 angle 缠绕）。"""
    fs = 1.0 / dt
    return np.abs(np.asarray(freqs)) > frac * fs / 2.0


def relative_error(f_hat: NDArray, f_true: NDArray) -> NDArray:
    return (np.asarray(f_hat) - np.asarray(f_true)) / np.asarray(f_true)
