"""TDMD 的**错误读法**变体 —— 仅供测试作为负对照使用。

⚠️ 本模块不得在实验中被调用。它是判据 T11 的负对照素材。

背景（docs/algorithm-spec.md §8.2）：无噪退化判据 T1–T3 **无法**拦截投影子空间取错，
因此 T11（含噪去偏有效性）是唯一的判别手段。要用 T11 标定阈值，就必须有
"确实会失败的实现"作为负对照。本模块提供两个候选：

  tdmd_left_star   A = U1* U2        —— 实测与正确版的频率差可达 0.064 Hz，
                                       在原文线性系统算例上偏差是正确版的 12 倍。
                                       **有效的负对照。**
  tdmd_left_pinv   A = (U1*U1)^-1 U1* U2
                                     —— 实测在 r=2 下与正确版**特征值完全相同**
                                       （10/10 样本，差 ~1e-15）。它是个等价变形，
                                       **不能**充当负对照——这一点曾导致误判。

  tdmd_no_augment  A = 标准 DMD      —— "遗漏增广矩阵"这一最常见错误形态，
                                       等价于 T11 完全无降偏效果。

其中 U1 = Ũ[:n, :r], U2 = Ũ[n:, :r]，Ũ 为增广矩阵 Z = [X;Y] 的左奇异向量。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from dmdnoise.estimators.base import (
    EstimateResult,
    Diagnostic,
    EstimatorError,
    check_nyquist,
    freqs_from_eigenvalues,
    order_by_frequency,
)
from dmdnoise.estimators.dmd import DMD

VARIANT_NAMES = ("tdmd_left_star", "tdmd_left_pinv", "tdmd_no_augment")


def _finish(A: NDArray, dt: float, n: int, m: int, tag: str) -> EstimateResult:
    evals = np.linalg.eigvals(A)
    idx = order_by_frequency(evals, dt)
    evals = evals[idx]
    freqs = freqs_from_eigenvalues(evals, dt)
    _, W = np.linalg.eig(A)
    modes = W[:, idx]
    return EstimateResult(
        freqs=freqs,
        eigenvalues=evals,
        modes=np.asarray(modes),
        diagnostics=Diagnostic(
            rank_used=A.shape[0],
            augmented=True,
            z_shape=(2 * n, m),
            near_nyquist=check_nyquist(freqs, dt),
            extra={"method": tag, "negative_control": True},
        ),
    )


def tdmd_left_star(X: NDArray, Y: NDArray, r: int, dt: float) -> EstimateResult:
    """错误读法：用 Z 的左奇异向量上下块，取 U1* U2。"""
    n, m = X.shape
    Z = np.vstack([X, Y])
    Uz, _, _ = np.linalg.svd(Z, full_matrices=False)
    U1, U2 = Uz[:n, :r], Uz[n:, :r]
    return _finish(U1.conj().T @ U2, dt, n, m, "tdmd_left_star")


def tdmd_left_pinv(X: NDArray, Y: NDArray, r: int, dt: float) -> EstimateResult:
    """错误读法：U1^+ U2。**实测与正确版特征值等价**，不可作负对照。"""
    n, m = X.shape
    Z = np.vstack([X, Y])
    Uz, _, _ = np.linalg.svd(Z, full_matrices=False)
    U1, U2 = Uz[:n, :r], Uz[n:, :r]
    return _finish(np.linalg.pinv(U1) @ U2, dt, n, m, "tdmd_left_pinv")


class TdmdNoAugment:
    """错误形态：遗漏增广矩阵，退化为标准 DMD。"""

    name = "tdmd_no_augment"

    def fit(self, X: NDArray, Y: NDArray, r: int, dt: float) -> EstimateResult:
        if X.shape != Y.shape:
            raise EstimatorError("X 与 Y 形状必须一致")
        res = DMD().fit(X, Y, r, dt)
        res.diagnostics.extra["negative_control"] = True
        res.diagnostics.extra["method"] = "tdmd_no_augment"
        return res
