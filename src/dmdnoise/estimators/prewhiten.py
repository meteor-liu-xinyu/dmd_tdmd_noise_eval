"""有色噪声的预白化。

**为什么预白化在原理上可行**：信号是阻尼指数的和，
`x_t = Σ_k c_k μ_k^t`，即**移位算子的本征函数**。对它施加时间滤波
`(I − ρS)`（`S` 为移位）得

    x_t − ρ x_{t-1} = Σ_k c_k μ_k^{t-1}(μ_k − ρ)

**仍是同一组 `μ_k` 的阻尼指数和，只是幅度被重新加权**。因此
**信号子空间的维数不变**，而 AR(1) 噪声 `ε_t = ρε_{t-1} + √(1−ρ²)w_t` 恰好被白化：

    ε_t − ρε_{t-1} = √(1−ρ²) w_t   （白）

对空间相关 `ε = L z`（`LLᵀ = R`），左乘 `L^{-1}` 白化噪声，
并把信号子空间 `span(C)` 映到 `span(L^{-1}C)` —— **维数同样不变**。

于是：**在预白化后的矩阵上套用标准（白噪声）秩判据即可。**

⚠️ 实用难点不在白化本身，而在**估计 `ρ` 或 `L`**——这是本模块另一部分的内容。
"""

from __future__ import annotations

import logging
import math

import numpy as np
from numpy.typing import NDArray

LOG = logging.getLogger(__name__)


class PrewhitenError(ValueError):
    """预白化输入非法。"""


# --------------------------------------------------------------------------- 时间方向
def temporal_whiten(Z: NDArray, rho: float) -> NDArray:
    """时间白化 `Z'[:, t] = Z[:, t] − ρ·Z[:, t−1]`，列数少 1。

    信号本征值不变（见模块说明），噪声被白化。
    """
    if not 0.0 <= rho < 1.0:
        raise PrewhitenError(f"rho 必须落在 [0, 1)，收到 {rho!r}")
    Z = np.asarray(Z)
    if Z.shape[1] < 3:
        raise PrewhitenError("列数过少，无法做时间白化")
    return Z[:, 1:] - rho * Z[:, :-1]


def _lag1_median(R: NDArray) -> float:
    """矩阵 `R` 各行的 lag-1 自相关中位数（实部）。"""
    acs = []
    for row in np.asarray(R):
        a, b_ = np.real(row[:-1]), np.real(row[1:])
        if a.std() > 0 and b_.std() > 0:
            acs.append(float(np.corrcoef(a, b_)[0, 1]))
    return float(np.median(acs)) if acs else 0.0


def low_rank_residual(Z: NDArray, *, kappa: float = 3.0,
                      min_keep: int = 1) -> tuple[NDArray, int]:
    """保守低秩截断后的残差：`s_i > kappa·median(s)` 的方向算作信号。

    `kappa=3` 是刻意取**偏保守**（宁可多留噪声方向、少截信号）：
    残差里混入一点信号会低估 `ρ`，而混入噪声方向无害。
    """
    Z = np.asarray(Z)
    U, sv, Vh = np.linalg.svd(Z, full_matrices=False)
    thr = kappa * float(np.median(sv))
    k = int(np.sum(sv > thr))
    k = min(max(k, min_keep), sv.size - 2)
    R = Z - (U[:, :k] * sv[:k]) @ Vh[:k]
    return R, k


def estimate_ar1_from_residual(Z: NDArray, *, kappa: float = 3.0) -> float:
    """由**低秩残差矩阵**估计 AR(1) 系数（可实现）。

    ⚠️ **不能用正交化后的奇异向量做估计**——正交归一化会破坏原始噪声的
    时间相关结构，实测会得到 `ρ̂ ≈ 0`（真值 0.5/0.9 时全部失效）。
    必须用**残差矩阵本身**的列方向 lag-1 相关。
    """
    R, _ = low_rank_residual(Z, kappa=kappa)
    if R.shape[1] < 3:
        return 0.0
    return float(min(max(_lag1_median(R), 0.0), 0.95))


# --------------------------------------------------------------------------- 空间方向
def spatial_corr_factor(rho: float, n: int) -> NDArray:
    """指数相关矩阵 `R_ij = ρ^{|i−j|}` 的 Cholesky 因子 `L`（`LLᵀ = R`）。"""
    if not 0.0 <= rho < 1.0:
        raise PrewhitenError(f"rho 必须落在 [0, 1)，收到 {rho!r}")
    idx = np.arange(n)
    R = rho ** np.abs(idx[:, None] - idx[None, :])
    return np.linalg.cholesky(R + 1e-12 * np.eye(n))


def spatial_whiten(Z: NDArray, rho: float) -> NDArray:
    """空间白化 `Z' = L^{-1} Z`；信号子空间维数不变。"""
    n, _ = Z.shape
    L = spatial_corr_factor(rho, n)
    return np.linalg.solve(L, np.asarray(Z))


def estimate_spatial_from_residual(Z: NDArray, *, kappa: float = 3.0) -> float:
    """由低秩残差估计通道间指数相关（可实现）。

    理由与时间方向相同：必须用**残差矩阵本身**，不能用正交化的左奇异向量。
    """
    R, _ = low_rank_residual(Z, kappa=kappa)
    n = R.shape[0]
    if n < 3:
        return 0.0
    ests = []
    for i in range(n - 1):
        a_, b_ = np.real(R[i]), np.real(R[i + 1])
        if a_.std() > 0 and b_.std() > 0:
            ests.append(abs(float(np.corrcoef(a_, b_)[0, 1])))
    return float(min(max(np.median(ests) if ests else 0.0, 0.0), 0.95))


# --------------------------------------------------------------------------- 门面
def whiten(Z: NDArray, *, ar1: float = 0.0, spatial: float = 0.0) -> NDArray:
    """按给定系数做预白化（先空间后时间，与噪声生成顺序相反）。"""
    out = np.asarray(Z)
    if spatial > 0.0:
        out = spatial_whiten(out, spatial)
    if ar1 > 0.0:
        out = temporal_whiten(out, ar1)
    return out


def whiten_estimated(Z: NDArray, *, use_spatial: bool = False,
                     kappa: float = 3.0) -> tuple[NDArray, float]:
    """用**从数据估计**的系数做预白化；返回 `(Z_白化, rhô)`。"""
    rho = estimate_ar1_from_residual(Z, kappa=kappa)
    if not use_spatial:
        return temporal_whiten(Z, rho), rho
    rho_s = estimate_spatial_from_residual(Z, kappa=kappa)
    return whiten(Z, ar1=rho, spatial=rho_s), rho
