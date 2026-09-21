"""⚠️ **实验性模块 —— 未能做通，不推荐使用。** 保留以记录失败的结构。

**目标**：解决预白化在极高空间相关（`ρ=0.9`）下的失效——那里 `L^{-1}` 条件数过大
（`cond ≈ 13.7`），把弱方向的噪声放大到单位方差、破坏信号的相对突出度。

**思路**：不改变数据，而是按噪声协方差**仿真噪声矩阵**，用其奇异值谱定阈值。

**实测结论（复值 `n=8`、SNR=10 dB、`m=200`、`J=200`，真值 `r=2`）**：

| 阈值形式 | 白噪声 | `ar1-0.5` | `spatial-0.5` | `spatial-0.9` |
|---|---|---|---|---|
| **单一阈值**（噪声 top 的 99 分位） | 1.000 | **0.000** | 1.000 | **1.000** |
| **逐阶阈值**（第 `i` 阶用噪声第 `i` 阶的分位） | **0.060** | 0.000 | 0.000 | 0.000 |

（对照：**预白化**在 `ar1-0.5/0.9` 与 `spatial-0.5` 下为 **1.000**，仅 `spatial-0.9` 为 0.000）

**两种形式的已识别缺陷**：

1. **单一阈值**：只卡住了噪声的 top，而**有色噪声的 top2/top3 与该阈值同量级**
   （时间相关使噪声谱顶部变"平"），它们会集体超阈 → 秩高估（实测 `ar1-0.5` 下
   `r̂` 稳定偏到 3–4）。
2. **逐阶阈值**：存在**索引错位**——观测的第 `i` 个奇异值（`i > r`）对应的是噪声的
   第 `i − r` 个，而不是第 `i` 个。因 `q` 与 `sv` 均降序，比较 `sv[i] > q[i]` 会
   **系统性偏向"超阈"** → 连白噪声都被判到 `r̂ = 2…11`（判对率仅 0.060）。
   正确的对齐依赖未知的 `r`，形成循环。

**已排除的假设**：不是 σ̂ 的偏差所致——信号第 2 奇异值与阈值的**余量为 248%–393%**，
而 σ̂ 的偏差仅 0.7%–4%，两者差两个数量级（见 `tests/test_spectral_rank.py`）。

**结论**：**"极高空间相关的秩判据"仍是开放问题。**
实际可用的替代路径是**增加通道数**（§3 区域 A/B），或改用基于实测噪声协方差的正则化白化。

---

"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from dmdnoise.estimators.rank import (
    RankError,
    estimate_rank,
    residual_sigma,
)
from dmdnoise.sim.noise import _colored_noise

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpectralThreshold:
    """一次谱标定的结果。"""

    rank: int
    threshold: float
    sigma_hat: float
    quantile: float
    n_sim: int
    noise: dict

    def as_dict(self) -> dict:
        return {
            "rank": self.rank, "threshold": self.threshold,
            "sigma_hat": self.sigma_hat, "quantile": self.quantile,
            "n_sim": self.n_sim, **{f"noise_{k}": v for k, v in self.noise.items()},
        }


@functools.lru_cache(maxsize=256)
def noise_spectrum_quantiles(n: int, m: int, *, complex_: bool = True,
                             ar1: float = 0.0, spatial: float = 0.0,
                             n_sim: int = 200, quantile: float = 0.99,
                             seed: int = 0) -> tuple[float, ...]:
    """噪声矩阵**逐阶**奇异值的 `quantile` 分位（降序，长度 `min(n,m)`）。

    ⚠️ **必须逐阶，不能用单一阈值。** 白噪声的 MP 谱是"平"的，单阈值即够；
    但**有色噪声的谱不平**——时间相关会使噪声的**多个**大奇异值都显著高于中位，
    此时"只看 top 的单一阈值"会被这些集体超阈的噪声方向击穿（实测 `ar1-0.5`
    下 `r̂` 稳定偏到 3–4）。

    ⚠️ **必须缓存**：阈值只依赖噪声配置、不依赖具体实现，而每次标定要做
    `n_sim` 次 SVD。不缓存则蒙特卡洛实验无法进行。
    """
    gen = np.random.default_rng(seed)
    rows = []
    for k in range(n_sim):
        N = _colored_noise((n, m), 1.0, gen, complex_, ar1=ar1, spatial=spatial)
        rows.append(np.linalg.svd(N, compute_uv=False))
    arr = np.asarray(rows)                      # (n_sim, min(n,m))
    return tuple(float(v) for v in np.quantile(arr, quantile, axis=0))


def spectral_rank(Z: NDArray, *, ar1: float = 0.0, spatial: float = 0.0,
                  rank_hint: int = 2, quantile: float = 0.99,
                  n_sim: int = 200, seed: int = 0) -> SpectralThreshold:
    """按噪声协方差标定阈值后定秩。

    `rank_hint` 仅用于估计 `σ̂`（取残差时的截断秩），不影响阈值本身。
    `quantile` 取 0.99 —— 即允许 1% 的误把噪声算作信号的概率。
    """
    if not 0.5 < quantile < 1.0:
        raise RankError("quantile 必须落在 (0.5, 1)")
    Z = np.asarray(Z)
    n, m = Z.shape
    complex_ = np.iscomplexobj(Z)

    sv = np.linalg.svd(Z, compute_uv=False)
    k = int(min(max(rank_hint, 1), sv.size - 2))
    resid = float(np.linalg.norm(sv[k:]))
    sigma_hat = residual_sigma(resid, n, m, rank=k)

    q = noise_spectrum_quantiles(n, m, complex_=complex_, ar1=ar1,
                                 spatial=spatial, n_sim=n_sim,
                                 quantile=quantile, seed=seed)
    # **连续条件**：遇到第一个未超阈的阶就停，避免"信号弱 + 噪声偶然强"打出空洞
    rank = 0
    for i in range(min(sv.size, len(q))):
        if sv[i] > q[i] * sigma_hat:
            rank = i + 1
        else:
            break
    thr = q[rank] * sigma_hat if rank < len(q) else float(q[-1]) * sigma_hat
    return SpectralThreshold(rank=rank, threshold=thr, sigma_hat=sigma_hat,
                             quantile=quantile, n_sim=n_sim,
                             noise={"ar1": ar1, "spatial": spatial})


def spectral_rank_estimated(Z: NDArray, *, kappa: float = 3.0,
                            rank_hint: int | None = None,
                            quantile: float = 0.99, n_sim: int = 200,
                            seed: int = 0) -> SpectralThreshold:
    """**盲**版本：噪声系数由 `prewhiten` 的残差法估计，再标定阈值。"""
    from dmdnoise.estimators.prewhiten import (
        estimate_ar1_from_residual,
        estimate_spatial_from_residual,
    )

    Z = np.asarray(Z)
    ar1 = estimate_ar1_from_residual(Z, kappa=kappa)
    spatial = estimate_spatial_from_residual(Z, kappa=kappa)
    hint = rank_hint if rank_hint is not None else estimate_rank(
        np.linalg.svd(Z, compute_uv=False), sigma_hat=1.0, m=Z.shape[0],
        n=Z.shape[1], method="gavish_donoho").rank
    hint = min(max(int(hint), 1), min(Z.shape) - 2)
    return spectral_rank(Z, ar1=ar1, spatial=spatial, rank_hint=hint,
                         quantile=quantile, n_sim=n_sim, seed=seed)
