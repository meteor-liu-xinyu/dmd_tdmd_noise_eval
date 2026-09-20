"""秩估计器：`r̂ = R(σ̂, Z)`。实验三（方案 3A）的核心。

规格依据：docs/algorithm-spec.md §10。

三个候选判据的 σ̂ 依赖关系**不同**，这本身是实验三的关键对照：

    energy             依赖 σ̂（阈值含 (σ̂/s_1)²）
    marchenko_pastur   依赖 σ̂（噪声地板上沿 ∝ σ̂）
    gavish_donoho      **不依赖 σ̂**（阈值由奇异值中位数自标定）

因此 GD 是"免标定"判据——它对 σ̂ 的误估天然免疫。这一点把实验三的
问题从"σ̂ 估不准会怎样"细化为"**哪些秩判据会受影响、哪些不会**"。

> ⚠️ 规格 §10.3 原写 `τ = σ̂·ω(β)·median(s)`，量纲不自洽（`σ̂ × s` 为 `s²`）。
> 已按原文更正为 `τ = ω(β)·median(s)`（Gavish & Donoho 2014 的原始形式）。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

LOG_RANK = logging.getLogger(__name__)

VALID_METHODS = ("energy", "marchenko_pastur", "gavish_donoho")


class RankError(ValueError):
    """秩估计输入非法。"""


@dataclass(frozen=True)
class RankEstimate:
    rank: int
    method: str
    sigma_hat: float
    threshold: float
    n_sv: int
    depends_on_sigma: bool
    extra: dict | None = None

    def as_dict(self) -> dict:
        return {
            "rank": self.rank,
            "method": self.method,
            "sigma_hat": self.sigma_hat,
            "threshold": self.threshold,
            "n_sv": self.n_sv,
            "depends_on_sigma": self.depends_on_sigma,
        }


# --------------------------------------------------------------------------- 三个判据
def energy_rank(sv: NDArray, sigma_hat: float, m: int, n: int, c: float = 1.0) -> int:
    """能量阈值法：丢弃"可由噪声解释"的尾部。

    噪声对 `‖Z‖_F²` 的贡献期望为 `M·N·σ̂²`（共 `M·N` 个元素、逐元素方差 `σ̂²`）。因此

        r̂ = min{ k : Σ_{i>k} s_i²  ≤  c · M · N · σ̂² }

    即：直到剩余尾部能量可完全由噪声解释为止。

    > ⚠️ 规格 §10.1 原写作 `Σ_{i≤k} s_i² ≥ (1 - c·(σ̂/s_1)²)·Σ_i s_i²`。
    > 该式**退化**：`σ̂/s_1` 通常远小于 1，右端恒接近总能量，导致 `r̂` 永远取满秩。
    > 实测已确认（`tmp/` 冒烟中 energy 判据无输出），已按上式更正。
    """
    if sv.size == 0:
        raise RankError("奇异值序列为空")
    if sigma_hat < 0:
        raise RankError("σ̂ 不得为负")
    budget = c * m * n * sigma_hat**2
    asc = np.cumsum(sv[::-1] ** 2)          # 由小到大累加：asc[j] = 最小的 j+1 个的平方和
    # 把"累计能量仍可由噪声解释"的最小奇异值逐个划入噪声
    noise_dirs = 0
    while noise_dirs < sv.size and asc[noise_dirs] <= budget:
        noise_dirs += 1
    rank = sv.size - noise_dirs
    return int(max(rank, 1))


def energy_threshold(sv: NDArray, sigma_hat: float, m: int, n: int,
                     c: float = 1.0) -> float:
    """能量判据的等价阈值形式：返回第 `r̂` 个奇异值的平方（供报告用）。"""
    r = energy_rank(sv, sigma_hat, m, n, c)
    return float(sv[r - 1] ** 2)


def mp_edge(sigma_hat: float, m: int, n: int) -> float:
    """Marchenko–Pastur 噪声奇异值上沿（`M × N` 实/复噪声矩阵）。

    `β = min/max`；上沿 `= σ̂·√max · (1 + √β)`。
    """
    if sigma_hat < 0:
        raise RankError("σ̂ 不得为负")
    big, small = max(m, n), min(m, n)
    beta = small / big
    return sigma_hat * math.sqrt(big) * (1.0 + math.sqrt(beta))


def gd_omega(beta: float) -> float:
    """Gavish–Donoho 最优硬阈值系数 ω(β)。"""
    return 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43


def gd_threshold(sv: NDArray, m: int, n: int) -> float:
    """Gavish–Donoho 最优硬阈值。**不依赖 σ̂**。"""
    if sv.size == 0:
        raise RankError("奇异值序列为空")
    beta = min(m, n) / max(m, n)
    return float(gd_omega(beta) * np.median(sv))


def estimate_rank(sv: NDArray, *, sigma_hat: float, m: int, n: int,
                  method: str = "gavish_donoho") -> RankEstimate:
    """由奇异值谱与噪声水平估计截断秩。"""
    if method not in VALID_METHODS:
        raise RankError(f"未知秩判据 {method!r}，可选 {VALID_METHODS}")
    sv = np.asarray(sv, dtype=float)
    if sv.size == 0:
        raise RankError("奇异值序列为空")

    if method == "energy":
        rank = energy_rank(sv, sigma_hat, m, n)
        thr = energy_threshold(sv, sigma_hat, m, n)
        depends = True
    elif method == "marchenko_pastur":
        thr = mp_edge(sigma_hat, m, n)
        rank = int(np.sum(sv > thr))
        depends = True
    else:  # gavish_donoho —— σ̂ 不参与
        thr = gd_threshold(sv, m, n)
        rank = int(np.sum(sv > thr))
        depends = False

    return RankEstimate(rank=max(rank, 0), method=method, sigma_hat=sigma_hat,
                        threshold=thr, n_sv=int(sv.size), depends_on_sigma=depends)


# --------------------------------------------------------------------------- 数据驱动 σ̂
def residual_sigma(residual_norm: float, m: int, n: int) -> float:
    """残差法估计 σ̂：`σ̂ = ‖Z - Z_r‖_F / √(M·N)`。"""
    return residual_norm / math.sqrt(m * n)


def tensor_sigma(Z) -> float:
    """由全矩阵能量给出的 σ̂ 上界（迭代初值）：`‖Z‖_F / √(M·N)`。"""
    Z = np.asarray(Z)
    m, n = Z.shape
    return float(np.linalg.norm(Z)) / math.sqrt(m * n)


def adaptive_sigma(Z: NDArray, *, method: str = "gavish_donoho",
                   max_iter: int = 10, return_status: bool = False,
                   sv: NDArray | None = None):
    """交替迭代：残差 -> sigma_hat -> 秩 -> 截断 -> 残差，直至秩稳定。

    对应“噪声水平完全未知”的场景（实验六最贴近实际应用的配置）。

    !! 该迭代会收敛到「虚假不动点」——这是本项目最需要警惕的失效模式。!!

    `sigma_hat` 依赖型判据（energy / marchenko_pastur）在此反馈回路中存在
    **自洽但错误**的解：`sigma_hat` 偏低 -> 阈值偏低 -> 秩高估 -> 残差偏小 ->
    `sigma_hat` 更低 -> …… 直到把噪声方向全部纳入。此时迭代**收敛**
    （秩稳定、`sigma_hat` 稳定），但两者都远离真值，且**无法从迭代本身察觉**。

    实测（复值 n=8、SNR=10 dB、m=200、J=200，真值 r=2）：

        gavish_donoho     收敛率 1.000   sigma_hat/sigma_true = 0.930   r_hat 众数 2   -> 正确
        marchenko_pastur  收敛率 1.000   sigma_hat/sigma_true = 0.158   r_hat 众数 16  -> 秩高估到上界
        energy            收敛率 1.000   sigma_hat/sigma_true = 2.584   r_hat 众数 1   -> 秩低估

    即 **“迭代收敛”不能作为可用性的证据**。gavish_donoho 的阈值
    `tau = omega(beta) * median(s)` 不依赖 `sigma_hat`，故不存在这条回路——
    这是它在端到端评估中零退化的根本原因（见 docs/final-report.md 2.8）。

    返回 `(sigma_hat, r_hat)`；`return_status=True` 时额外返回是否在 max_iter 内稳定。
    **注意**：`converged=True` 只说明迭代稳定，**不说明结果正确**。
    """

    Z = np.asarray(Z)
    m, n = Z.shape
    if sv is None:                       # 允许调用方复用已算好的奇异值谱
        sv = np.linalg.svd(Z, compute_uv=False)
    else:
        sv = np.asarray(sv, dtype=float)
    if sv[0] <= 0:
        raise RankError("Z 的首奇异值非正")

    sigma_hat = residual_sigma(float(np.linalg.norm(Z)), m, n)   # 初始：全能量
    rank = 1
    converged = False
    for _ in range(max_iter):
        est = estimate_rank(sv, sigma_hat=sigma_hat, m=m, n=n, method=method)
        rank_new = max(int(est.rank), 1)
        if rank_new > min(m, n):
            rank_new = min(m, n)
        resid = math.sqrt(max(float(np.sum(sv[rank_new:] ** 2)), 0.0))
        sigma_new = residual_sigma(resid, m, n)
        if rank_new == rank:
            sigma_hat = 0.5 * (sigma_hat + sigma_new)
            converged = True
            break
        rank, sigma_hat = rank_new, sigma_new
    if not converged:
        LOG_RANK.warning(
            "adaptive_sigma 未在 %d 次内收敛（method=%s，末态 r̂=%d、σ̂=%.4g）；"
            "注意：即使收敛也可能落在虚假不动点上",
            max_iter, method, rank, sigma_hat,
        )
    return (sigma_hat, rank, converged) if return_status else (sigma_hat, rank)
