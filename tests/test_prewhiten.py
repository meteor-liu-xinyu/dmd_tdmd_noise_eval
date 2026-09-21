"""预白化（实验十）的正确性测试。

核心待验证的性质：

  1. **时间白化不改变信号的本征值** —— 这是预白化在理论上成立的前提
     （信号是移位算子的本征函数，`(I−ρS)` 只改幅度不改 `μ`）；
  2. **不能用正交化的奇异向量估计 ρ** —— 正交归一化破坏原始噪声的时间相关，
     实测 `ρ̂ ≈ 0`；必须用**低秩残差矩阵**；
  3. 参数非法必须被拒。
"""

from __future__ import annotations

import numpy as np
import pytest

from dmdnoise.config import Config
from dmdnoise.estimators.prewhiten import (
    PrewhitenError,
    estimate_ar1_from_residual,
    estimate_spatial_from_residual,
    low_rank_residual,
    spatial_corr_factor,
    spatial_whiten,
    temporal_whiten,
    whiten,
)
from dmdnoise.sim import build, rank_for, sigma_from_snr
from dmdnoise.sim.noise import inject


def _cells(freqs, amps, zetas, dt, m):
    """由阻尼指数和构造 (n_modes, m) 的干净信号。"""
    t = np.arange(m) * dt
    return np.array([a * np.exp(-z * 2 * np.pi * f * t) * np.cos(2 * np.pi * f * t)
                     for f, a, z in zip(freqs, amps, zetas)])


def test_temporal_whiten_preserves_signal_eigenvalues() -> None:
    """★ 时间白化不得改变信号的本征值——这是预白化成立的前提。

    信号 `Σ c_k μ_k^t` 经 `(I − ρS)` 作用后仍是同一组 `μ_k` 的组合，
    只是幅度变为 `c_k(μ_k − ρ)`。故估计到的频率必须**逐位不变**。
    """
    from dmdnoise.estimators import DMD

    cfg = Config()
    dt = cfg.oscillator.dt
    m = 200
    system = build(cfg, "complex", np.random.default_rng(0), m)
    r = rank_for(cfg, "complex")

    for rho in (0.0, 0.3, 0.6, 0.9):
        Xw = temporal_whiten(system.X, rho)
        Yw = temporal_whiten(system.Y, rho)
        res = DMD().fit(Xw, Yw, r, dt)
        got = np.sort(res.freqs)
        np.testing.assert_allclose(got, np.sort(system.f_true), rtol=1e-8,
                                   err_msg=f"rho={rho} 改变了本征值")


def test_spatial_whiten_preserves_rank() -> None:
    """空间白化 `L^{-1}` 不改变信号子空间的维数。"""
    cfg = Config()
    system = build(cfg, "complex", np.random.default_rng(0), 200)
    for rho in (0.0, 0.3, 0.6):
        W = spatial_whiten(system.X, rho)
        sv = np.linalg.svd(W, compute_uv=False)
        r = rank_for(cfg, "complex")
        assert sv[r] / sv[0] < 1e-10, (rho, sv[r] / sv[0])


def test_spatial_corr_factor_reproduces_target() -> None:
    rho, n = 0.6, 8
    L = spatial_corr_factor(rho, n)
    R = L @ L.T
    idx = np.arange(n)
    np.testing.assert_allclose(R, rho ** np.abs(idx[:, None] - idx[None, :]),
                               atol=1e-8)


@pytest.mark.parametrize("rho", [-0.1, 1.0, 1.5])
def test_invalid_rho_rejected(rho: float) -> None:
    with pytest.raises(PrewhitenError):
        temporal_whiten(np.zeros((4, 10)), rho)
    with pytest.raises(PrewhitenError):
        spatial_corr_factor(rho, 4)


def test_temporal_whiten_short_input_rejected() -> None:
    with pytest.raises(PrewhitenError):
        temporal_whiten(np.zeros((4, 2)), 0.5)


def test_estimate_ar1_from_residual_recovers_target() -> None:
    """★ 必须用**残差矩阵**估计 ρ；用正交化的奇异向量会得到 `ρ̂ ≈ 0`。"""
    cfg = Config()
    m = 200
    system = build(cfg, "complex", np.random.default_rng(0), m)
    sigma = sigma_from_snr(system.X, 10.0)
    for rho in (0.0, 0.5, 0.9):
        ests = []
        for j in range(40):
            Xn, Yn, _ = inject(system.X, system.Y, sigma,
                               np.random.default_rng(j), ar1=rho)
            ests.append(estimate_ar1_from_residual(np.vstack([Xn, Yn])))
        got = float(np.median(ests))
        assert abs(got - rho) < 0.20, (rho, got)


def test_tail_singular_vectors_would_fail() -> None:
    """★ 回归：**正交化的尾部奇异向量**不保留噪声的时间相关。

    这条性质曾经害我写出一个完全失效的估计器（`ρ̂ ≈ 0`）。此处把它固化为
    反面记录：尾部右奇异向量的 lag-1 自相关在 `ar1=0.9` 时仍接近 0。
    """
    cfg = Config()
    system = build(cfg, "complex", np.random.default_rng(0), 200)
    sigma = sigma_from_snr(system.X, 10.0)
    acs = []
    for j in range(40):
        Xn, Yn, _ = inject(system.X, system.Y, sigma,
                           np.random.default_rng(j), ar1=0.9)
        _, _, Vh = np.linalg.svd(np.vstack([Xn, Yn]), full_matrices=False)
        tail = Vh[Vh.shape[0] // 2:]
        for row in tail:
            a, b = np.real(row[:-1]), np.real(row[1:])
            if a.std() > 0 and b.std() > 0:
                acs.append(float(np.corrcoef(a, b)[0, 1]))
    # 真值为 0.9，而正交化向量给出接近 0 的估计 —— 说明此路不通
    assert abs(float(np.median(acs))) < 0.3, np.median(acs)


def test_low_rank_residual_keeps_enough_rows() -> None:
    """残差必须保留足够的奇异值个数（不能截到满秩）。"""
    cfg = Config()
    system = build(cfg, "complex", np.random.default_rng(0), 200)
    R, k = low_rank_residual(np.vstack([system.X, system.Y]))
    assert 1 <= k <= min(R.shape) - 2


def test_estimate_spatial_and_whiten_facade() -> None:
    cfg = Config()
    system = build(cfg, "complex", np.random.default_rng(0), 200)
    sigma = sigma_from_snr(system.X, 10.0)
    Xn, Yn, _ = inject(system.X, system.Y, sigma, np.random.default_rng(0),
                       spatial=0.5)
    Z = np.vstack([Xn, Yn])
    rho_s = estimate_spatial_from_residual(Z)
    assert 0.0 <= rho_s <= 0.95
    W = whiten(Z, spatial=0.5)
    assert W.shape == Z.shape
    Wt = whiten(Z, ar1=0.5)
    assert Wt.shape == (Z.shape[0], Z.shape[1] - 1)


def test_whiten_with_zero_is_identity_or_truncation() -> None:
    Z = np.arange(24, dtype=float).reshape(4, 6)
    np.testing.assert_array_equal(whiten(Z), Z)
    np.testing.assert_array_equal(whiten(Z, ar1=0.0, spatial=0.0), Z)
