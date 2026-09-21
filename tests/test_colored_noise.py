"""有色噪声生成器（实验九）的正确性测试。

核心待验证的性质：

  1. `_colored_noise` 必须保持**逐元素单位方差**——否则 `sigma_from_snr` 的
     标定被破坏，SNR 就不再是设定的值；
  2. 时间方向须给出目标 AR(1) 系数，通道方向须给出目标指数相关；
  3. `ar1 = spatial = 0` 时必须**逐位退化**为原白噪声路径（回归保护）；
  4. 非法参数（`≥ 1`）必须被拒。
"""

from __future__ import annotations

import numpy as np
import pytest

from dmdnoise.sim.noise import _colored_noise, inject
from dmdnoise.sim import build
from dmdnoise.config import Config


def _ac_temporal(z: np.ndarray) -> float:
    """时间方向 lag-1 自相关（逐通道平均）。"""
    return float(np.mean([np.corrcoef(z[i, :-1], z[i, 1:])[0, 1]
                          for i in range(z.shape[0])]))


def _ac_spatial(z: np.ndarray) -> float:
    """通道方向 lag-1 相关（合并所有通道-时刻对，避免小样本偏差）。"""
    return float(np.corrcoef(z[:-1].ravel(), z[1:].ravel())[0, 1])


@pytest.mark.parametrize("kw", [
    {}, {"ar1": 0.5}, {"ar1": 0.9}, {"spatial": 0.5}, {"spatial": 0.9},
    {"ar1": 0.5, "spatial": 0.5},
])
@pytest.mark.parametrize("complex_", [False, True])
def test_unit_variance_is_preserved(kw: dict, complex_: bool) -> None:
    """★ 必须保持逐元素单位方差——否则 SNR 标定失真。"""
    rng = np.random.default_rng(0)
    z = _colored_noise((8, 4000), 1.0, rng, complex_, **kw)
    assert z.shape == (8, 4000)
    assert abs(np.mean(np.abs(z) ** 2) - 1.0) < 0.08, np.mean(np.abs(z) ** 2)


def test_temporal_correlation_matches_target() -> None:
    rng = np.random.default_rng(0)
    for rho in (0.25, 0.5, 0.9):
        z = _colored_noise((8, 8000), 1.0, rng, False, ar1=rho)
        assert abs(_ac_temporal(z) - rho) < 0.03, rho


def test_spatial_correlation_matches_target() -> None:
    rng = np.random.default_rng(0)
    for rho in (0.25, 0.5, 0.9):
        z = _colored_noise((8, 20000), 1.0, rng, False, spatial=rho)
        assert abs(_ac_spatial(z) - rho) < 0.03, rho


def test_white_and_colored_are_declared() -> None:
    """白噪声的时间/空间相关都应为 0，有色则应显著非零。"""
    rng = np.random.default_rng(0)
    w = _colored_noise((8, 6000), 1.0, rng, False)
    assert abs(_ac_temporal(w)) < 0.05
    assert abs(_ac_spatial(w)) < 0.05
    t = _colored_noise((8, 6000), 1.0, rng, False, ar1=0.8)
    assert _ac_temporal(t) > 0.7
    s = _colored_noise((8, 6000), 1.0, rng, False, spatial=0.8)
    assert _ac_spatial(s) > 0.7


def test_zero_parameters_degenerate_to_white_path() -> None:
    """`ar1 = spatial = 0` 必须与显式白噪声路径**逐位相同**（回归保护）。"""
    a = _colored_noise((4, 50), 2.0, np.random.default_rng(7), False)
    b = _colored_noise((4, 50), 2.0, np.random.default_rng(7), False,
                       ar1=0.0, spatial=0.0)
    np.testing.assert_array_equal(a, b)


def test_sigma_scales_linearly() -> None:
    rng1 = np.random.default_rng(3)
    rng2 = np.random.default_rng(3)
    a = _colored_noise((4, 200), 1.0, rng1, False, ar1=0.5)
    b = _colored_noise((4, 200), 3.0, rng2, False, ar1=0.5)
    np.testing.assert_allclose(b, 3.0 * a)


@pytest.mark.parametrize("kw", [{"ar1": 1.0}, {"ar1": 1.5},
                                {"spatial": 1.0}, {"spatial": 2.0}])
def test_invalid_correlation_rejected(kw: dict) -> None:
    with pytest.raises(Exception):
        _colored_noise((4, 50), 1.0, np.random.default_rng(0), False, **kw)


def test_inject_trajectory_overlap_preserved_with_colored_noise() -> None:
    """有色噪声下 trajectory 模式的重叠结构必须保持（否则 EIV 模型失效）。"""
    cfg = Config()
    system = build(cfg, "complex", np.random.default_rng(0), 100)
    from dmdnoise.sim.noise import sigma_from_snr
    sig = sigma_from_snr(system.X, 10.0)
    for kw in ({}, {"ar1": 0.5}, {"spatial": 0.5}, {"ar1": 0.5, "spatial": 0.5}):
        X, Y, meta = inject(system.X, system.Y, sig, np.random.default_rng(1),
                            mode="trajectory", **kw)
        assert meta["overlap_ok"] is True, kw
        np.testing.assert_array_equal(X[:, 1:] - system.X[:, 1:],
                                      Y[:, :-1] - system.Y[:, :-1])
        assert meta["ar1"] == kw.get("ar1", 0.0)
        assert meta["spatial"] == kw.get("spatial", 0.0)
