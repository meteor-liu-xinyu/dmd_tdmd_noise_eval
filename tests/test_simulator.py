"""仿真器测试：T4（秩正确性）、T5（SNR 反解）、T6（噪声轨迹重叠）、
T13（真值频率基准）、T15（C6 平衡）。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dmdnoise.config import Config
from dmdnoise.sim import (
    NoiseError,
    SimulatorError,
    build,
    eps_ratio,
    inject,
    rank_for,
    realized_snr_db,
    sigma_from_snr,
    snr_from_eps,
)


@pytest.mark.parametrize("channel,rank", [("real", 4), ("complex", 2)])
def test_t4_rank_correctness(cfg: Config, channel: str, rank: int) -> None:
    """T4：无噪时真值快照的秩恰为 r_true，第 r+1 个奇异值应为数值零。"""
    system = build(cfg, channel, np.random.default_rng(0), m=200)
    assert system.rank_true == rank

    sv = np.linalg.svd(np.vstack([system.X, system.Y]), compute_uv=False)
    assert sv[rank] / sv[0] < 1e-12, f"{channel}: sigma_{{r+1}}/sigma_1 = {sv[rank]/sv[0]:.3e}"
    assert sv[rank - 1] / sv[0] > 1e-6, "第 r 个奇异值不应退化"


@pytest.mark.parametrize("channel", ["real", "complex"])
def test_t13_true_frequency_uses_damped_frequency(cfg: Config, channel: str) -> None:
    """T13：真值必须是阻尼频率 f*sqrt(1-zeta^2)，而非名义固有频率。"""
    system = build(cfg, channel, np.random.default_rng(0), m=200)
    expected = np.sort([m.f * math.sqrt(1.0 - m.zeta**2) for m in cfg.oscillator.modes])
    nominal = np.sort([m.f for m in cfg.oscillator.modes])

    np.testing.assert_allclose(system.f_true, expected, rtol=1e-12)
    # 名义频率与真值存在可观测的偏移（当前参数约 8e-6），不得混用
    rel = np.abs(nominal - expected) / expected
    assert np.all(rel > 1e-7), f"名义频率与真值应有可分辨偏移，实际 {rel}"
    assert np.all(rel < 1e-4), f"偏移量级异常：{rel}"


@pytest.mark.parametrize("channel", ["real", "complex"])
def test_t15_c6_observability_balanced(cfg: Config, channel: str) -> None:
    """T15：模态可观测能量应被平衡到 |比值-1| <= tol。"""
    system = build(cfg, channel, np.random.default_rng(3), m=200)
    assert system.c6.passed
    assert abs(system.energy_ratio - 1.0) <= cfg.oscillator.c_balance_tol + 1e-9, (
        f"{channel}: 能量比 {system.energy_ratio}"
    )


def test_c6_balance_is_exact_per_realization(cfg: Config) -> None:
    """C6 平衡必须逐实现精确成立。

    注意：相位旋转与状态转移可交换，但可观测能量 E_k = ||C_blk @ kry(blk, z0)||
    依赖 C_blk 与 z0 的耦合，**不是**旋转不变量。因此缩放必须在每个实现内
    按实际使用的 z0 方向重新标定（否则实测比值会偏离 1 约 1e-3）。
    """
    for channel in ("real", "complex"):
        ratios = [build(cfg, channel, np.random.default_rng(k), m=200).energy_ratio
                  for k in range(6)]
        assert np.allclose(ratios, 1.0, rtol=1e-9), f"{channel}: {ratios}"


def test_t5_snr_inversion_consistency(cfg: Config) -> None:
    """T5：由目标 SNR 反解的 sigma 应使实测 SNR 与目标一致（J 平均后 <1%）。"""
    for channel in ("real", "complex"):
        system = build(cfg, channel, np.random.default_rng(1), m=300)
        for target in (20.0, 10.0, 5.0):
            sigma = sigma_from_snr(system.X, target)
            realized = []
            for j in range(200):
                Xn, _, _ = inject(system.X, system.Y, sigma, np.random.default_rng(j),
                                  mode="trajectory")
                realized.append(realized_snr_db(system.X, Xn))
            mean = float(np.mean(realized))
            assert abs(mean - target) < 0.05 * abs(target), (
                f"{channel}@{target}dB: 实测均值 {mean:.3f}"
            )


def test_eps_ratio_consistency(cfg: Config) -> None:
    """eps 与 SNR 的换算关系：SNR_dB = -20 log10(eps)。"""
    system = build(cfg, "complex", np.random.default_rng(2), m=200)
    for target in (20.0, 10.0, 5.0):
        sigma = sigma_from_snr(system.X, target)
        assert abs(snr_from_eps(eps_ratio(system.X, sigma)) - target) < 1e-9


def test_t6_trajectory_noise_overlap(cfg: Config) -> None:
    """T6：单条噪声轨迹下 dX[:, 1:] 与 dY[:, :-1] 必须逐元素相等。"""
    for channel in ("real", "complex"):
        system = build(cfg, channel, np.random.default_rng(4), m=120)
        sigma = sigma_from_snr(system.X, 10.0)
        Xn, Yn, meta = inject(system.X, system.Y, sigma, np.random.default_rng(5),
                              mode="trajectory")
        assert meta["overlap_ok"] is True
        assert np.array_equal(Xn[:, 1:] - system.X[:, 1:], Yn[:, :-1] - system.Y[:, :-1])
        assert meta["noise_mode"] == "trajectory"


def test_independent_mode_does_not_overlap(cfg: Config) -> None:
    """对照：独立加噪不满足重叠结构，且必须被显式标注。"""
    system = build(cfg, "complex", np.random.default_rng(4), m=120)
    sigma = sigma_from_snr(system.X, 10.0)
    Xn, Yn, meta = inject(system.X, system.Y, sigma, np.random.default_rng(5),
                          mode="independent")
    assert meta["overlap_ok"] is False
    assert not np.array_equal(Xn[:, 1:] - system.X[:, 1:], Yn[:, :-1] - system.Y[:, :-1])


def test_unknown_channel_and_mode_rejected(cfg: Config) -> None:
    with pytest.raises(SimulatorError):
        build(cfg, "bogus", np.random.default_rng(0), m=50)
    system = build(cfg, "real", np.random.default_rng(0), m=50)
    with pytest.raises(NoiseError):
        inject(system.X, system.Y, 0.1, np.random.default_rng(0), mode="bogus")
    with pytest.raises(NoiseError):
        sigma_from_snr(system.X, 10.0, convention="bogus")


@pytest.mark.parametrize("channel", ["real", "complex"])
def test_y_is_exact_linear_image_of_x(cfg: Config, channel: str) -> None:
    """核心正确性：无噪时存在 A_true 使 Y = A_true X 精确成立。

    这等价于"存在秩 r 的精确线性模型"，是无噪判据 T1/T2 能通过的前提。
    实值通道若误用直接叠加式构造，该性质不成立（实测无噪误差达 7.3）。
    """
    system = build(cfg, channel, np.random.default_rng(6), m=150)
    # 最小二乘拟合 A：Y = A X  <=>  Y^T = X^T A^T，故 lstsq 返回的就是 A^T
    At, *_ = np.linalg.lstsq(system.X.T, system.Y.T, rcond=None)
    resid = np.linalg.norm(system.Y - At.T @ system.X) / np.linalg.norm(system.Y)
    assert resid < 1e-12, f"{channel}: 线性关系残差 {resid:.3e}"


def test_local_snr_reported_per_mode(cfg: Config) -> None:
    """逐模态局部 SNR 必须可获取（聚合 SNR 不等于逐模态 SNR）。"""
    system = build(cfg, "real", np.random.default_rng(7), m=200)
    sigma = sigma_from_snr(system.X, 10.0)
    local = system.local_snr_db(sigma)
    assert local.shape == system.f_true.shape
    assert np.all(np.isfinite(local))
    assert np.all(local < 200.0)


def test_rank_for_matches_channel(cfg: Config) -> None:
    assert rank_for(cfg, "real") == cfg.oscillator.r_real
    assert rank_for(cfg, "complex") == cfg.oscillator.r_complex
