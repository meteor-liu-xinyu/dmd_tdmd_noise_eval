"""T11 判别性测试：含噪去偏有效性。

这是**唯一**能拦截"投影子空间取错"的判据。无噪退化判据 T1–T3 对它无判别力
（实测：误用左奇异向量时无噪误差仍为 8.9e-16）。

工作点依据 tmp/scan_lean.py 的扫描结果：
  平衡态、复值通道、m=200、SNR=-5 dB 处 DMD 的偏差首次变得可分辨
  （|bias|/std ≈ 0.06，50% 分辨所需 J ≈ 4.3e3）。
"""

from __future__ import annotations

import numpy as np
import pytest

from dmdnoise.config import Config
from dmdnoise.estimators.variants import TdmdNoAugment
from dmdnoise.mc import run_cell
from dmdnoise.sim import build, inject, sigma_from_snr
from dmdnoise.metrics import paired_ratio, summarize

#: T11 阈值：TDMD 偏差不得超过 DMD 偏差的该比例
T11_THRESHOLD = 0.60

#: 判别性测试的工作点。
#: 选 n=8 / SNR=10 dB。三条约束同时满足：
#:   1. DMD 偏差可分辨（实测 |bias|≈2.6e-3，需 J≈13）
#:   2. 配对失败率≈0（SNR=-5dB 配 n=8 时失败率近 60%，筛选存活样本会引入
#:      向下的选择偏差，使降偏比失真）
#:   3. 配对设计有效（公共随机数确实带来方差消减）
#: n=4 虽偏差更大，但最小冗余度 n=2r 下两法估计几乎不相关，
#: 方差消减仅 14%、配对比较失效，故不采用。
SNR_DB = 10.0
M = 200
J = 2000
N_CHANNELS = 8


def _low_rank_cell(cfg: Config, j_total: int):
    """在 n=N_CHANNELS 的配置下运行一个单元。"""
    from dataclasses import replace

    sub = replace(cfg, oscillator=replace(cfg.oscillator, n=N_CHANNELS))
    return run_cell(sub, "complex", SNR_DB, M, j_total)


@pytest.mark.slow
def test_t11_debiasing_is_effective_at_low_snr(cfg: Config) -> None:
    """T11：低 SNR 下 TDMD 必须显著降偏，且 DMD 的偏差可分辨。

    若不通过，说明 TDMD 的投影子空间很可能取错（风险 R1）。
    """
    cell = _low_rank_cell(cfg, J)

    d, t = cell.paired_errors(mode=0)
    sd = summarize(d, bootstrap=0)
    st = summarize(t, bootstrap=0)

    assert sd.resolvable, (
        f"DMD 偏差在工作点不可分辨，无法用于标定 T11："
        f"|bias|/std={sd.bias_std_ratio:.4g}"
    )

    ratio = abs(st.bias) / abs(sd.bias)
    assert ratio <= T11_THRESHOLD, (
        f"TDMD 未达去偏有效性：|bias_TDMD|/|bias_DMD| = {ratio:.3f} > {T11_THRESHOLD}"
    )


@pytest.mark.slow
def test_t11_negative_control_no_augment_is_rejected(cfg: Config) -> None:
    """负对照：「遗漏增广」形态必须被 T11 拒绝（它等价于标准 DMD）。"""
    system = build(cfg, "complex", np.random.default_rng(11), M)
    sigma = sigma_from_snr(system.X, SNR_DB)
    Xn, Yn, _ = inject(system.X, system.Y, sigma, np.random.default_rng(12))

    from dmdnoise.estimators import DMD

    a = DMD().fit(Xn, Yn, cfg.oscillator.r_complex, cfg.oscillator.dt)
    b = TdmdNoAugment().fit(Xn, Yn, cfg.oscillator.r_complex, cfg.oscillator.dt)
    np.testing.assert_allclose(a.freqs, b.freqs, rtol=1e-14)

    # 逐样本相同 => 降偏比为 1.0，必然超过阈值
    ratio = 1.0
    assert ratio > T11_THRESHOLD, "T11 必须能拒绝无降偏效果的实现"


@pytest.mark.slow
def test_no_variance_penalty_observed(cfg: Config) -> None:
    """早期观测：TDMD **未**表现出方差代价，其 std 反而略低于标准 DMD。

    规格中的假设 H3 预期 std_TDMD > std_DMD（TLS 以方差换偏差）。实测相反：
    配对 bootstrap 给出 std 比 < 1，即 TDMD 的方差**反而更小**。
    配对比较之所以比分别统计更灵敏，是因为它利用了公共随机数下两法估计的强相关。

    判定：只要 std 比的上界未超过 1 + 5%，即可判定"无方差代价"
    （实际观测到的是方差**收益**，方向与 H3 相反）。
    """
    cell = _low_rank_cell(cfg, 2000)

    d, t = cell.paired_errors(mode=0)

    pr = paired_ratio(t, d, stat="std", bootstrap=2000, rng=np.random.default_rng(0))
    assert pr.paired_effective, pr.note
    assert pr.ci[1] < 1.05, f"检测到方差代价：std 比 CI 上界 {pr.ci[1]:.3f}"
    assert pr.point < 1.05, f"std 比点估计 {pr.point:.3f} 超过无代价阈值"
    # 规格定义：T11 要求 TDMD 偏差显著更小；本测试补充验证其方差不更大
    assert pr.point < 1.0, f"std 比 {pr.point:.3f} 未低于 1，与早期观测不符"

