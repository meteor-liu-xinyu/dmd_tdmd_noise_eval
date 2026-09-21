"""谱标定秩判据的**失败记录**（实验十一，未成功）。

⚠️ 本文件记录的是一个**未能做通的尝试**，不是可用功能的测试。
其价值在于把失败的结构固定下来，防止后来者重复同样的路。

被检验的假设是："不改变数据、只按噪声协方差仿真噪声谱来标定阈值，
即可回避预白化在极高空间相关下的条件数问题。"

**结论：两种阈值形式都失败。**
  - 单一阈值：白噪声 1.000 / `spatial-0.5` 1.000 / `spatial-0.9` 1.000，
    但 `ar1-0.5` **0.000**（噪声 top2/top3 与阈值同量级，集体超阈）
  - 逐阶阈值：**连白噪声都掉到 0.060**（索引错位：观测第 i 阶对应噪声第 i−r 阶）
  - 对照：**预白化**在 `ar1-0.5/0.9`、`spatial-0.5` 下均为 1.000

**已排除的假设**：不是 `σ̂` 偏差所致——信号第 2 奇异值与阈值的余量为 248%–393%，
而 `σ̂` 偏差仅 0.7%–4%，差两个数量级（见 `test_margin_far_exceeds_sigma_bias`）。
"""

from __future__ import annotations

import numpy as np
import pytest

from dmdnoise.config import Config
from dmdnoise.estimators.rank import residual_sigma
from dmdnoise.estimators.spectral_rank import (
    SpectralThreshold,
    noise_spectrum_quantiles,
    spectral_rank,
)
from dmdnoise.sim import build, sigma_from_snr
from dmdnoise.sim.noise import inject


@pytest.fixture(scope="module")
def system():
    cfg = Config()
    s = build(cfg, "complex", np.random.default_rng(0), 200)
    sigma = sigma_from_snr(s.X, 10.0)
    return s, sigma


def test_noise_spectrum_quantiles_shape_and_ordering() -> None:
    """噪声谱的逐阶分位必须降序、长度等于奇异值个数。"""
    q = noise_spectrum_quantiles(16, 200, complex_=True, n_sim=60)
    assert len(q) == 16
    assert all(q[i] >= q[i + 1] - 1e-9 for i in range(15))
    assert q[0] > 0


def test_quantile_cache_is_effective() -> None:
    """★ 阈值必须被缓存——否则每次标定做 `n_sim` 次 SVD，实验无法进行。"""
    import time

    noise_spectrum_quantiles.cache_clear()
    t0 = time.perf_counter()
    noise_spectrum_quantiles(16, 200, complex_=True, n_sim=60, seed=7)
    first = time.perf_counter() - t0
    t0 = time.perf_counter()
    noise_spectrum_quantiles(16, 200, complex_=True, n_sim=60, seed=7)
    second = time.perf_counter() - t0
    assert second < first / 10.0, (first, second)


def _single_threshold_rank(Z, *, ar1=0.0, spatial=0.0, rank_hint=2, n_sim=60):
    """**第一版**形式：只用噪声 top 的分位作单一阈值（已弃用）。

    内联实现是为了忠实记录那次尝试的行为——模块里现有的是后续的逐阶形式。
    """
    n, m = Z.shape
    sv = np.linalg.svd(Z, compute_uv=False)
    k = min(max(rank_hint, 1), sv.size - 2)
    sh = residual_sigma(float(np.linalg.norm(sv[k:])), n, m, rank=k)
    tau = noise_spectrum_quantiles(n, m, complex_=np.iscomplexobj(Z), ar1=ar1,
                                   spatial=spatial, n_sim=n_sim)[0] * sh
    return int(np.sum(sv > tau))


def test_single_threshold_works_for_white_and_spatial(system) -> None:
    """第一版（单一阈值）在**白噪声与空间相关**下有效——这是该形式的适用区。"""
    s, sigma = system
    for kw, label in (({}, "white"), ({"spatial": 0.5}, "spatial-0.5")):
        ranks = []
        for j in range(40):
            X, Y, _ = inject(s.X, s.Y, sigma, np.random.default_rng(j), **kw)
            ranks.append(_single_threshold_rank(np.vstack([X, Y]),
                                                ar1=kw.get("ar1", 0.0),
                                                spatial=kw.get("spatial", 0.0)))
        acc = float(np.mean(np.asarray(ranks) == 2))
        assert acc > 0.9, (label, acc)


def test_single_threshold_fails_for_temporal(system) -> None:
    """★ 失败记录：单一阈值在**时间相关**下崩溃（真值 2，实测稳定偏到 3–4）。

    原因是时间相关使噪声谱顶部变"平"，`top2`/`top3` 与"top 的 99 分位"同量级，
    它们会**集体超阈**。此测试**断言失败存在**，以便在修复后立刻察觉。
    """
    s, sigma = system
    ranks = []
    for j in range(40):
        X, Y, _ = inject(s.X, s.Y, sigma, np.random.default_rng(j), ar1=0.5)
        ranks.append(_single_threshold_rank(np.vstack([X, Y]), ar1=0.5))
    acc = float(np.mean(np.asarray(ranks) == 2))
    assert acc < 0.2, f"若此断言失败说明单一阈值已被修好，需更新模块说明：{acc}"
    assert max(ranks) > 2, ranks


def test_margin_far_exceeds_sigma_bias(system) -> None:
    """★ 排除"阈值精度不足"这一假设。

    信号第 2 奇异值与噪声阈值的**余量**为数百 %，而 `σ̂` 的相对偏差只有几个 %——
    两者差两个数量级，故失败**不是** `σ̂` 偏差造成的。
    """
    s, sigma = system
    margins, biases = [], []
    for j in range(40):
        X, Y, _ = inject(s.X, s.Y, sigma, np.random.default_rng(j), ar1=0.5)
        sv = np.linalg.svd(np.vstack([X, Y]), compute_uv=False)
        sh = residual_sigma(float(np.linalg.norm(sv[2:])), 16, 200, rank=2)
        margins.append(sv[1] / sigma)
        biases.append(sh / sigma)
    q = noise_spectrum_quantiles(16, 200, complex_=True, ar1=0.5, n_sim=60)[0]
    margin = float(np.median(margins)) / q - 1.0
    bias = 1.0 - float(np.median(biases))
    assert margin > 1.0, margin          # 余量 >100%
    assert bias < 0.10, bias             # σ̂ 偏差 <10%
    assert margin > 10 * bias, (margin, bias)


def test_per_order_threshold_is_worse_than_single(system) -> None:
    """★ 失败记录：逐阶阈值**更差**——连白噪声都被判错（索引错位）。

    观测的第 `i` 个奇异值（`i > r`）对应噪声的第 `i − r` 个，而 `q` 与 `sv`
    均降序，故 `sv[i] > q[i]` 系统性偏向"超阈"，秩被一路推高。
    """
    s, sigma = system
    ranks = []
    for j in range(30):
        X, Y, _ = inject(s.X, s.Y, sigma, np.random.default_rng(j))
        sv = np.linalg.svd(np.vstack([X, Y]), compute_uv=False)
        q = noise_spectrum_quantiles(16, 200, complex_=True, n_sim=60)
        sh = residual_sigma(float(np.linalg.norm(sv[2:])), 16, 200, rank=2)
        rank = 0
        for i in range(min(sv.size, len(q))):
            if sv[i] > q[i] * sh:
                rank = i + 1
            else:
                break
        ranks.append(rank)
    acc = float(np.mean(np.asarray(ranks) == 2))
    assert acc < 0.3, f"若此断言失败说明逐阶形式已被修好：{acc}"


def test_threshold_result_is_serialisable(system) -> None:
    s, sigma = system
    X, Y, _ = inject(s.X, s.Y, sigma, np.random.default_rng(0))
    res = spectral_rank(np.vstack([X, Y]))
    assert isinstance(res, SpectralThreshold)
    d = res.as_dict()
    assert "rank" in d and "threshold" in d and "sigma_hat" in d
