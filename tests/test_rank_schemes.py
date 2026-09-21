"""实验七（秩估计方案）+ σ̂ 偏差校正的正确性测试。

核心待验证的性质：

  1. 残差法 σ̂ 的**有限样本偏差有解析形式** `sqrt((M-r)/M)`，
     校正因子 `sqrt(M/(M-r))` 在真秩处须把 σ̂ 恢复为无偏；
  2. **σ̂ 校正是 MP 判据失效的根本解**——补正后四种方案全部恢复正常；
  3. `energy` 与 MP 失效机制**不同**，仍需 `single` 的一次性结构；
  4. `gavish_donoho` 因阈值不依赖 σ̂，对全部方案不变；
  5. `multi` 必须返回**连贯的一对** (σ̂, r̂)；
  6. 方案/判据名非法必须被拒。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dmdnoise.config import Config
from dmdnoise.estimators.rank import (
    DAMPING_ALPHA,
    residual_sigma,
    MULTI_START_FACTORS,
    SCHEMES,
    RankError,
    robust_sigma_rank,
)
from dmdnoise.sim import build, inject, sigma_from_snr


@pytest.fixture(scope="module")
def sample():
    """一个固定的复值系统 + 一批含噪实现（复现成本高，模块级缓存）。"""
    cfg = Config()
    system = build(cfg, "complex", np.random.default_rng(0), 200)
    sigma = sigma_from_snr(system.X, 10.0)
    Zs = []
    for j in range(150):
        Xn, Yn, _ = inject(system.X, system.Y, sigma, np.random.default_rng(j))
        Zs.append(np.vstack([Xn, Yn]))
    return system, sigma, Zs


def _stats(sample, method, scheme):
    _, sigma_true, Zs = sample
    ranks, sigmas = [], []
    for Z in Zs:
        sh, rh = robust_sigma_rank(Z, method=method, scheme=scheme)
        ranks.append(rh)
        sigmas.append(sh / sigma_true)
    return np.asarray(ranks), np.asarray(sigmas)


def test_scheme_names_and_constants() -> None:
    assert SCHEMES == ("iter", "single", "multi", "damped")
    assert len(MULTI_START_FACTORS) >= 3
    assert 0.0 < DAMPING_ALPHA < 1.0


# --------------------------------------------------------------------------- sigma 偏差校正
def test_residual_sigma_correction_is_exact_at_true_rank() -> None:
    """★ 校正因子 `sqrt(M/(M-r))` 必须在真秩处把 σ̂ 恢复为无偏。

    朴素残差法偏低 `sqrt((M-r)/M)`——这是"截断丢掉了落在前 r 个方向上的
    噪声能量"所致，与该解析预测必须吻合。
    """
    cfg = Config()
    system = build(cfg, "complex", np.random.default_rng(0), 200)
    sigma = sigma_from_snr(system.X, 10.0)
    M, r_true = 16, 2
    naive, corrected = [], []
    for j in range(200):
        Xn, Yn, _ = inject(system.X, system.Y, sigma, np.random.default_rng(j))
        sv = np.linalg.svd(np.vstack([Xn, Yn]), compute_uv=False)
        resid = float(np.linalg.norm(sv[r_true:]))
        naive.append(residual_sigma(resid, 16, 200, corrected=False) / sigma)
        corrected.append(residual_sigma(resid, 16, 200, rank=r_true) / sigma)
    # 朴素估计须与解析预测吻合
    assert abs(np.mean(naive) - math.sqrt((M - r_true) / M)) < 0.01
    # 校正后须接近无偏
    assert abs(np.mean(corrected) - 1.0) < 0.02, np.mean(corrected)


def test_correction_can_be_disabled() -> None:
    """`corrected=False` 必须给出朴素估计（供对照与回归）。"""
    a = residual_sigma(1.0, 16, 200, rank=2, corrected=False)
    b = residual_sigma(1.0, 16, 200, rank=2, corrected=True)
    assert b > a
    assert a == pytest.approx(1.0 / math.sqrt(16 * 200))


def test_correction_needs_rank() -> None:
    """未给出秩时不做校正（避免误用）。"""
    a = residual_sigma(1.0, 16, 200)
    b = residual_sigma(1.0, 16, 200, corrected=False)
    assert a == pytest.approx(b)


# --------------------------------------------------------------------------- 四方案的效力
def test_marchenko_pastur_rescued_by_sigma_correction(sample) -> None:
    """★ σ̂ 偏差校正把 marchenko_pastur 从"几乎全错"救到"几乎全对"。

    这是本项目最重要的修正：MP 失效的**根本原因是 σ̂ 系统性偏低 7%**，
    而不是迭代回路本身——补正 σ̂ 后全部四种方案都恢复正常。
    """
    for scheme in SCHEMES:
        ranks, _ = _stats(sample, "marchenko_pastur", scheme)
        acc = (ranks == 2).mean()
        assert acc > 0.95, (scheme, acc)


def test_energy_still_requires_single_scheme(sample) -> None:
    """energy 与 MP 的失效机制**不同**：它仍需 `single` 的一次性结构。

    `iter`/`multi`/`damped` 全部秩低估到 1，只有 `single` 正确。
    """
    for scheme in ("iter", "multi", "damped"):
        ranks, _ = _stats(sample, "energy", scheme)
        assert (ranks == 1).mean() > 0.8, (scheme, np.bincount(ranks))
    r_single, sh_single = _stats(sample, "energy", "single")
    assert (r_single == 2).mean() > 0.9, np.bincount(r_single)
    assert abs(sh_single.mean() - 1.0) < 0.05


@pytest.mark.parametrize("scheme", SCHEMES)
def test_gavish_donoho_invariant_and_unbiased(sample, scheme: str) -> None:
    """GD 阈值不依赖 σ̂，故对全部方案给出完全相同且无偏的结果。"""
    ranks, sigmas = _stats(sample, "gavish_donoho", scheme)
    assert (ranks == 2).mean() == 1.0, np.bincount(ranks)
    assert abs(sigmas.mean() - 1.0) < 0.05, sigmas.mean()


def test_multi_returns_coherent_pair(sample) -> None:
    """★ `multi` 返回的 (σ̂, r̂) 必须来自**同一次**迭代。

    若分别取 σ̂ 与 r̂ 的边际中位数，会出现"σ̂ 看着准确但秩完全错误"的假象。
    """
    from dmdnoise.estimators.rank import estimate_rank
    _, _, Zs = sample
    for Z in Zs[:25]:
        sh, rh = robust_sigma_rank(Z, method="marchenko_pastur", scheme="multi")
        sv = np.linalg.svd(Z, compute_uv=False)
        r_check = estimate_rank(sv, sigma_hat=sh, m=Z.shape[0], n=Z.shape[1],
                                method="marchenko_pastur").rank
        assert min(max(int(r_check), 1), min(Z.shape)) == rh, (sh, rh, r_check)


def test_unknown_scheme_and_method_rejected(sample) -> None:
    _, _, Zs = sample
    with pytest.raises(RankError):
        robust_sigma_rank(Zs[0], method="marchenko_pastur", scheme="bogus")
    with pytest.raises(RankError):
        robust_sigma_rank(Zs[0], method="bogus", scheme="single")


def test_reuses_provided_singular_values(sample) -> None:
    _, _, Zs = sample
    Z = Zs[0]
    sv = np.linalg.svd(Z, compute_uv=False)
    a = robust_sigma_rank(Z, method="gavish_donoho", scheme="single")
    b = robust_sigma_rank(Z, method="gavish_donoho", scheme="single", sv=sv)
    assert a == b


def test_rank_never_exceeds_min_dimension(sample) -> None:
    _, _, Zs = sample
    for Z in Zs[:30]:
        for method in ("energy", "marchenko_pastur", "gavish_donoho"):
            for scheme in SCHEMES:
                _, r = robust_sigma_rank(Z, method=method, scheme=scheme)
                assert 1 <= r <= min(Z.shape)
