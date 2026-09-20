"""实验七（秩估计方案）的正确性测试。

核心待验证的性质：
  1. `single` 通过**断开反馈回路**恢复 σ̂ 的准确性；
  2. `damped` **只降低收敛速率、不改变不动点位置**——故结果应与 `iter` 几乎相同；
  3. `gavish_donoho` 因阈值不依赖 σ̂，对全部方案不变；
  4. 方案名非法必须被拒。
"""

from __future__ import annotations

import numpy as np
import pytest

from dmdnoise.config import Config
from dmdnoise.estimators.rank import (
    DAMPING_ALPHA,
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
    for j in range(120):
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


def test_single_recovers_sigma_accuracy(sample) -> None:
    """★ `single` 必须把 σ̂ 恢复到接近真值——这是它有效的直接原因。"""
    ranks, sigmas = _stats(sample, "marchenko_pastur", "single")
    assert abs(sigmas.mean() - 1.0) < 0.15, sigmas.mean()
    # 而基线会塌到远低于真值
    _, sig_iter = _stats(sample, "marchenko_pastur", "iter")
    assert sig_iter.mean() < 0.5, sig_iter.mean()


def test_single_improves_rank_for_sigma_dependent_criterion(sample) -> None:
    """`single` 对 σ̂ 依赖型判据的秩判定应有实质改善。"""
    r_iter, _ = _stats(sample, "marchenko_pastur", "iter")
    r_single, _ = _stats(sample, "marchenko_pastur", "single")
    err_iter = np.abs(r_iter - 2).mean()
    err_single = np.abs(r_single - 2).mean()
    assert err_single < err_iter, (err_iter, err_single)


def test_single_rescues_energy_criterion(sample) -> None:
    """`energy` 在基线下 100% 秩低估；`single` 应把它救回正确量级。"""
    r_iter, _ = _stats(sample, "energy", "iter")
    assert (r_iter == 1).mean() > 0.8       # 基线几乎全部低估到 1
    r_single, sh_single = _stats(sample, "energy", "single")
    assert (r_single == 2).mean() > 0.4, np.bincount(r_single)
    assert abs(sh_single.mean() - 1.0) < 0.15


def test_damped_does_not_move_the_fixed_point(sample) -> None:
    """★ `damped` 只降低收敛速率、**不改变不动点位置**。

    因此其结果应与 `iter` 几乎相同（这正是它无效的原因）。
    """
    r_iter, s_iter = _stats(sample, "marchenko_pastur", "iter")
    r_damp, s_damp = _stats(sample, "marchenko_pastur", "damped")
    assert r_iter.mean() == pytest.approx(r_damp.mean(), abs=0.05)
    assert s_iter.mean() == pytest.approx(s_damp.mean(), abs=0.05)


def test_multi_does_not_beat_single(sample) -> None:
    """多起点取中位**不优于** `single`——它混合了不同吸引域。

    注意必须比较【秩】质量而非 σ̂ 的边际准确度：`multi` 曾因分别取
    σ̂ 与 r̂ 的边际中位数，出现过"σ̂ ≈ 1.0 但秩完全错误"的假象。
    """
    r_multi, _ = _stats(sample, "marchenko_pastur", "multi")
    r_single, _ = _stats(sample, "marchenko_pastur", "single")
    assert np.abs(r_single - 2).mean() < np.abs(r_multi - 2).mean(),         (np.abs(r_single - 2).mean(), np.abs(r_multi - 2).mean())


def test_multi_returns_coherent_pair(sample) -> None:
    """★ `multi` 返回的 (σ̂, r̂) 必须来自**同一次**迭代。

    判据：在每个实现上，返回的 σ̂ 应等于"以该 σ̂ 为输入时判据给出的秩"
    所对应的残差估计——即两者自洽。
    """
    from dmdnoise.estimators.rank import estimate_rank
    _, sigma_true, Zs = sample
    for Z in Zs[:25]:
        sh, rh = robust_sigma_rank(Z, method="marchenko_pastur", scheme="multi")
        sv = np.linalg.svd(Z, compute_uv=False)
        r_check = estimate_rank(sv, sigma_hat=sh, m=Z.shape[0], n=Z.shape[1],
                                method="marchenko_pastur").rank
        assert min(max(int(r_check), 1), min(Z.shape)) == rh, (sh, rh, r_check)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_gavish_donoho_invariant_across_schemes(sample, scheme: str) -> None:
    """GD 的阈值不依赖 σ̂，故对全部方案给出**完全相同**的结果。"""
    ranks, sigmas = _stats(sample, "gavish_donoho", scheme)
    assert (ranks == 2).mean() == 1.0, np.bincount(ranks)
    assert abs(sigmas.mean() - 0.93) < 0.05


def test_unknown_scheme_and_method_rejected(sample) -> None:
    _, _, Zs = sample
    with pytest.raises(RankError):
        robust_sigma_rank(Zs[0], method="marchenko_pastur", scheme="bogus")
    with pytest.raises(RankError):
        robust_sigma_rank(Zs[0], method="bogus", scheme="single")


def test_reuses_provided_singular_values(sample) -> None:
    """允许复用外部算好的奇异值谱，结果必须一致。"""
    _, _, Zs = sample
    Z = Zs[0]
    sv = np.linalg.svd(Z, compute_uv=False)
    a = robust_sigma_rank(Z, method="gavish_donoho", scheme="single")
    b = robust_sigma_rank(Z, method="gavish_donoho", scheme="single", sv=sv)
    assert a == b


def test_rank_never_exceeds_min_dimension(sample) -> None:
    """秩必须落在 [1, min(m, n)] 内。"""
    _, _, Zs = sample
    for Z in Zs[:30]:
        for method in ("energy", "marchenko_pastur", "gavish_donoho"):
            for scheme in SCHEMES:
                _, r = robust_sigma_rank(Z, method=method, scheme=scheme)
                assert 1 <= r <= min(Z.shape)
