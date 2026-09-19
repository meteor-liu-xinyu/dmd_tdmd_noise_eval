"""估计器测试：T1（DMD 无噪精确）、T2（TDMD 无噪精确）、T3/D1（退化判据）、
T9（定序不变性）以及诊断字段完整性。"""

from __future__ import annotations

import numpy as np
import pytest

from dmdnoise.config import Config
from dmdnoise.estimators import DMD, TDMD, pair_to_truth
from dmdnoise.estimators.base import EstimatorError
from dmdnoise.estimators.variants import (
    TdmdNoAugment,
    tdmd_left_pinv,
    tdmd_left_star,
)
from dmdnoise.sim import build, rank_for


def _noiseless(cfg: Config, channel: str, m: int = 200, seed: int = 0):
    system = build(cfg, channel, np.random.default_rng(seed), m)
    return system, rank_for(cfg, channel)


@pytest.mark.parametrize("channel", ["real", "complex"])
def test_t1_dmd_exact_without_noise(cfg: Config, channel: str) -> None:
    """T1：无噪时标准 DMD 精确恢复真值频率（相对误差 < 1e-10）。"""
    system, r = _noiseless(cfg, channel)
    res = DMD().fit(system.X, system.Y, r, cfg.oscillator.dt)
    pr = pair_to_truth(res.freqs, system.f_true)
    assert pr.ok, pr.reason
    rel = np.abs(pr.freqs - system.f_true) / system.f_true
    assert rel.max() < 1e-10, f"{channel}: 最大相对误差 {rel.max():.3e}"


@pytest.mark.parametrize("channel", ["real", "complex"])
def test_t2_tdmd_exact_without_noise(cfg: Config, channel: str) -> None:
    """T2：无噪时 TDMD 精确恢复真值频率。"""
    system, r = _noiseless(cfg, channel)
    res = TDMD().fit(system.X, system.Y, r, cfg.oscillator.dt)
    pr = pair_to_truth(res.freqs, system.f_true)
    assert pr.ok, pr.reason
    rel = np.abs(pr.freqs - system.f_true) / system.f_true
    assert rel.max() < 1e-10, f"{channel}: 最大相对误差 {rel.max():.3e}"


@pytest.mark.parametrize("channel", ["real", "complex"])
def test_t3_d1_tdmd_degenerates_to_dmd(cfg: Config, channel: str) -> None:
    """T3 / 判据 D1：无噪时 TDMD 退化为标准 DMD。

    ⚠️ 注意：该判据是**必要但不充分**条件。实测表明它无法拦截投影子空间取错
       （见 test_left_star_is_not_detectable_without_noise），因此必须配合
       含噪去偏有效性判据 T11（tests/test_debias.py）。
    """
    system, r = _noiseless(cfg, channel)
    a = DMD().fit(system.X, system.Y, r, cfg.oscillator.dt)
    b = TDMD().fit(system.X, system.Y, r, cfg.oscillator.dt)
    rel = np.linalg.norm(b.operator - a.operator) / np.linalg.norm(a.operator)
    assert rel < 1e-10, f"{channel}: ||At-Ad||/||Ad|| = {rel:.3e}"


def test_negative_control_variants_are_available(cfg: Config) -> None:
    """负对照变体必须可调用、被标记，且输出形状与正确版一致（供 T11 使用）。

    ⚠️ 本测试**不**断言"某个错误实现必然失效"。早期校验中曾观察到
       U1^+ U2 在部分设定下与正确版特征值一致（等价变形），但该现象依赖数据设定、
       并非普遍成立，故不作为判据。真正的判别依赖含噪判据 T11（tests/test_debias.py）。
    """
    system, r = _noiseless(cfg, "complex")
    dt = cfg.oscillator.dt
    correct = TDMD().fit(system.X, system.Y, r, dt)
    for fn, tag in ((tdmd_left_star, "tdmd_left_star"),
                    (tdmd_left_pinv, "tdmd_left_pinv")):
        res = fn(system.X, system.Y, r, dt)
        assert res.diagnostics.extra["negative_control"] is True
        assert res.diagnostics.extra["method"] == tag
        assert res.freqs.shape == correct.freqs.shape
        assert np.all(np.isfinite(res.freqs))


def test_conjugate_duplicates_are_merged_by_pairing(cfg: Config) -> None:
    """实值通道 r=4 含两组共轭对，配对前必须合并重复频率。"""
    system, r = _noiseless(cfg, "real")
    res = DMD().fit(system.X, system.Y, r, cfg.oscillator.dt)
    assert res.freqs.size == 4
    assert pair_to_truth(res.freqs, system.f_true).ok
    np.testing.assert_allclose(res.freqs[::2], res.freqs[1::2], rtol=1e-9)


def test_t9_ordering_invariance(cfg: Config) -> None:
    """T9：配对与排序在真值顺序变化后结果不变。"""
    system, r = _noiseless(cfg, "complex")
    res = TDMD().fit(system.X, system.Y, r, cfg.oscillator.dt)
    assert np.all(np.diff(res.freqs) >= 0), "freqs 必须升序"

    p1 = pair_to_truth(res.freqs, system.f_true)
    p2 = pair_to_truth(res.freqs[::-1], system.f_true[::-1])
    assert p1.ok and p2.ok
    np.testing.assert_allclose(np.sort(p1.freqs), np.sort(p2.freqs))


def test_diagnostics_fields_present(cfg: Config) -> None:
    """诊断字段是必需项：低 SNR、大 m 下的病态风险必须可事后审计。"""
    system, r = _noiseless(cfg, "real")
    for est in (DMD(), TDMD()):
        res = est.fit(system.X, system.Y, r, cfg.oscillator.dt)
        d = res.diagnostics.as_dict()
        for key in ("cond", "cond_full", "sv_zeroed", "sv_min_ratio",
                    "rank_used", "augmented", "z_shape", "near_nyquist"):
            assert key in d, f"{est.name} 缺少诊断字段 {key}"
        assert res.operator is not None
        assert res.operator.shape == (r, r)


def test_tdmd_marks_augmented_and_shape(cfg: Config) -> None:
    """G3-c：TDMD 必须标注使用了增广矩阵，并记录其形状。"""
    system, r = _noiseless(cfg, "real")
    res = TDMD().fit(system.X, system.Y, r, cfg.oscillator.dt)
    assert res.diagnostics.augmented is True
    assert tuple(res.diagnostics.z_shape) == (2 * system.X.shape[0], system.X.shape[1])
    assert res.diagnostics.extra["projection"] == "right_singular_vectors_of_Z"


def test_dmd_marks_not_augmented(cfg: Config) -> None:
    system, r = _noiseless(cfg, "real")
    res = DMD().fit(system.X, system.Y, r, cfg.oscillator.dt)
    assert res.diagnostics.augmented is False


def test_no_augment_variant_equals_dmd(cfg: Config) -> None:
    """负对照：「遗漏增广」形态必须与标准 DMD 完全一致（T11 可稳定拦截）。"""
    system, r = _noiseless(cfg, "complex")
    a = DMD().fit(system.X, system.Y, r, cfg.oscillator.dt)
    b = TdmdNoAugment().fit(system.X, system.Y, r, cfg.oscillator.dt)
    np.testing.assert_allclose(a.freqs, b.freqs, rtol=1e-14)
    assert b.diagnostics.extra["negative_control"] is True


def test_invalid_inputs_rejected(cfg: Config) -> None:
    system, r = _noiseless(cfg, "complex")
    with pytest.raises(EstimatorError):
        DMD().fit(system.X, system.Y[:, :-1], r, cfg.oscillator.dt)
    with pytest.raises(EstimatorError):
        DMD().fit(system.X, system.Y, 0, cfg.oscillator.dt)
    with pytest.raises(EstimatorError):
        DMD().fit(system.X, system.Y, 10_000, cfg.oscillator.dt)


def test_pair_to_truth_rejects_mispair(cfg: Config) -> None:
    """配错必须被识别（返回 ok=False），样本须剔除而非计为真实偏差。"""
    f_true = np.array([12.0, 17.5])
    assert pair_to_truth(np.array([12.001, 17.499]), f_true).ok
    bad = pair_to_truth(np.array([12.0, 15.0]), f_true)
    assert not bad.ok and "配错" in bad.reason
    assert not pair_to_truth(np.array([12.0]), f_true).ok
