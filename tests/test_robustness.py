"""实验四（参数稳健性）的正确性测试。

重点验证：**扫描范围必须落在设计约束之内**，否则扫出来的"稳健性"结论
可能只是越界区域的产物。本项目的参数扫描曾因此被迫从 `f2/f1 = 1.12`
上移到 C2 硬边界 `1.25`。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dmdnoise.config import Config, ConfigError
from dmdnoise.experiments.exp4_robustness import (
    SEPARATIONS,
    WINDOW_DECAYS,
    RobustPoint,
    default_points,
    make_config,
    zeta_from_decay,
)


def test_zeta_from_decay_roundtrip(cfg: Config) -> None:
    """由目标衰减反解的 ζ 必须精确复现该衰减。"""
    m, dt = 200, cfg.oscillator.dt
    for target in (0.90, 0.70, 0.50, 0.35):
        z = zeta_from_decay(12.0, target, m, dt)
        got = math.exp(-z * 2 * math.pi * 12.0 * m * dt)
        assert abs(got - target) < 1e-12, (target, got)


def test_zeta_from_decay_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        zeta_from_decay(12.0, 1.0, 200, 0.005)
    with pytest.raises(ValueError):
        zeta_from_decay(12.0, 0.0, 200, 0.005)


def test_all_default_points_satisfy_hard_constraints(cfg: Config) -> None:
    """所有默认参数点都必须通过硬约束——越界的点不得进入扫描网格。"""
    for pt in default_points(cfg):
        sub, results = make_config(cfg, pt, m=200)
        failed = [r.name for r in results if not r.passed]
        assert not failed, f"{pt.describe()} 违反硬约束：{failed}"


def test_c2_boundary_is_the_lowest_separation(cfg: Config) -> None:
    """最低间隔点恰在 C2 硬边界（f2/f1 = 1.25）上，须触发警戒而非拒绝。"""
    pt = RobustPoint("complex", 1.25, 0.70, 8, 10.0)
    sub, results = make_config(cfg, pt, m=200)
    c2 = next(r for r in results if r.name.startswith("C2"))
    assert c2.passed is True
    assert c2.warning is True


def test_separation_below_c2_is_rejected(cfg: Config) -> None:
    """越过 C2 硬边界（如 f2/f1 = 1.12）必须被拒绝，不得静默扫到界外。"""
    pt = RobustPoint("complex", 1.12, 0.70, 8, 10.0)
    sub, results = make_config(cfg, pt, m=200)
    assert any(not r.passed for r in results), "f2/f1=1.12 应违反 C2 硬约束"


@pytest.mark.parametrize("decay", WINDOW_DECAYS)
def test_large_decay_keeps_c3_compliant(cfg: Config, decay: float) -> None:
    """扫描阻尼时 C3 必须按**实际窗口 m** 求值，否则会误触硬约束。"""
    pt = RobustPoint("complex", 1.46, decay, 8, 10.0)
    _, results = make_config(cfg, pt, m=200)
    c3 = next(r for r in results if r.name.startswith("C3"))
    assert c3.passed is True, f"decay={decay} 不应违反 C3：{c3.detail}"


def test_make_config_records_the_actual_window(cfg: Config) -> None:
    """配置的 m_min/m_max/m_scan 必须都等于实际使用的 m。"""
    sub, _ = make_config(cfg, RobustPoint("complex", 1.46, 0.70, 8, 10.0), m=300)
    assert (sub.grid.m_min, sub.grid.m_max, sub.grid.m_scan) == (300, 300, (300,))


def test_separations_and_decays_are_ordered_unique() -> None:
    assert list(SEPARATIONS) == sorted(set(SEPARATIONS))
    assert list(WINDOW_DECAYS) == sorted(set(WINDOW_DECAYS), reverse=True)


def test_channel_mixed_points_present(cfg: Config) -> None:
    """默认网格须同时覆盖复值与实值两个通道。"""
    chans = {pt.channel for pt in default_points(cfg)}
    assert chans == {"complex", "real"}
