"""实验五（交叉点定位）的正确性测试。

`_crossings` 的分支逻辑曾是错的：当**扫描下界点已经 ρ ≥ 1** 时，
旧实现会把交叉点误报在网格下界上，而正确结论是"交叉点在下界之外"。
本文件把四类情形全部固化为回归测试。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dmdnoise.experiments.exp5_crossover import (
    _crossings,
    rho_ci,
    summary,
)


def _row(sep, rho, lo, hi, *, fail=0.0, resolvable=True, mode=1,
         n=16, channel="real"):
    return {
        "channel": channel, "n": n, "mode": mode, "f2_over_f1": sep,
        "pair_fail_rate": fail, "dmd_resolvable": resolvable,
        "rho": rho, "rho_lo": lo, "rho_hi": hi,
        "covers_one": bool(lo <= 1.0 <= hi),
    }


def test_rho_ci_recovers_known_ratio() -> None:
    """人工构造已知 ρ：公共项 + 独立噪声，检验点估计与 CI 覆盖。"""
    rng = np.random.default_rng(0)
    n = 4000
    common = 0.02 * rng.standard_normal(n)
    d = common + 0.001 * rng.standard_normal(n) + 0.010   # bias_D = 0.010
    t = common + 0.001 * rng.standard_normal(n) + 0.002   # bias_T = 0.002
    point, lo, hi = rho_ci(d, t, bootstrap=2000, rng=rng)
    assert abs(point - 0.2) < 0.08, point
    assert lo < point < hi


def test_rho_ci_paired_is_tighter_than_unpaired() -> None:
    """配对 CI 必须显著窄于"分别统计再相除"——这是定准交叉点的前提。"""
    rng = np.random.default_rng(1)
    n = 3000
    common = 0.02 * rng.standard_normal(n)
    d = common + 0.002 * rng.standard_normal(n) + 0.01
    t = common + 0.002 * rng.standard_normal(n) + 0.004
    _, lo_p, hi_p = rho_ci(d, t, bootstrap=2000, rng=rng)

    a = d[rng.integers(0, n, size=(2000, n))].mean(axis=1)
    b = t[rng.integers(0, n, size=(2000, n))].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.abs(b) / np.abs(a)
    r = r[np.isfinite(r)]
    lo_u, hi_u = np.percentile(r, [2.5, 97.5])
    assert (hi_p - lo_p) < (hi_u - lo_u)


def test_rho_ci_guards_short_or_mismatched() -> None:
    a = np.zeros(3)
    b = np.zeros(3)
    p, lo, hi = rho_ci(a, b)
    assert not np.isfinite(p)
    p, lo, hi = rho_ci(np.zeros(50), np.zeros(40))
    assert not np.isfinite(p)


def test_crossing_brackets_two_points() -> None:
    """ρ 由下方穿越 1：应落在夹住它的两个网格点之间。"""
    t = pd.DataFrame([
        _row(1.60, 0.70, 0.60, 0.80),
        _row(1.70, 0.85, 0.75, 0.95),
        _row(1.80, 1.10, 1.02, 1.20),   # 首个 >= 1；下界 CI 上界 0.95 < 1
        _row(1.90, 1.35, 1.20, 1.55),
    ])
    c = _crossings(t).iloc[0]
    assert c["crossing_lo"] == 1.70 and c["crossing_hi"] == 1.80
    assert 1.70 < c["crossing"] < 1.80
    assert bool(c["well_determined"]) is True


def test_crossing_below_range_is_not_reported_at_boundary() -> None:
    """★ 回归：下界点已 ρ ≥ 1 时，必须报"下界之外"，不得把交叉点钉在下界。"""
    t = pd.DataFrame([
        _row(1.70, 1.20, 1.05, 1.40),
        _row(1.80, 1.45, 1.25, 1.70),
        _row(1.90, 1.80, 1.50, 2.20),
    ])
    c = _crossings(t).iloc[0]
    assert not np.isfinite(c["crossing"]), c["crossing"]
    assert "下界之外" in c["note"], c["note"]


def test_crossing_above_range_when_always_below_one() -> None:
    t = pd.DataFrame([
        _row(1.50, 0.30, 0.20, 0.45),
        _row(1.70, 0.50, 0.40, 0.65),
        _row(1.90, 0.80, 0.65, 0.95),
    ])
    c = _crossings(t).iloc[0]
    assert not np.isfinite(c["crossing"])
    assert "上界之外" in c["note"], c["note"]


def test_crossing_flags_wide_interval_when_ci_covers_one() -> None:
    """夹住 1 的两点若 CI 含 1，须标记为"确定度不足"，不得声称定准。"""
    t = pd.DataFrame([
        _row(1.60, 0.70, 0.50, 0.95),
        _row(1.70, 0.90, 0.60, 1.40),   # CI 含 1
        _row(1.80, 1.20, 0.90, 1.60),   # CI 含 1
        _row(1.90, 1.50, 1.10, 2.00),
    ])
    c = _crossings(t).iloc[0]
    assert bool(c["well_determined"]) is False
    assert "偏宽" in c["note"]


def test_crossing_skips_unreliable_points() -> None:
    """配对失败率过高或偏差不可分辨的点不得进入交叉点定位。"""
    t = pd.DataFrame([
        _row(1.50, 0.60, 0.50, 0.70),
        _row(1.60, 0.99, 0.90, 1.10, fail=0.50),        # 失败率超门槛
        _row(1.70, 1.30, 1.15, 1.50, resolvable=False),  # 偏差不可分辨
        _row(1.80, 1.20, 1.05, 1.40),
        _row(1.90, 1.45, 1.25, 1.70),
    ])
    c = _crossings(t).iloc[0]
    # 1.50(<1) 与 1.80(>=1) 是被保留点中的夹点
    assert c["crossing_lo"] == 1.50 and c["crossing_hi"] == 1.80


def test_crossing_reports_insufficient_when_too_few_points() -> None:
    t = pd.DataFrame([_row(1.70, 0.9, 0.8, 1.0), _row(1.80, 1.1, 1.0, 1.2)])
    c = _crossings(t).iloc[0]
    assert "不足" in c["note"], c["note"]


def test_summary_shape() -> None:
    t = pd.DataFrame([
        _row(1.60, 0.70, 0.60, 0.80),
        _row(1.70, 0.90, 0.80, 0.98),
        _row(1.80, 1.10, 1.02, 1.20),
    ])
    res = summary(type("R", (), {"crossing": _crossings(t)})())
    assert "交叉点区间" in res.columns
    assert res.iloc[0]["交叉点区间"].startswith("[")
