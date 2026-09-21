"""实验八（交叉点的实现间散布）的正确性测试。

核心待验证的性质：

  1. **同一 `n`、不同 `C` 实现之间必须能给出不同的交叉点**——
     这是"随 `n` 非单调"可能是实现噪声的前提；
  2. 交叉点定位逻辑须正确处理三类情形（在下界之下 / 在上界之上 / 内插得到）；
  3. **`C` 与相位的控制必须有效**：同一配置编号、只改频率时，
     观测矩阵应逐位相同（这是"只变间隔"这一对照成立的前提）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dmdnoise.config import Config
from dmdnoise.experiments.exp8_crossover_scatter import (
    N_VALUES,
    SEPARATIONS,
    _config_for,
    _crossings,
    _spread,
    verdict,
)
from dmdnoise.sim import build


def _row(n, k, mode, sep, rho, *, fail=0.0):
    return {"n_axis": n, "config": k, "mode": mode, "f2_over_f1": sep,
            "rho": rho, "pair_fail_dmd": fail, "pair_fail_tdmd": fail}


def test_separation_grid_covers_observed_region() -> None:
    """间隔网格须覆盖实验五观察到的交叉区间（1.66–2.00）并向外留余量。"""
    assert min(SEPARATIONS) < 1.66
    assert max(SEPARATIONS) >= 1.95
    assert list(SEPARATIONS) == sorted(SEPARATIONS)


def test_n_values_span_the_anomaly() -> None:
    assert set(N_VALUES) >= {16, 32}


def test_crossing_below_range_reported_as_such() -> None:
    """ρ 全 > 1 -> 交叉点在扫描下界之下，**不得**把交叉点钉在下界。"""
    t = pd.DataFrame([_row(16, 0, 1, s, 1.5) for s in SEPARATIONS])
    c = _crossings(t).iloc[0]
    assert not np.isfinite(c["crossing"])
    assert "下界之下" in c["note"]


def test_crossing_above_range_reported_as_such() -> None:
    t = pd.DataFrame([_row(16, 0, 1, s, 0.5) for s in SEPARATIONS])
    c = _crossings(t).iloc[0]
    assert not np.isfinite(c["crossing"])
    assert "上界之上" in c["note"]


def test_crossing_interpolated_inside_range() -> None:
    t = pd.DataFrame([
        _row(16, 0, 1, 1.55, 0.60),
        _row(16, 0, 1, 1.75, 1.20),
        _row(16, 0, 1, 1.95, 1.60),
    ])
    c = _crossings(t).iloc[0]
    assert 1.55 < c["crossing"] < 1.75
    assert c["note"] == "线性内插"


def test_crossing_skips_high_pair_failure() -> None:
    """配对失效率超门槛的网格点不得参与交叉点定位。"""
    t = pd.DataFrame([
        _row(16, 0, 1, 1.55, 0.60),
        _row(16, 0, 1, 1.75, 0.99, fail=0.9),      # 应被剔除
        _row(16, 0, 1, 1.95, 1.40),
    ])
    c = _crossings(t).iloc[0]
    assert 1.55 < c["crossing"] < 1.95


def test_implementations_can_differ_wildly() -> None:
    """★ 同一 `n`、不同 `C` 实现必须能给出**不同**的交叉点分类。

    这是"随 n 非单调可能是实现噪声"这一假设的前提：若所有实现
    都给同一结论，则该假设不成立。
    """
    t = pd.DataFrame(
        [_row(16, 0, 1, s, 1.8) for s in SEPARATIONS]        # 全 > 1
        + [_row(16, 1, 1, s, 0.4) for s in SEPARATIONS]      # 全 < 1
        + [_row(16, 2, 1, 1.55, 0.6), _row(16, 2, 1, 1.75, 1.1),
           _row(16, 2, 1, 1.95, 1.5)]                        # 中间穿越
    )
    c = _crossings(t)
    notes = set(c["note"])
    assert "ρ 全 > 1，交叉点在下界之下" in notes
    assert "ρ 全 < 1，交叉点在上界之上" in notes
    assert "线性内插" in notes


def test_spread_counts_out_of_range() -> None:
    t = pd.DataFrame(
        [_row(16, 0, 1, s, 1.8) for s in SEPARATIONS]
        + [_row(16, 1, 1, s, 0.4) for s in SEPARATIONS]
    )
    sp = _spread(_crossings(t))
    r = sp[(sp["n_axis"] == 16) & (sp["mode"] == 1)].iloc[0]
    assert r["below_lo"] == 1
    assert r["above_hi"] == 1
    assert r["n_crossing_inside"] == 0


def test_verdict_flags_insufficient_quartiles() -> None:
    """四分位区间不足时必须如实标注，不得假装可比。"""
    t = pd.DataFrame([_row(16, 0, 1, s, 1.8) for s in SEPARATIONS])
    v = verdict(type("R", (), {"spread": _spread(_crossings(t))})())
    assert "不足" in v.iloc[0]["判定"]


def test_same_config_seed_reproduces_observation_matrix() -> None:
    """★ 控制必须有效：同一配置编号、只改频率时**观测矩阵 `C` 逐位相同**。

    这是"只变模态间隔"这一对照成立的前提——否则交叉点的变化会同时混入
    `C` 的变化，无法判定是否为 `n` 的效应。

    注意：`X` 本身会因逐模态幅度配平（`C6`）而随频率变化，故只能比对 `C`。
    这也是 `C` 必须暴露在 `System.meta` 中的原因。
    """
    cfg = Config()
    m = 200
    for k in (0, 3):
        mats = []
        for sep in SEPARATIONS:
            sub = _config_for(cfg, 16, sep, m)
            rng = np.random.default_rng(k)      # 同一配置编号 -> 同一 C
            mats.append(build(sub, "real", rng, m).meta["C"])
        for other in mats[1:]:
            np.testing.assert_array_equal(mats[0], other)


def test_different_config_seeds_differ() -> None:
    """不同配置编号必须给出不同的观测矩阵（否则对照失效）。"""
    cfg = Config()
    m = 200
    sub = _config_for(cfg, 16, SEPARATIONS[0], m)
    a = build(sub, "real", np.random.default_rng(0), m).meta["C"]
    b = build(sub, "real", np.random.default_rng(1), m).meta["C"]
    assert not np.allclose(a, b)
