"""实验六（端到端）的正确性测试。

重点：
  1. 秩越界必须被夹紧（`r̂ > min(n, m)` 会让估计器直接抛异常）；
  2. **配对比较必须按同一实现 `j` 合并**——否则"退化是否显著"的 CI 不可信；
  3. 裁决表须正确区分"端到端可用"与"不可用"。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dmdnoise.estimators.rank import VALID_METHODS
from dmdnoise.experiments.exp6_end_to_end import (
    STRATEGIES,
    _clamp_rank,
    _compare,
    verdict,
)


def test_strategies_cover_oracle_and_all_criteria() -> None:
    assert STRATEGIES[0] == "oracle"
    assert set(STRATEGIES[1:]) == set(VALID_METHODS)


@pytest.mark.parametrize("raw,expect", [(-3, 1), (0, 1), (1, 1), (5, 5),
                                        (10_000, 8)])
def test_clamp_rank_bounds(raw: int, expect: int) -> None:
    """秩必须夹在 [1, min(n, m)] 内——越界会让估计器直接抛异常。"""
    assert _clamp_rank(raw, n=8, m=200) == expect


def test_compare_merges_by_realization() -> None:
    """★ 配对比较必须按实现 j 合并，不得按位置对齐。"""
    rows = []
    # oracle 与 data_driven 用【不同顺序】的 j，若按位置对齐会得到错误结果
    for j in range(60):
        rows.append({"channel": "complex", "n_axis": 8, "snr_db": 10.0,
                     "strategy": "oracle", "method": "tdmd", "j": j,
                     "err1": 0.001})
    for j in range(60):
        rows.append({"channel": "complex", "n_axis": 8, "snr_db": 10.0,
                     "strategy": "gavish_donoho", "method": "tdmd", "j": 59 - j,
                     "err1": 0.001})
    out = _compare(pd.DataFrame(rows))
    assert len(out) == 1
    # 真实值完全相同 -> 比值必须为 1，且不显著
    assert abs(out.iloc[0]["mse_ratio"] - 1.0) < 1e-9
    assert bool(out.iloc[0]["退化显著"]) is False
    assert out.iloc[0]["n_pair"] == 60


def test_compare_detects_significant_degradation() -> None:
    """数据驱动秩明显更差时，退化必须被判为显著。"""
    rng = np.random.default_rng(0)
    rows = []
    for j in range(400):
        base = 0.01 + 0.001 * rng.standard_normal()
        rows.append({"channel": "complex", "n_axis": 8, "snr_db": 10.0,
                     "strategy": "oracle", "method": "tdmd", "j": j, "err1": base})
        rows.append({"channel": "complex", "n_axis": 8, "snr_db": 10.0,
                     "strategy": "marchenko_pastur", "method": "tdmd", "j": j,
                     "err1": base * 3.0})       # 误差放大 3 倍 -> MSE 放大 9 倍
    out = _compare(pd.DataFrame(rows))
    assert bool(out.iloc[0]["退化显著"]) is True
    assert abs(out.iloc[0]["mse_ratio"] - 9.0) < 1.5


def test_compare_skips_insufficient_overlap() -> None:
    rows = [{"channel": "c", "n_axis": 8, "snr_db": 10.0, "strategy": "oracle",
             "method": "tdmd", "j": j, "err1": 0.01} for j in range(30)]
    rows += [{"channel": "c", "n_axis": 8, "snr_db": 10.0, "strategy": "energy",
              "method": "tdmd", "j": j, "err1": 0.02} for j in range(5)]
    out = _compare(pd.DataFrame(rows))
    assert out.empty, "共同实现不足 20 对时应跳过"


def test_compare_empty_input() -> None:
    assert _compare(pd.DataFrame()).empty


def _fake_result(table_rows, rank_rows) -> object:
    return type("R", (), {"table": pd.DataFrame(table_rows),
                          "ranks": pd.DataFrame(rank_rows)})()


def test_verdict_flags_unusable_when_pair_failures_high() -> None:
    table = pd.DataFrame([
        {"channel": "complex", "n_axis": 8, "snr_db": 10.0, "strategy": "energy",
         "method": "dmd", "mode": 1, "pair_fail_rate": 1.0, "resolvable": False,
         "mse_decomp": np.nan},
        {"channel": "complex", "n_axis": 8, "snr_db": 10.0, "strategy": "energy",
         "method": "tdmd", "mode": 1, "pair_fail_rate": 1.0, "resolvable": False,
         "mse_decomp": np.nan},
    ])
    ranks = pd.DataFrame([
        {"channel": "complex", "n_axis": 8, "snr_db": 10.0, "strategy": "energy",
         "p_exact": 0.0},
    ])
    v = verdict(_fake_result(table.to_dict("records"), ranks.to_dict("records")))
    assert bool(v.iloc[0]["端到端可用"]) is False


def test_verdict_flags_usable_when_clean() -> None:
    table = pd.DataFrame([
        {"channel": "complex", "n_axis": 8, "snr_db": 10.0, "strategy": "gavish_donoho",
         "method": "dmd", "mode": 1, "pair_fail_rate": 0.0, "resolvable": True,
         "mse_decomp": 1e-3},
        {"channel": "complex", "n_axis": 8, "snr_db": 10.0, "strategy": "gavish_donoho",
         "method": "tdmd", "mode": 1, "pair_fail_rate": 0.0, "resolvable": True,
         "mse_decomp": 1e-4},
    ])
    ranks = pd.DataFrame([
        {"channel": "complex", "n_axis": 8, "snr_db": 10.0, "strategy": "gavish_donoho",
         "p_exact": 1.0},
    ])
    v = verdict(_fake_result(table.to_dict("records"), ranks.to_dict("records")))
    assert bool(v.iloc[0]["端到端可用"]) is True
    assert abs(v.iloc[0]["TDMD_vs_DMD_MSE"] - 0.1) < 1e-9
