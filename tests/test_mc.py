"""蒙特卡洛层测试：T12（公共随机数自检）、嵌套设计、种子派生、配对公共子集。"""

from __future__ import annotations

import numpy as np
import pytest

from dmdnoise.config import Config
from dmdnoise.mc import CellResult, derive_rng, run_cell, run_m_scan


def test_derive_rng_is_deterministic_and_keyed() -> None:
    a1 = derive_rng(7, "complex", 10.0, 200, "noise", 0).standard_normal(4)
    a2 = derive_rng(7, "complex", 10.0, 200, "noise", 0).standard_normal(4)
    b = derive_rng(7, "complex", 10.0, 200, "noise", 1).standard_normal(4)
    c = derive_rng(8, "complex", 10.0, 200, "noise", 0).standard_normal(4)
    np.testing.assert_array_equal(a1, a2)
    assert not np.array_equal(a1, b)
    assert not np.array_equal(a1, c)


def test_cell_result_paired_errors_uses_common_subset() -> None:
    """配对误差必须取两法**共同**配对成功的子集。

    若两法各自过滤有效样本，失败集合不同会破坏配对结构，
    使 Var(delta) 自检失真、比值 CI 不可信。
    """
    est = {
        "dmd": np.array([[12.0], [12.1], [np.nan], [12.3]]),
        "tdmd": np.array([[12.0], [np.nan], [12.2], [12.3]]),
    }
    ok = {
        "dmd": np.array([True, True, False, True]),
        "tdmd": np.array([True, False, True, True]),
    }
    cell = CellResult(
        channel="complex", snr_db=0.0, m=10, j_total=4,
        f_true=np.array([12.0]), estimates=est, pair_ok=ok, sigma=0.1,
    )
    assert cell.n_paired == 2                       # 仅第 0、3 次两法都成功
    d, t = cell.paired_errors(mode=0, relative=True)
    assert d.size == t.size == 2
    np.testing.assert_allclose(d, [0.0, 0.025])
    np.testing.assert_allclose(t, [0.0, 0.025])


def test_cell_result_paired_errors_handles_no_overlap() -> None:
    est = {"dmd": np.array([[12.0]]), "tdmd": np.array([[12.0]])}
    ok = {"dmd": np.array([True]), "tdmd": np.array([False])}
    cell = CellResult("complex", 0.0, 10, 1, np.array([12.0]), est, ok, 0.1)
    d, t = cell.paired_errors()
    assert d.size == 0 and t.size == 0
    assert cell.n_paired == 0


def test_run_cell_shares_noise_between_methods(cfg: Config) -> None:
    """T12：公共随机数下两法必须面对同一条噪声实现。

    判据：两法估计的相关系数显著为正（若各自独立加噪，相关性应接近 0）。
    """
    cell = run_cell(cfg, "complex", 10.0, m=200, j_total=200)
    d, t = cell.paired_errors(mode=0)
    assert d.size == t.size >= 100
    r = float(np.corrcoef(d, t)[0, 1])
    assert r > 0.5, f"两法估计相关性仅 {r:.3f}，公共随机数可能未生效"


def test_run_cell_reports_axis_metadata(cfg: Config) -> None:
    cell = run_cell(cfg, "hankel", 10.0, m=120, j_total=30, embed=16, amp_ratio=1.5)
    assert cell.meta["embed"] == 16
    assert cell.meta["amp_ratio"] == 1.5
    assert cell.meta["n"] == 16          # hankel 通道下 n 即嵌入维数
    assert cell.channel == "hankel"


def test_nested_m_scan_returns_all_prefix_lengths(cfg: Config) -> None:
    ms = (50, 100, 200)
    out = run_m_scan(cfg, "complex", 10.0, ms, j_total=30)
    assert set(out) == set(ms)
    for mm, cell in out.items():
        assert cell.m == mm
        assert cell.meta["nested"] is True
        assert cell.meta["m_max"] == max(ms)
    # 嵌套设计下真值一致（同一系统实例的相位随机化不改变真值）
    for mm in ms[1:]:
        np.testing.assert_allclose(out[mm].f_true, out[ms[0]].f_true)


def test_nested_prefix_is_consistent(cfg: Config) -> None:
    """嵌套前缀：同一实现下不同 m 的信号必须来自同一条序列。"""
    from dmdnoise.sim import build

    m_max = 200
    rng = derive_rng(cfg.run.seed, "complex", 0.0, "scan", "phase", 0)
    full = build(cfg, "complex", rng, m_max)
    for mm in (50, 100, 200):
        np.testing.assert_allclose(full.X[:, :mm][:, :50], full.X[:, :50])


def test_pair_failure_rate_is_reported(cfg: Config) -> None:
    """配对失败率必须可获取——失败率高时存在选择偏差，结论不可靠。"""
    cell = run_cell(cfg, "complex", -5.0, m=200, j_total=100)
    for method in ("dmd", "tdmd"):
        rate = 1.0 - cell.pair_ok[method].mean()
        assert 0.0 <= rate <= 1.0
    assert "pair_fail" in cell.meta
