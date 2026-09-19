"""统计层测试：T8（指标实现正确性）、T12（公共随机数自检）。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dmdnoise.metrics import (
    MetricsError,
    paired_ratio,
    required_j,
    summarize,
)


def test_summarize_matches_analytic_values() -> None:
    """T8：bias/std/MSE 必须与解析值一致（人工构造的样本）。"""
    rng = np.random.default_rng(0)
    n = 20000
    true_bias, true_std = 0.3, 1.0
    d = true_bias + true_std * rng.standard_normal(n)

    s = summarize(d, bootstrap=0)
    assert s.n == n
    assert abs(s.bias - true_bias) < 5e-3
    assert abs(s.std - true_std) < 5e-3
    assert abs(s.mse_decomp - (s.bias**2 + s.std**2)) < 1e-15
    assert abs(s.mse_plugin - float(np.mean(d**2))) < 1e-15
    # 大样本下两种 MSE 形式应一致
    assert abs(s.mse_plugin - s.mse_decomp) < 0.05


def test_normalization_scales_all_outputs() -> None:
    d = np.linspace(-1.0, 1.0, 1001)
    a = summarize(d, bootstrap=0)
    b = summarize(d, bootstrap=0, normalize=10.0)
    assert abs(b.bias - a.bias / 10.0) < 1e-15
    assert abs(b.std - a.std / 10.0) < 1e-15


def test_delta_min_and_resolvability_rule() -> None:
    """分辨规则：|bias|/std > 3.92/sqrt(J) 才可点估计，否则只报上界。"""
    rng = np.random.default_rng(1)
    n = 100

    weak = summarize(0.005 * rng.standard_normal(n), bootstrap=0)
    assert not weak.resolvable
    assert weak.note and "只报上界" in weak.note
    assert weak.bias_upper == pytest.approx(weak.delta_min)

    strong = summarize(1.0 + 0.01 * rng.standard_normal(n), bootstrap=0)
    assert strong.resolvable
    assert strong.note == ""


def test_delta_min_formula() -> None:
    d = np.array([1.0, 2.0, 3.0, 4.0])
    s = summarize(d, bootstrap=0)
    assert s.delta_min == pytest.approx(1.96 * s.std / math.sqrt(4))
    assert s.required_j == pytest.approx(required_j(s.bias, s.std))


def test_required_j_formula() -> None:
    # 50% 分辨：J = (1.96/0.5)^2 * (std/bias)^2 = 15.3664 * (std/bias)^2
    j = required_j(1.0, 0.1)
    assert j == pytest.approx(15.3664 * 0.01, rel=1e-6)
    assert required_j(0.0, 1.0) == math.inf
    assert required_j(1.0, 0.0) == 1.0


def test_bootstrap_ci_contains_point_estimate() -> None:
    rng = np.random.default_rng(2)
    d = 0.2 + rng.standard_normal(500)
    s = summarize(d, bootstrap=2000, rng=rng)
    assert s.bias_ci[0] < s.bias < s.bias_ci[1]
    assert s.std_ci[0] < s.std < s.std_ci[1]
    assert s.std_ci[1] - s.std_ci[0] > 0


def test_paired_ratio_detects_shared_structure() -> None:
    """公共随机数下，配对差方差应远小于两法方差之和（T12 自检）。"""
    rng = np.random.default_rng(3)
    n = 5000
    common = rng.standard_normal(n)
    a = 0.02 * common + 0.001 * rng.standard_normal(n)   # 两法共享同一实现效应
    b = 0.02 * common + 0.001 * rng.standard_normal(n)
    pr = paired_ratio(a, b, stat="std", bootstrap=1000, rng=rng)
    assert pr.paired_effective is True, pr.note
    assert pr.variance_reduction > 0.5
    assert abs(pr.point - 1.0) < 0.05


def test_paired_ratio_flags_independent_samples() -> None:
    """两法独立抽样时配对无效，必须被标记（防止误以为配对生效）。"""
    rng = np.random.default_rng(4)
    a = rng.standard_normal(3000)
    b = rng.standard_normal(3000)
    pr = paired_ratio(a, b, stat="std", bootstrap=500, rng=rng)
    assert pr.paired_effective is False
    assert "方差消减" in pr.note


def test_paired_ratio_mse_statistic() -> None:
    rng = np.random.default_rng(5)
    a = 0.1 * rng.standard_normal(2000)
    b = 2.0 * a
    pr = paired_ratio(a, b, stat="mse", bootstrap=200, rng=rng)
    assert pr.point == pytest.approx(0.25, rel=0.15)


def test_metrics_input_validation() -> None:
    with pytest.raises(MetricsError):
        summarize(np.array([1.0]))
    with pytest.raises(MetricsError):
        summarize(np.array([1.0, np.nan, 2.0]))
    with pytest.raises(MetricsError):
        summarize(np.array([1.0, 2.0]), normalize=0.0)
    with pytest.raises(MetricsError):
        paired_ratio(np.arange(5.0), np.arange(4.0))
    with pytest.raises(MetricsError):
        paired_ratio(np.zeros(5), np.zeros(5))
