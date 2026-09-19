"""配置层测试：判据 T10（约束断言）、T14（警戒带触发）与指纹稳定性。"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from dmdnoise.config import (
    Config,
    ConfigError,
    Mode,
    OscillatorConfig,
    check_c6,
    config_fingerprint,
    constraint_summary,
    load_config,
    save_fingerprint,
    validate,
)


def test_default_config_passes_all_hard_constraints(cfg: Config) -> None:
    """T10：默认配置必须全部通过硬约束。"""
    results = validate(cfg)
    assert results, "应返回约束结果清单"
    failed = [r.name for r in results if not r.passed]
    assert not failed, f"默认配置不应违反硬约束，实际违反：{failed}"


def test_default_config_triggers_expected_warning_bands(cfg: Config) -> None:
    """T14：当前参数集必须触发 C3、C4 两条警戒（已知固有性质）。"""
    warned = {r.name for r in validate(cfg) if r.warning}
    assert any(w.startswith("C3") for w in warned), f"C3 应进入警戒带，实际 {warned}"
    assert any(w.startswith("C4") for w in warned), f"C4 应进入警戒带，实际 {warned}"


def test_c1_rejects_frequency_outside_band() -> None:
    bad = Config(oscillator=replace(OscillatorConfig(), modes=(Mode(0.5, 0.005), Mode(0.6, 0.004))))
    with pytest.raises(ConfigError, match="违反硬约束"):
        validate(bad)


def test_c2_rejects_too_close_modes() -> None:
    bad = Config(oscillator=replace(OscillatorConfig(), modes=(Mode(12.0, 0.005), Mode(12.5, 0.004))))
    with pytest.raises(ConfigError, match="违反硬约束"):
        validate(bad)


def test_c3_rejects_excessive_decay() -> None:
    bad = Config(oscillator=replace(OscillatorConfig(),
                                    modes=(Mode(12.0, 0.30), Mode(17.5, 0.30))))
    with pytest.raises(ConfigError, match="违反硬约束"):
        validate(bad)


def test_c5_rejects_insufficient_channels() -> None:
    bad = Config(oscillator=replace(OscillatorConfig(), n=8, r_real=4))
    with pytest.raises(ConfigError, match="违反硬约束"):
        validate(bad)


def test_structural_invariants_rejected() -> None:
    with pytest.raises(ConfigError):
        validate(Config(noise=replace(Config().noise, mode="bogus")))
    with pytest.raises(ConfigError):
        validate(Config(noise=replace(Config().noise, convention="bogus")))
    with pytest.raises(ConfigError):
        validate(Config(grid=replace(Config().grid, m_min=2000, m_max=500)))


def test_c6_helper_thresholds() -> None:
    assert check_c6(1.0).passed and not check_c6(1.0).warning
    assert check_c6(1.5).passed and check_c6(1.5).warning       # 警戒带
    assert not check_c6(3.0).passed                              # 硬失败


def test_fingerprint_is_deterministic_and_sensitive(cfg: Config) -> None:
    assert config_fingerprint(cfg) == config_fingerprint(Config())
    assert len(config_fingerprint(cfg)) == 16
    changed = Config(oscillator=replace(cfg.oscillator, n=128))
    assert config_fingerprint(changed) != config_fingerprint(cfg)


def test_load_config_extends_and_roundtrip(tmp_path) -> None:
    (tmp_path / "base.yaml").write_text(
        "oscillator:\n  fs: 200.0\n  n: 64\ngrid:\n  j_main: 1234\n", encoding="utf-8"
    )
    (tmp_path / "exp.yaml").write_text(
        "extends: base.yaml\noscillator:\n  n: 96\n", encoding="utf-8"
    )
    cfg, results = load_config(tmp_path / "exp.yaml")
    assert cfg.oscillator.n == 96          # 子配置覆盖
    assert cfg.oscillator.fs == 200.0      # 父配置保留
    assert cfg.grid.j_main == 1234
    assert all(r.passed for r in results)


def test_save_fingerprint_payload(tmp_path, cfg: Config) -> None:
    out = tmp_path / "meta.json"
    save_fingerprint(cfg, validate(cfg), out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["fingerprint"] == config_fingerprint(cfg)
    assert payload["constraints"]["all_passed"] is True
    assert any(k.startswith("C3") for k in payload["constraints"]["items"])


def test_constraint_summary_shape(cfg: Config) -> None:
    summary = constraint_summary(validate(cfg))
    assert set(summary) == {"all_passed", "warnings", "items"}
    assert len(summary["warnings"]) >= 2
