"""配置 schema、加载、指纹与设计约束校验。

规格依据：docs/algorithm-spec.md §2.2（六条设计约束 C1–C6）、§5.3 配置架构。

约束分两档：
  - 硬阈值：违反即抛 ConfigError
  - 警戒带：进入即 logging.warning 并写入结果元数据
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

LOG = logging.getLogger(__name__)

# --------------------------------------------------------------------------- 约束常量
C1_RANGE = (0.05, 0.45)          # f_i / f_s 允许区间
C2_HARD, C2_WARN = 0.25, 0.30    # 模态可分辨性
C3_HARD, C3_WARN = 0.30, 0.40    # 最长窗内幅度衰减
C4_HARD, C4_WARN = 3.0, 3.5      # 最短窗内低频模态周期数
C6_HARD = (0.5, 2.0)             # 模态可观测能量比
C6_WARN = (0.7, 1.4)

VALID_NOISE_MODES = ("trajectory", "independent")
VALID_SNR_CONVENTIONS = ("aggregate", "fixed_sigma")


class ConfigError(ValueError):
    """配置违反硬约束。"""


# --------------------------------------------------------------------------- schema
@dataclass(frozen=True)
class Mode:
    """单阶模态：固有频率与阻尼比。"""

    f: float
    zeta: float


@dataclass(frozen=True)
class OscillatorConfig:
    fs: float = 200.0
    modes: tuple[Mode, ...] = (Mode(12.0, 0.005), Mode(17.5, 0.004))
    n: int = 64
    r_real: int = 4
    r_complex: int = 2
    c_balance_tol: float = 0.10

    @property
    def dt(self) -> float:
        return 1.0 / self.fs

    @property
    def freqs(self) -> tuple[float, ...]:
        return tuple(m.f for m in self.modes)


@dataclass(frozen=True)
class GridConfig:
    m_min: int = 50
    m_max: int = 500
    m_scan: tuple[int, ...] = (50, 75, 100, 150, 200, 300, 400, 500)
    snr_main: tuple[float, ...] = (20.0, 10.0, 5.0)
    snr_ext: tuple[float, ...] = (30.0, 25.0, 20.0, 15.0, 10.0, 7.0, 5.0, 3.0)
    j_main: int = 5000
    j_key: int = 20000
    j_coarse: int = 1000
    sig_level: float = 0.05


@dataclass(frozen=True)
class NoiseConfig:
    mode: str = "trajectory"
    convention: str = "aggregate"


@dataclass(frozen=True)
class RunConfig:
    seed: int = 20260919
    bootstrap_resamples: int = 10000
    out_dir: str = "results"


@dataclass(frozen=True)
class Config:
    oscillator: OscillatorConfig = field(default_factory=OscillatorConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    run: RunConfig = field(default_factory=RunConfig)


# --------------------------------------------------------------------------- 指纹
def _plain(obj: Any) -> Any:
    """把 dataclass 树转成可 JSON 化的纯结构（保持键序稳定）。"""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(_plain(v) for v in obj)
    return obj


def config_fingerprint(cfg: Config) -> str:
    """规范 JSON 的 SHA-256 前 16 位。结果文件必须携带该值。"""
    payload = json.dumps(
        _plain(cfg), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- 约束检查
@dataclass(frozen=True)
class ConstraintResult:
    name: str
    value: str
    passed: bool
    warning: bool
    detail: str = ""


def _check_c1(cfg: OscillatorConfig) -> ConstraintResult:
    lo, hi = C1_RANGE
    ratios = [m.f / cfg.fs for m in cfg.modes]
    ok = all(lo <= r <= hi for r in ratios)
    txt = " / ".join(f"{r:.4f}" for r in ratios)
    return ConstraintResult(
        "C1 避开频率边界",
        txt,
        ok,
        not ok or any(r < lo * 1.1 or r > hi * 0.9 for r in ratios),
        f"要求 {lo} <= f_i/f_s <= {hi}",
    )


def _check_c2(cfg: OscillatorConfig) -> ConstraintResult:
    freqs = sorted(cfg.freqs)
    if len(freqs) < 2:
        return ConstraintResult("C2 模态可分辨性", "n/a", True, False, "模态少于 2 阶")
    gaps = [freqs[i + 1] - freqs[i] for i in range(len(freqs) - 1)]
    rel = min(gaps) / freqs[0]
    return ConstraintResult(
        "C2 模态可分辨性",
        f"{rel:.4f}",
        rel >= C2_HARD,
        C2_HARD <= rel < C2_WARN,
        f"要求相邻频率间隔 / min(f) >= {C2_HARD}",
    )


def _check_c3(cfg: OscillatorConfig, m_max: int) -> ConstraintResult:
    vals = []
    for m in cfg.modes:
        w = 2.0 * math.pi * m.f
        vals.append(math.exp(-m.zeta * w * m_max * cfg.dt))
    worst = min(vals)
    return ConstraintResult(
        "C3 窗内幅度衰减",
        " / ".join(f"{v:.4f}" for v in vals),
        worst >= C3_HARD,
        C3_HARD <= worst < C3_WARN,
        f"要求 exp(-zeta*w*m_max*dt) >= {C3_HARD}（m_max={m_max}）",
    )


def _check_c4(cfg: OscillatorConfig, m_min: int) -> ConstraintResult:
    vals = [m.f * m_min * cfg.dt for m in cfg.modes]
    worst = min(vals)
    return ConstraintResult(
        "C4 最少周期数",
        " / ".join(f"{v:.2f}" for v in vals),
        worst >= C4_HARD,
        C4_HARD <= worst < C4_WARN,
        f"要求 min_k f_k*m_min*dt >= {C4_HARD}（m_min={m_min}）",
    )


def _check_c5(cfg: OscillatorConfig) -> ConstraintResult:
    need = max(4 * cfg.r_real, 2 * cfg.r_complex)
    return ConstraintResult(
        "C5 空间维数充分",
        f"n={cfg.n}, 4r={4 * cfg.r_real}",
        cfg.n >= need,
        False,
        "要求 n >= 4r",
    )


def check_c6(energy_ratio: float, tol: float = 0.10) -> ConstraintResult:
    """C6 模态可观测性平衡。需在仿真器构造出数据后调用。

    energy_ratio = E_max / E_min（两模态可观测能量之比）。
    """
    ok = C6_HARD[0] <= energy_ratio <= C6_HARD[1]
    warn = not (1.0 - tol <= energy_ratio <= 1.0 + tol)
    return ConstraintResult(
        "C6 模态可观测性平衡",
        f"{energy_ratio:.4f}",
        ok,
        warn,
        f"要求 E_max/E_min ∈ {C6_HARD}，构造目标 |比值-1| <= {tol}",
    )


def validate(cfg: Config, *, strict: bool = True) -> list[ConstraintResult]:
    """校验 C1–C5（纯参数约束）。C6 需数据，见 `check_c6`。

    strict=True 时任一硬约束失败即抛 ConfigError。
    返回全部结果（含通过项），供写入结果元数据。
    """
    results = [
        _check_c1(cfg.oscillator),
        _check_c2(cfg.oscillator),
        _check_c3(cfg.oscillator, cfg.grid.m_max),
        _check_c4(cfg.oscillator, cfg.grid.m_min),
        _check_c5(cfg.oscillator),
    ]

    # 结构性约束
    if cfg.oscillator.fs <= 0:
        raise ConfigError("fs 必须为正")
    if cfg.grid.m_min < 2 * max(cfg.oscillator.r_real, cfg.oscillator.r_complex):
        raise ConfigError("m_min 过小，无法支撑截断秩")
    if cfg.grid.m_min > cfg.grid.m_max:
        raise ConfigError("m_min 不得大于 m_max")
    if tuple(sorted(cfg.grid.m_scan)) != cfg.grid.m_scan:
        raise ConfigError("m_scan 必须严格递增")
    if not (cfg.grid.m_min <= cfg.grid.m_scan[0] and cfg.grid.m_scan[-1] <= cfg.grid.m_max):
        raise ConfigError("m_scan 必须落在 [m_min, m_max] 内")
    if cfg.noise.mode not in VALID_NOISE_MODES:
        raise ConfigError(f"noise.mode 必须为 {VALID_NOISE_MODES} 之一")
    if cfg.noise.convention not in VALID_SNR_CONVENTIONS:
        raise ConfigError(f"noise.convention 必须为 {VALID_SNR_CONVENTIONS} 之一")
    if cfg.grid.j_coarse > cfg.grid.j_main > cfg.grid.j_key:
        raise ConfigError("要求 j_coarse <= j_main <= j_key")

    failures = [r for r in results if not r.passed]
    warns = [r for r in results if r.warning]
    for r in warns:
        LOG.warning("设计约束进入警戒带：%s = %s（%s）", r.name, r.value, r.detail)
    if failures and strict:
        msg = "；".join(f"{r.name}={r.value}（{r.detail}）" for r in failures)
        raise ConfigError(f"违反硬约束：{msg}")
    return results


def constraint_summary(results: list[ConstraintResult]) -> dict[str, Any]:
    """压缩成可写入结果元数据的结构。"""
    return {
        "all_passed": all(r.passed for r in results),
        "warnings": [r.name for r in results if r.warning],
        "items": {r.name: {"value": r.value, "passed": r.passed, "warning": r.warning}
                  for r in results},
    }


# --------------------------------------------------------------------------- 加载
def _nested_dataclass(cls: type) -> dict[str, type]:
    """返回字段名 -> 嵌套 dataclass 类型 的映射（含 default / default_factory）。"""
    out: dict[str, type] = {}
    for f in fields(cls):
        cand: Any = None
        if f.default is not MISSING:
            cand = f.default
        elif f.default_factory is not MISSING:  # type: ignore[misc]
            try:
                cand = f.default_factory()
            except Exception:  # pragma: no cover - 默认工厂不应失败
                cand = None
        if cand is not None and is_dataclass(cand) and not isinstance(cand, type):
            out[f.name] = type(cand)
    return out


def _construct(cls: type, data: dict[str, Any]) -> Any:
    """按 dataclass schema 递归构造；嵌套 dataclass 逐层展开，元组字段还原为 tuple。"""
    if not isinstance(data, dict):
        return data
    nested = _nested_dataclass(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        raw = data[f.name]
        if raw is None:
            continue
        if f.name == "modes" and isinstance(raw, list):
            kwargs[f.name] = tuple(_construct(Mode, dict(m) if isinstance(m, dict) else m)
                                   for m in raw)
        elif f.name in nested and isinstance(raw, dict):
            kwargs[f.name] = _construct(nested[f.name], raw)
        elif isinstance(raw, (list, tuple)):
            kwargs[f.name] = tuple(raw)
        else:
            kwargs[f.name] = raw
    return cls(**kwargs)


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path, *, strict: bool = True) -> tuple[Config, list[ConstraintResult]]:
    """从 YAML 加载配置。支持顶层 `extends: base.yaml`（相对同目录）。"""
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    parent = raw.pop("extends", None)
    if parent:
        pp = p.parent / parent
        with pp.open("r", encoding="utf-8") as fh:
            base = yaml.safe_load(fh) or {}
        base.pop("extends", None)
        raw = _deep_merge(base, raw)

    cfg = _construct(Config, raw)
    results = validate(cfg, strict=strict)
    return cfg, results


def save_fingerprint(cfg: Config, results: list[ConstraintResult], path: str | Path) -> None:
    """把配置指纹与约束校验结果写入 JSON，供结果追溯。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fingerprint": config_fingerprint(cfg),
        "version": __import__("dmdnoise").__version__,
        "config": _plain(cfg),
        "constraints": constraint_summary(results),
    }
    with p.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, sort_keys=True)
