"""实验四 · 参数稳健性复核。

目的：现有结论建立在**一组固定参数**上（`f=(12.0,17.5)`、`ζ=(0.005,0.004)`、`m=200`）。
本实验检验这些结论对两个**最相关的物理参数**是否稳健：

  轴 A · 模态频率间隔 `f2/f1`
      实测（tmp 预扫描）显示间隔收紧会显著放大偏差，但那次越过了设计约束 C2。
      本实验在 **C2 硬边界（`f2/f1 ≥ 1.25`）之内**做正式扫描。

  轴 B · 窗内幅度衰减 `d = exp(-ζ·ω·m·dt)`
      它同时决定有效信噪比与被观测信号的能量分布，是 C3 约束的对象。
      直接扫描 `d` 比扫描 `ζ` 更可解释（`ζ` 还与频率耦合）。
      `ζ` 由目标衰减反解，故各点均满足 C3（配置的 `grid.m_max` 取实际使用的 `m`）。

不稳健则说明结论只在该参数点上成立，必须收窄适用范围声明。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

import numpy as np
import pandas as pd

from dmdnoise.config import Config, ConfigError, Mode, config_fingerprint, validate
from dmdnoise.mc import run_cell
from dmdnoise.metrics import paired_ratio, summarize

LOG = logging.getLogger(__name__)

METHODS = ("dmd", "tdmd")
PAIR_FAIL_GATE = 0.20

#: 轴 A：模态频率间隔（C2 硬要求 f2/f1 >= 1.25，故 1.25 即约束边界）
SEPARATIONS: tuple[float, ...] = (1.25, 1.40, 1.60, 1.83)
#: 轴 B：窗内幅度衰减比（C3 硬要求 >= 0.3）
WINDOW_DECAYS: tuple[float, ...] = (0.90, 0.70, 0.50, 0.35)
#: 扫描时固定 `f1`
F1_HZ = 12.0
N_DEFAULT = 8
SNR_DB = 10.0
M_DEFAULT = 200
J_DEFAULT = 5000


@dataclass(frozen=True)
class RobustPoint:
    """一个参数点。"""

    channel: str
    f2_over_f1: float
    window_decay: float
    n: int = N_DEFAULT
    snr_db: float = SNR_DB

    def describe(self) -> str:
        return (f"{self.channel} f2/f1={self.f2_over_f1:.2f} "
                f"decay={self.window_decay:.2f} n={self.n} SNR={self.snr_db:g}dB")


@dataclass
class Exp4Result:
    table: pd.DataFrame
    paired: pd.DataFrame
    constraints: pd.DataFrame
    fingerprint: str
    meta: dict[str, Any] = field(default_factory=dict)


def zeta_from_decay(f_hz: float, decay: float, m: int, dt: float) -> float:
    """由目标窗内幅度衰减比反解阻尼比：`ζ = -ln(d) / (2π f m Δt)`。"""
    if not 0.0 < decay < 1.0:
        raise ValueError("decay 必须落在 (0, 1)")
    return -math.log(decay) / (2.0 * math.pi * f_hz * m * dt)


def make_config(cfg: Config, pt: RobustPoint, m: int) -> tuple[Config, list]:
    """按参数点构造配置。

    `grid.m_min/m_max/m_scan` 一律取实际的 `m`：本实验不做样本量扫描，
    且 C3（窗内衰减）与 C4（最少周期数）都按 `m` 求值，须与实际窗口一致——
    否则扫描阻尼比时会因 `m_max=500` 而误触 C3 硬约束。
    """
    f2 = F1_HZ * pt.f2_over_f1
    # 逐模态反解 ζ，使【每个模态】的窗内衰减都等于目标值。
    # 若改用固定 ζ2/ζ1 比值，则 f2 越高衰减越快，长间隔点的模态 2 会先越过 C3 硬界
    # （实测 f2/f1=1.46、d=0.35 时模态 2 仅 0.293 < 0.3），使扫描网格自身越界。
    z1 = zeta_from_decay(F1_HZ, pt.window_decay, m, cfg.oscillator.dt)
    z2 = zeta_from_decay(f2, pt.window_decay, m, cfg.oscillator.dt)
    osc = replace(cfg.oscillator, modes=(Mode(F1_HZ, z1), Mode(f2, z2)), n=pt.n)
    grid = replace(cfg.grid, m_min=m, m_max=m, m_scan=(m,))
    sub = replace(cfg, oscillator=osc, grid=grid)
    # strict=False：让调用方检查约束结果，而不是在此直接抛异常
    results = validate(sub, strict=False)
    return sub, results


def default_points(cfg: Config) -> list[RobustPoint]:
    """默认参数点：复值全交叉 + 实值两条切片。"""
    out: list[RobustPoint] = []
    base_sep, base_decay = 1.46, 0.70
    for sep in SEPARATIONS:
        for dec in WINDOW_DECAYS:
            out.append(RobustPoint("complex", sep, dec, N_DEFAULT))
    for sep in SEPARATIONS:
        out.append(RobustPoint("real", sep, base_decay, 16))
    for dec in WINDOW_DECAYS:
        out.append(RobustPoint("real", base_sep, dec, 16))
    return out


def _safe_max(series) -> float:
    v = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    return float(v.max()) if v.size else float("nan")


def _rows(cell, pt: RobustPoint, results: Sequence, zeta1: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    warns = [r.name for r in results if r.warning]
    for method in METHODS:
        if method not in cell.estimates:
            continue
        for mode in range(cell.n_modes):
            err = cell.errors(method, mode=mode, relative=True)
            good = err[np.isfinite(err)]
            row: dict[str, Any] = {
                "channel": pt.channel,
                "f2_over_f1": pt.f2_over_f1,
                "window_decay": pt.window_decay,
                "n_axis": pt.n,
                "snr_db": pt.snr_db,
                "m": cell.m,
                "method": method,
                "mode": mode + 1,
                "j_total": cell.j_total,
                "j_valid": int(good.size),
                "pair_fail_rate": 1.0 - good.size / max(cell.j_total, 1),
                "zeta1": zeta1,
                "constraint_warnings": ";".join(warns),
            }
            if good.size >= 10:
                row.update(summarize(good, bootstrap=1000).as_row())
            out.append(row)
    return out


def _paired(cell, pt: RobustPoint) -> list[dict[str, Any]]:
    if not {"dmd", "tdmd"} <= set(cell.estimates):
        return []
    out: list[dict[str, Any]] = []
    for mode in range(cell.n_modes):
        d, t = cell.paired_errors(mode=mode, relative=True)
        if d.size < 10:
            continue
        sd = summarize(d, bootstrap=0)
        st = summarize(t, bootstrap=0)
        pr = paired_ratio(t, d, stat="std", bootstrap=2000)
        out.append({
            "channel": pt.channel, "f2_over_f1": pt.f2_over_f1,
            "window_decay": pt.window_decay, "mode": mode + 1,
            "std_ratio": pr.point, "std_lo": pr.ci[0], "std_hi": pr.ci[1],
            "n_pair": pr.n, "paired_effective": pr.paired_effective,
            "bias_ratio": abs(st.bias) / abs(sd.bias) if sd.bias else float("nan"),
            "bias_ratio_valid": bool(sd.resolvable),
            "dmd_resolvable": bool(sd.resolvable),
        })
    return out


def run(cfg: Config, *, points: Sequence[RobustPoint] | None = None,
        m: int = M_DEFAULT, j_total: int = J_DEFAULT,
        progress: Any = None) -> Exp4Result:
    pts = list(points) if points is not None else default_points(cfg)
    rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    cons_rows: list[dict[str, Any]] = []

    for i, pt in enumerate(pts, 1):
        sub, results = make_config(cfg, pt, m)
        failed = [r.name for r in results if not r.passed]
        if failed:
            raise ConfigError(
                f"参数点 {pt.describe()} 违反硬约束：{failed}——扫描范围须落在约束内"
            )
        if progress is not None:
            progress(i, len(pts), pt.describe())
        LOG.info("[%d/%d] %s", i, len(pts), pt.describe())

        cell = run_cell(sub, pt.channel, pt.snr_db, m, j_total, methods=METHODS)
        rows.extend(_rows(cell, pt, results, sub.oscillator.modes[0].zeta))
        pair_rows.extend(_paired(cell, pt))
        for r in results:
            cons_rows.append({
                "channel": pt.channel, "f2_over_f1": pt.f2_over_f1,
                "window_decay": pt.window_decay,
                "constraint": r.name, "passed": r.passed, "warning": r.warning,
                "detail": r.detail,
            })

    return Exp4Result(
        table=pd.DataFrame(rows),
        paired=pd.DataFrame(pair_rows),
        constraints=pd.DataFrame(cons_rows),
        fingerprint=config_fingerprint(cfg),
        meta={"m": m, "j_total": j_total, "n_points": len(pts),
              "separations": list(SEPARATIONS), "decays": list(WINDOW_DECAYS)},
    )


def verdict(res: Exp4Result) -> pd.DataFrame:
    """对结论的稳健性下判定：偏差量级、降偏比、可用性是否随参数变化。"""
    df = res.table
    if df.empty:
        return df
    out: list[dict[str, Any]] = []
    for (ch, sep, dec), g in df.groupby(["channel", "f2_over_f1", "window_decay"]):
        d = g[g["method"] == "dmd"]
        t = g[g["method"] == "tdmd"]
        bias_d = _safe_max(d["bias"].abs()) if "bias" in d else float("nan")
        bias_t = _safe_max(t["bias"].abs()) if "bias" in t else float("nan")
        fail = _safe_max(g["pair_fail_rate"])
        res_any = bool(d["resolvable"].eq(True).any()) if "resolvable" in d else False
        usable = bool(np.isfinite(bias_d) and fail <= PAIR_FAIL_GATE and res_any)
        out.append({
            "channel": ch, "f2_over_f1": sep, "window_decay": dec,
            "DMD_偏差": bias_d, "TDMD_偏差": bias_t,
            "降偏比": bias_t / bias_d if bias_d else float("nan"),
            "配对失败率": fail,
            "偏差可分辨": res_any,
            "可用": usable,
            "结论": ("TDMD 值得用" if usable else "两者都不行"),
        })
    return pd.DataFrame(out).sort_values(["channel", "f2_over_f1", "window_decay"])
