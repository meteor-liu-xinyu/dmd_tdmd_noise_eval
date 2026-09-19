"""实验一 · 偏差-方差分解（多轴）。

规格依据：docs/experiment-protocol.md §2；设计决策见 ADR-011 ~ ADR-014。

三条实验轴（由实测确定其必要性）：
  `n` / `L`  通道数或嵌入维数 —— 决定效应量能否被测出（algorithm-spec §7.1.4）
  SNR        信噪比        —— 偏差 ∝ σ²，所需 J 约按 10^(SNR/10) 变化
  幅度比     模态可观测能量比 —— 非平衡使所需 J 降低 3.6–8 倍（ADR-012）

报告口径 A/B 分组：
  A 组（SNR <= 0 dB）偏差可分辨 —— 报告点估计 + CI
  B 组（SNR > 0 dB）  偏差低于检测下限 —— **只报上界**，禁止方法间偏差对比
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from dmdnoise.config import Config, config_fingerprint
from dmdnoise.mc import CellResult, run_cell
from dmdnoise.metrics import paired_ratio, summarize

LOG = logging.getLogger(__name__)

METHODS = ("dmd", "tdmd")
M_DEFAULT = 200

#: 配对失败率门槛：超过此值的网格点存在选择偏差，结论不可靠
PAIR_FAIL_GATE = 0.20


@dataclass(frozen=True)
class Slice:
    """一个实验轴上的一个网格点。"""

    channel: str
    snr_db: float
    n: int | None = None          # real/complex 为传感器数；hankel 为嵌入维数 L
    amp_ratio: float = 1.0
    label: str = ""

    @property
    def group(self) -> str:
        return "A" if self.snr_db <= 0.0 else "B"

    @property
    def axis_n(self) -> int | None:
        return self.n

    def describe(self) -> str:
        return (f"{self.channel}/SNR{self.snr_db:g}/{self.label or 'n'}"
                f"{self.n}/r{self.amp_ratio:g}")


def default_slices(cfg: Config) -> list[Slice]:
    """默认网格：三条轴 + 三个通道。"""
    out: list[Slice] = []
    # 1) 通道数 n 扫描（complex / real）—— 决定效应可测性
    for ch in ("complex", "real"):
        for n in (4, 8, 16, 32, 64):
            out.append(Slice(ch, 10.0, n, 1.0, label="n"))
    # 2) SNR 扫描（complex 全档，real 取可分辨区）
    for snr in (20.0, 10.0, 5.0, 0.0, -5.0, -10.0):
        out.append(Slice("complex", snr, 8, 1.0, label="snr"))
    for snr in (10.0, 0.0, -5.0):
        out.append(Slice("real", snr, 8, 1.0, label="snr"))
    # 3) 幅度比扫描（ADR-012）
    for ratio in (1.0, 1.5, 2.0):
        out.append(Slice("complex", 0.0, 8, ratio, label="amp"))
    # 4) 时延嵌入维数扫描（单测点场景，ADR-013）
    for L in (4, 8, 16, 32):
        out.append(Slice("hankel", 10.0, L, 1.0, label="embed"))
    for snr in (10.0, 0.0, -5.0):
        out.append(Slice("hankel", snr, 8, 1.0, label="snr"))
    return out


@dataclass
class Exp1Result:
    rows: pd.DataFrame
    paired: pd.DataFrame
    fingerprint: str
    meta: dict[str, Any] = field(default_factory=dict)


def _cfg_for(cfg: Config, sl: Slice) -> Config:
    """按切片的 n 或 embed 生成变体配置（Config 为 frozen dataclass）。"""
    if sl.n is None:
        return cfg
    if sl.channel == "hankel":
        return replace(cfg, oscillator=replace(cfg.oscillator, embed_grid=(sl.n,)))
    return replace(cfg, oscillator=replace(cfg.oscillator, n=sl.n))


def _build_kwargs(sl: Slice) -> dict[str, Any]:
    kw: dict[str, Any] = {"amp_ratio": sl.amp_ratio, "axis_tag": sl.label}
    if sl.channel == "hankel":
        kw["embed"] = sl.n
    return kw


def _safe_summary(err: NDArray, bootstrap: int = 2000):
    """样本不足或含 NaN 时返回 None，而不是抛异常。

    配对失败（模态合并 / 配错）本身是一个有意义的指标——例如时延嵌入维数过小时，
    两个模态会被估计成一个。此时该实现不应计入偏差统计，但失败率必须报出来。
    """
    e = np.asarray(err, dtype=float).ravel()
    e = e[np.isfinite(e)]
    if e.size < 4:
        return None
    return summarize(e, bootstrap=bootstrap)


def _mode_rows(cell: CellResult, sl: Slice) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for method in METHODS:
        if method not in cell.estimates:
            continue
        for mode in range(cell.n_modes):
            err = cell.errors(method, mode=mode, relative=True)
            s = _safe_summary(err)
            base: dict[str, Any] = {
                "channel": cell.channel,
                "axis": sl.label,
                "group": sl.group,
                "snr_db": cell.snr_db,
                "n_axis": cell.meta.get("n"),
                "amp_ratio": sl.amp_ratio,
                "m": cell.m,
                "method": method,
                "mode": mode + 1,
                "f_true_hz": float(cell.f_true[mode]),
                "j_total": cell.j_total,
                "j_valid": int(err.size),
                "pair_fail_rate": 1.0 - err.size / max(cell.j_total, 1),
            }
            if s is None:
                out.append({**base, "insufficient": True,
                            "note": f"有效样本仅 {err.size}，判定为配对/分辨率失效"})
            else:
                out.append({**base, "insufficient": False, **s.as_row()})
    return out


def _paired_rows(cell: CellResult, sl: Slice) -> list[dict[str, Any]]:
    if not {"dmd", "tdmd"} <= set(cell.estimates):
        return []
    out: list[dict[str, Any]] = []
    for mode in range(cell.n_modes):
        d, t = cell.paired_errors(mode=mode, relative=True)
        if d.size < 10:
            out.append({
                "channel": cell.channel, "axis": sl.label, "snr_db": cell.snr_db,
                "n_axis": cell.meta.get("n"), "amp_ratio": sl.amp_ratio, "mode": mode + 1,
                "stat": "std", "insufficient": True,
                "note": f"两法公共有效配对仅 {d.size}，无法做配对比较",
            })
            continue
        sd = summarize(d, bootstrap=0)
        st = summarize(t, bootstrap=0)
        for stat in ("std", "mse"):
            pr = paired_ratio(t, d, stat=stat, bootstrap=2000)
            out.append({
                "channel": cell.channel,
                "axis": sl.label,
                "snr_db": cell.snr_db,
                "n_axis": cell.meta.get("n"),
                "amp_ratio": sl.amp_ratio,
                "mode": mode + 1,
                "stat": stat,
                "ratio": pr.point,
                "ci_lo": pr.ci[0],
                "ci_hi": pr.ci[1],
                "n_pair": pr.n,
                "variance_reduction": pr.variance_reduction,
                "paired_effective": pr.paired_effective,
                "bias_ratio": abs(st.bias) / abs(sd.bias) if sd.bias else float("nan"),
                "bias_ratio_valid": bool(sd.resolvable),
                "insufficient": False,
            })
    return out


def run(cfg: Config, *, slices: Sequence[Slice] | None = None,
        m: int = M_DEFAULT, j_total: int = 2000,
        progress: Any = None) -> Exp1Result:
    """运行实验一全部切片。"""
    sl_list = list(slices) if slices is not None else default_slices(cfg)
    rows: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []

    for idx, sl in enumerate(sl_list, 1):
        sub = _cfg_for(cfg, sl)
        kw = _build_kwargs(sl)
        LOG.info("[%d/%d] %s", idx, len(sl_list), sl.describe())
        if progress is not None:
            progress(idx, len(sl_list), sl.describe())
        cell = run_cell(sub, sl.channel, sl.snr_db, m, j_total, methods=METHODS, **kw)
        rows.extend(_mode_rows(cell, sl))
        paired.extend(_paired_rows(cell, sl))

    return Exp1Result(
        rows=pd.DataFrame(rows),
        paired=pd.DataFrame(paired),
        fingerprint=config_fingerprint(cfg),
        meta={"m": m, "j_total": j_total, "n_slices": len(sl_list)},
    )


def annotate_reportable(res: Exp1Result) -> pd.DataFrame:
    """按规格 §6 规则标注每个网格点的可报告口径。"""
    df = res.rows.copy()
    if df.empty:
        return df
    df["reportable"] = "点估计 + CI"
    df.loc[~df["resolvable"].eq(True), "reportable"] = "仅上界 delta_min"
    df.loc[df["insufficient"].eq(True), "reportable"] = "配对/分辨率失效，不计入偏差统计"
    # 选择偏差门槛：配对失败率高时，"存活样本"是估计恰好接近真值的子集，
    # 会把偏差系统性压低。此类网格点不得报告偏差。
    df.loc[df["pair_fail_rate"].fillna(0) > PAIR_FAIL_GATE, "reportable"] = (
        "配对失效率过高（选择偏差风险），不得报告偏差"
    )
    return df


def headline(res: Exp1Result) -> pd.DataFrame:
    """汇总：按 (通道, 轴, 网格点, 方法) 折叠模态，并给出配对失败率。"""
    df = annotate_reportable(res)
    if df.empty:
        return df
    keys = ["channel", "axis", "snr_db", "n_axis", "amp_ratio", "method"]
    g = df.groupby(keys, dropna=False)
    out = g.agg(
        bias_abs_max=("bias", lambda s: float(s.abs().max()) if s.notna().any() else np.nan),
        std_max=("std", "max"),
        delta_min_max=("delta_min", "max"),
        resolvable_any=("resolvable", lambda s: bool(s.eq(True).any())),
        required_j_min=("required_j", "min"),
        j_valid_min=("j_valid", "min"),
        pair_fail_rate_max=("pair_fail_rate", "max"),
    ).reset_index()
    out["reportable"] = np.where(out["resolvable_any"], "点估计 + CI", "仅上界 delta_min")
    out.loc[out["pair_fail_rate_max"].fillna(0) > PAIR_FAIL_GATE, "reportable"] = (
        "配对失效率过高（选择偏差风险），不得报告偏差"
    )
    return out
