"""实验一 · 偏差-方差分解。

规格依据：docs/experiment-protocol.md §2。

分两组（ADR-011）：
  A 组（偏差可分辨）  SNR ∈ {0, -5, -10} dB，J = 5000
  B 组（只报上界）    SNR ∈ {20, 10, 5} dB，J = 5000

B 组的偏差低于检测下限，按规格 §6 规则**只报上界**，不做方法间偏差对比。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import pandas as pd

from dmdnoise.config import Config, config_fingerprint
from dmdnoise.mc import CellResult, run_cell
from dmdnoise.metrics import paired_ratio, summarize

#: 默认网格（见 experiment-protocol.md §2.3）
SNR_RESOLVABLE = (0.0, -5.0, -10.0)
SNR_UPPER_BOUND = (20.0, 10.0, 5.0)
M_DEFAULT = 200
METHODS = ("dmd", "tdmd")


@dataclass
class Exp1Result:
    rows: pd.DataFrame
    paired: pd.DataFrame
    fingerprint: str
    meta: dict[str, Any] = field(default_factory=dict)


def _mode_rows(cell: CellResult, group: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for method in METHODS:
        if method not in cell.estimates:
            continue
        for mode in range(cell.n_modes):
            err = cell.errors(method, mode=mode, relative=True)
            s = summarize(err, bootstrap=2000)
            row = {
                "channel": cell.channel,
                "group": group,
                "snr_db": cell.snr_db,
                "m": cell.m,
                "method": method,
                "mode": mode + 1,
                "f_true_hz": float(cell.f_true[mode]),
                "j_total": cell.j_total,
                "j_valid": int(err.size),
                "sigma": cell.sigma,
                **s.as_row(),
            }
            out.append(row)
    return out


def _paired_rows(cell: CellResult) -> list[dict[str, Any]]:
    if not {"dmd", "tdmd"} <= set(cell.estimates):
        return []
    out: list[dict[str, Any]] = []
    for mode in range(cell.n_modes):
        d = cell.errors("dmd", mode=mode, relative=True)
        t = cell.errors("tdmd", mode=mode, relative=True)
        if d.size != t.size or d.size < 10:
            continue
        # 偏差点估计同样用相对量
        sd = summarize(d, bootstrap=0)
        st = summarize(t, bootstrap=0)
        for stat, name in (("std", "std"), ("mse", "mse")):
            pr = paired_ratio(t, d, stat=stat, bootstrap=2000)
            out.append({
                "channel": cell.channel,
                "snr_db": cell.snr_db,
                "m": cell.m,
                "mode": mode + 1,
                "stat": name,
                f"{name}_tdmd_over_dmd": pr.point,
                f"{name}_ci_lo": pr.ci[0],
                f"{name}_ci_hi": pr.ci[1],
                "n_pair": pr.n,
                "var_delta": pr.var_delta,
                "var_sum": pr.var_sum,
                "variance_reduction": pr.variance_reduction,
                "paired_effective": pr.paired_effective,
                # 偏差比：仅当 DMD 偏差自身可分辨时才有效
                "bias_ratio_tdmd_over_dmd": (
                    abs(st.bias) / abs(sd.bias) if sd.bias else float("nan")
                ),
                "bias_ratio_valid": bool(sd.resolvable),
            })
    return out


def run(
    cfg: Config,
    *,
    channels: Sequence[str] = ("real", "complex"),
    snr_resolvable: Iterable[float] = SNR_RESOLVABLE,
    snr_upper: Iterable[float] = SNR_UPPER_BOUND,
    m: int = M_DEFAULT,
    j_total: int | None = None,
    progress: Any = None,
) -> Exp1Result:
    """运行实验一。返回 (逐模态摘要, 配对比较, 配置指纹)。"""
    j = j_total if j_total is not None else cfg.grid.j_main
    rows: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []

    grid = [(g, s) for g, ss in (("A", snr_resolvable), ("B", snr_upper)) for s in ss]
    for channel in channels:
        for group, snr in grid:
            cell = run_cell(cfg, channel, float(snr), m, j, methods=METHODS,
                            progress=progress)
            rows.extend(_mode_rows(cell, group))
            paired.extend(_paired_rows(cell))

    return Exp1Result(
        rows=pd.DataFrame(rows),
        paired=pd.DataFrame(paired),
        fingerprint=config_fingerprint(cfg),
        meta={
            "m": m,
            "j_total": j,
            "channels": list(channels),
            "snr_resolvable": list(snr_resolvable),
            "snr_upper": list(snr_upper),
        },
    )


def decision_notes(res: Exp1Result) -> pd.DataFrame:
    """按规格 §6 规则标注每个网格点的可报告口径。"""
    df = res.rows.copy()
    if df.empty:
        return df
    df["reportable"] = df["resolvable"].map(
        {True: "点估计 + CI", False: "仅上界 delta_min"}
    )
    df.loc[df["group"] == "B", "reportable"] = "仅上界 delta_min（B 组）"
    return df[["channel", "group", "snr_db", "m", "method", "mode",
               "bias", "std", "delta_min", "resolvable", "required_j", "reportable"]]
