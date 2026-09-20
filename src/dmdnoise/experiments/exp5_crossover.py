"""实验五 · 交叉点定位：降偏比 ρ 随模态间隔的变化。

背景（实验四发现）：DMD 的偏差随模态间隔拉开而快速下降，而 TDMD 的残余偏差几乎恒定，
两条曲线在某处相交——**"TDMD 值得用"的边界不是噪声水平决定的，而是几何决定的**。

本实验把这个交叉点**定准**，并给出它与观测维数 `n` 的关系：

    定义   ρ(f2/f1) = |bias_TDMD| / |bias_DMD|      （逐模态）
    交叉点 ρ = 1 的位置

**统计口径**：ρ 的置信区间必须用**配对 bootstrap**（按实现索引整体重采样）。
两法共享同一条噪声序列，配对后差值的方差远小于各自方差，
故 ρ 的 CI 比"分别统计再相除"窄得多——这是能否定准交叉点的关键。

网格须落在设计约束之内（C2 要求 `f2/f1 ≥ 1.25`）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from dmdnoise.config import Config, config_fingerprint
from dmdnoise.mc import run_cell
from dmdnoise.metrics import summarize

LOG = logging.getLogger(__name__)

#: 细网格：覆盖实验四观察到的交叉点附近
SEPARATIONS: tuple[float, ...] = (1.50, 1.60, 1.70, 1.75, 1.80, 1.85, 1.90, 2.00)
#: (通道, 通道数) 组合——用于给出交叉点与 n 的关系
COMBOS: tuple[tuple[str, int], ...] = (
    ("complex", 8), ("complex", 16),
    ("real", 8), ("real", 16), ("real", 32),
)
F1_HZ = 12.0
WINDOW_DECAY = 0.70          # 固定：实验四显示它几乎不影响偏差
SNR_DB = 10.0
M_DEFAULT = 200
J_DEFAULT = 3000
BOOTSTRAP = 4000
PAIR_FAIL_GATE = 0.20


@dataclass
class Exp5Result:
    table: pd.DataFrame          # 逐 (通道, n, f2/f1, 模态) 的 ρ 与 CI
    crossing: pd.DataFrame       # 逐 (通道, n, 模态) 的交叉点估计
    fingerprint: str
    meta: dict[str, Any] = field(default_factory=dict)


def zeta_from_decay(f_hz: float, decay: float, m: int, dt: float) -> float:
    import math
    return -math.log(decay) / (2.0 * math.pi * f_hz * m * dt)


def rho_ci(d: NDArray, t: NDArray, *, bootstrap: int = BOOTSTRAP,
           rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    """降偏比 ρ = |bias_TDMD| / |bias_DMD| 及其配对 bootstrap 置信区间。

    d / t 为两法在**共同有效子集**上的误差序列（同长度、逐实现对应）。
    """
    if d.size < 10 or d.size != t.size:
        return float("nan"), float("nan"), float("nan")
    bd, bt = float(d.mean()), float(t.mean())
    if bd == 0.0:
        return float("nan"), float("nan"), float("nan")
    point = abs(bt) / abs(bd)
    gen = rng if rng is not None else np.random.default_rng(0)
    pos = gen.integers(0, d.size, size=(bootstrap, d.size))
    sb = d[pos].mean(axis=1)
    st = t[pos].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.abs(st) / np.abs(sb)
    r = r[np.isfinite(r)]
    if r.size < 100:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(r, [2.5, 97.5])
    return point, float(lo), float(hi)


def _cell_config(cfg: Config, channel: str, n: int, sep: float, m: int) -> Config:
    from dmdnoise.config import Mode

    f2 = F1_HZ * sep
    z1 = zeta_from_decay(F1_HZ, WINDOW_DECAY, m, cfg.oscillator.dt)
    z2 = zeta_from_decay(f2, WINDOW_DECAY, m, cfg.oscillator.dt)
    osc = replace(cfg.oscillator, modes=(Mode(F1_HZ, z1), Mode(f2, z2)), n=n)
    grid = replace(cfg.grid, m_min=m, m_max=m, m_scan=(m,))
    return replace(cfg, oscillator=osc, grid=grid)


def _pair_fail(cell) -> float:
    ok = cell.pair_ok["dmd"] & cell.pair_ok["tdmd"]
    return 1.0 - float(ok.mean())


def run(cfg: Config, *, combos: Sequence[tuple[str, int]] = COMBOS,
        separations: Sequence[float] = SEPARATIONS, m: int = M_DEFAULT,
        j_total: int = J_DEFAULT, progress: Any = None) -> Exp5Result:
    rows: list[dict[str, Any]] = []
    total = len(combos) * len(separations)
    k = 0

    for channel, n in combos:
        for sep in separations:
            k += 1
            if progress is not None:
                progress(k, total, f"{channel} n={n} f2/f1={sep:.2f}")
            sub = _cell_config(cfg, channel, n, sep, m)
            cell = run_cell(sub, channel, SNR_DB, m, j_total,
                            methods=("dmd", "tdmd"))
            for mode in range(cell.n_modes):
                d, t = cell.paired_errors(mode=mode, relative=True)
                rho, lo, hi = rho_ci(d, t)
                sd = summarize(d, bootstrap=0) if d.size >= 4 else None
                st = summarize(t, bootstrap=0) if t.size >= 4 else None
                rows.append({
                    "channel": channel, "n": n, "f2_over_f1": sep,
                    "mode": mode + 1, "j_total": j_total, "j_pair": int(d.size),
                    "pair_fail_rate": _pair_fail(cell),
                    "bias_dmd": float(d.mean()) if d.size else np.nan,
                    "bias_tdmd": float(t.mean()) if t.size else np.nan,
                    "dmd_resolvable": bool(sd.resolvable) if sd else False,
                    "tdmd_resolvable": bool(st.resolvable) if st else False,
                    "rho": rho, "rho_lo": lo, "rho_hi": hi,
                    "covers_one": bool(np.isfinite(lo) and lo <= 1.0 <= hi),
                })

    table = pd.DataFrame(rows)
    return Exp5Result(
        table=table,
        crossing=_crossings(table),
        fingerprint=config_fingerprint(cfg),
        meta={"m": m, "j_total": j_total, "snr_db": SNR_DB,
              "window_decay": WINDOW_DECAY, "separations": list(separations),
              "combos": [list(c) for c in combos]},
    )


def _crossings(table: pd.DataFrame) -> pd.DataFrame:
    """逐 (通道, n, 模态) 定位 ρ 穿越 1 的位置。

    判定逻辑（先找**夹住 1 的相邻网格点对**，再看这对是否足够确定）：

      ρ 在扫描范围内始终 < 1  -> 交叉点在扫描上界之外
      ρ 在下界点已 >= 1        -> 交叉点在扫描下界之外
      否则取 y[i-1] < 1 <= y[i] 的那一对，线性内插给点估计；
      若这对两端的 CI 都不含 1，则交叉点**确定**，区间即两网格点之间。
    """
    out: list[dict[str, Any]] = []
    for (ch, n, mode), g in table.groupby(["channel", "n", "mode"]):
        g = g.sort_values("f2_over_f1")
        usable = g[(g["pair_fail_rate"] <= PAIR_FAIL_GATE)
                   & (g["dmd_resolvable"]) & np.isfinite(g["rho"])].reset_index(drop=True)
        base = {"channel": ch, "n": n, "mode": mode}
        if len(usable) < 3:
            out.append({**base, "crossing": np.nan, "crossing_lo": np.nan,
                        "crossing_hi": np.nan, "well_determined": False,
                        "note": f"可报告网格点不足（{len(usable)} 个）"})
            continue

        x = usable["f2_over_f1"].to_numpy(float)
        y = usable["rho"].to_numpy(float)
        above = np.where(y >= 1.0)[0]
        if above.size == 0:
            out.append({**base, "crossing": np.nan, "crossing_lo": np.nan,
                        "crossing_hi": np.nan, "well_determined": False,
                        "rho_max": float(y.max()),
                        "note": "ρ 在扫描范围内始终 < 1，交叉点在上界之外"})
            continue
        i = int(above[0])
        if i == 0:
            out.append({**base, "crossing": np.nan, "crossing_lo": np.nan,
                        "crossing_hi": float(x[0]), "well_determined": False,
                        "rho_at_lo": float(y[0]),
                        "note": "ρ 在下界已 >= 1，交叉点在下界之外"})
            continue

        x_lo, x_hi = float(x[i - 1]), float(x[i])
        y_lo, y_hi = float(y[i - 1]), float(y[i])
        est = float(np.interp(1.0, [y_lo, y_hi], [x_lo, x_hi]))
        well = bool(usable["rho_hi"].iloc[i - 1] < 1.0
                    and usable["rho_lo"].iloc[i] > 1.0)
        out.append({
            **base, "crossing": est, "crossing_lo": x_lo, "crossing_hi": x_hi,
            "rho_at_lo": y_lo, "rho_at_hi": y_hi,
            "well_determined": well,
            "note": ("两端 CI 均不含 1，交叉点确定" if well
                     else "两端 CI 至少一端含 1，交叉区间偏宽"),
        })
    return pd.DataFrame(out)


def summary(res: Exp5Result) -> pd.DataFrame:
    """按 (通道, n, 模态) 汇总交叉点与 ρ 的单调性。"""
    c = res.crossing.copy()
    if c.empty:
        return c
    c["交叉点区间"] = c.apply(
        lambda r: (f"[{r['crossing_lo']:.2f}, {r['crossing_hi']:.2f}]"
                   if np.isfinite(r["crossing_lo"]) else "—"), axis=1)
    return c[["channel", "n", "mode", "crossing", "交叉点区间", "note"]]
