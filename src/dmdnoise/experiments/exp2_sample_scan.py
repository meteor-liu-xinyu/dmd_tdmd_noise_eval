"""实验二 · 样本量扫描。

规格依据：docs/experiment-protocol.md §3。

设计要点（§7.3）：
  - 嵌套前缀设计：每次实现只生成一条长序列，各 m 取其前缀
  - 由此引入跨 m 相关性，**禁止**对合并数据直接 OLS
  - RMSE 斜率：**逐实现拟合后跨实现统计**（t 区间）
  - 偏差斜率：跨实现自助（按实现整体重采样，保留嵌套结构）

工作点由实验一确定：`n = 8` 时可用 SNR 窗口为 ≥ 5 dB，故取 **SNR = 5 dB**
（原定的 10 dB 处偏差虽可分辨但量级偏小，−5 dB 处配对失败率 56% 不可用）。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from dmdnoise.config import Config, config_fingerprint
from dmdnoise.mc import CellResult, run_m_scan
from dmdnoise.metrics import summarize

LOG = logging.getLogger(__name__)

METHODS = ("dmd", "tdmd")

#: 工作点（由实验一实测确定）
SNR_DB = 5.0
M_VALUES: tuple[int, ...] = (50, 75, 100, 150, 200, 300, 400, 500)
J_DEFAULT = 10000
PAIR_FAIL_GATE = 0.20


@dataclass
class Exp2Result:
    table: pd.DataFrame          # 逐 (channel, m, method, mode) 的偏差-方差
    slopes: pd.DataFrame         # 斜率与置信区间
    fingerprint: str
    meta: dict[str, Any] = field(default_factory=dict)


def _error_matrix(cell: CellResult, method: str, mode: int) -> NDArray:
    """(J,) 相对误差，配对失败处为 NaN。"""
    est = cell.estimates[method][:, mode]
    return (est - cell.f_true[mode]) / cell.f_true[mode]


def _table_rows(cells: dict[int, CellResult], channel: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in sorted(cells):
        cell = cells[m]
        for method in METHODS:
            if method not in cell.estimates:
                continue
            for mode in range(cell.n_modes):
                err = _error_matrix(cell, method, mode)
                good = err[np.isfinite(err)]
                row: dict[str, Any] = {
                    "channel": channel,
                    "snr_db": cell.snr_db,
                    "n": cell.meta.get("n"),
                    "m": m,
                    "method": method,
                    "mode": mode + 1,
                    "j_total": cell.j_total,
                    "j_valid": int(good.size),
                    "pair_fail_rate": 1.0 - good.size / max(cell.j_total, 1),
                }
                if good.size >= 10:
                    row.update(summarize(good, bootstrap=1000).as_row())
                out.append(row)
    return out


def _per_realization_slopes(cells: dict[int, CellResult], method: str, mode: int,
                            ms: Sequence[int]) -> NDArray:
    """逐实现在 log-log 上拟合 |误差| 对 m 的斜率。

    仅保留所有 m 上均配对成功的实现（否则该实现的序列不完整）。
    """
    ms = sorted(ms)
    E = np.column_stack([_error_matrix(cells[m], method, mode) for m in ms])
    keep = np.all(np.isfinite(E), axis=1) & np.all(E != 0.0, axis=1)
    E = np.abs(E[keep])
    if E.shape[0] < 5:
        return np.empty(0)
    x = np.log(np.asarray(ms, dtype=float))
    x = x - x.mean()
    slopes = np.log(E) @ x / (x @ x)
    return np.asarray(slopes, dtype=float)


def _t_ci(samples: NDArray, alpha: float = 0.05) -> tuple[float, float, float]:
    """均值与 t 区间。"""
    n = samples.size
    if n < 2:
        return (float("nan"), float("nan"), float("nan"))
    mean = float(samples.mean())
    se = float(samples.std(ddof=1) / math.sqrt(n))
    # 正态近似（J 足够大，t 与正态差异可忽略）
    half = 1.96 * se
    return mean, mean - half, mean + half


def _bias_slope_ci(cells: dict[int, CellResult], method: str, mode: int,
                   ms: Sequence[int], *, bootstrap: int = 2000,
                   rng: np.random.Generator | None = None) -> dict[str, float]:
    """偏差对 m 的 log-log 斜率，CI 用**按实现整体重采样**的自助。

    保留嵌套结构：一次重采样同时抽中同一实现的全部 m。
    """
    ms = sorted(ms)
    E = np.column_stack([_error_matrix(cells[m], method, mode) for m in ms])
    n_j = E.shape[0]
    x = np.log(np.asarray(ms, dtype=float))
    xc = x - x.mean()
    denom = xc @ xc

    def slope_of(sample: NDArray) -> float:
        with np.errstate(invalid="ignore"):
            b = np.nanmean(sample, axis=0)
        good = np.isfinite(b) & (b != 0.0)
        if good.sum() < 3:
            return float("nan")
        return float((np.log(np.abs(b[good])) @ xc[good]) / (xc[good] @ xc[good]))

    point = slope_of(E)
    gen = rng if rng is not None else np.random.default_rng(0)
    pos = gen.integers(0, n_j, size=(bootstrap, n_j))
    boots = np.array([slope_of(E[p]) for p in pos])
    boots = boots[np.isfinite(boots)]
    if boots.size < 10:
        return {"slope": point, "ci_lo": float("nan"), "ci_hi": float("nan"),
                "n_boot": 0}
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"slope": point, "ci_lo": float(lo), "ci_hi": float(hi),
            "n_boot": int(boots.size)}


def run(cfg: Config, *, channels: Sequence[str] = ("complex", "real"),
        snr_db: float = SNR_DB, m_values: Sequence[int] = M_VALUES,
        j_total: int = J_DEFAULT, samples: dict[str, int] | None = None,
        snr_normalization: str = "m_max",
        progress: Any = None) -> Exp2Result:
    """运行实验二：样本量扫描。

    samples 为每通道的通道数（complex/real 分别取 8/16；实验一显示这两个点
    偏差可分辨且配对失败率低）。
    """
    samples = samples or {"complex": 8, "real": 16}
    rows: list[dict[str, Any]] = []
    slope_rows: list[dict[str, Any]] = []

    for channel in channels:
        from dataclasses import replace

        n = samples.get(channel, cfg.oscillator.n)
        sub = replace(cfg, oscillator=replace(cfg.oscillator, n=n))
        LOG.info("实验二 [%s] n=%d SNR=%g m=%s J=%d", channel, n, snr_db,
                 list(m_values), j_total)
        cells = run_m_scan(sub, channel, snr_db, m_values, j_total,
                           methods=METHODS, snr_normalization=snr_normalization,
                           progress=progress)
        rows.extend(_table_rows(cells, channel))

        for method in METHODS:
            for mode in range(cells[min(m_values)].n_modes):
                sl = _per_realization_slopes(cells, method, mode, m_values)
                mean, lo, hi = _t_ci(sl)
                bs = _bias_slope_ci(cells, method, mode, m_values)
                slope_rows.append({
                    "channel": channel,
                    "n": n,
                    "snr_normalization": snr_normalization,
                    "method": method,
                    "mode": mode + 1,
                    "n_realizations_used": int(sl.size),
                    "rmse_slope": mean,
                    "rmse_slope_lo": lo,
                    "rmse_slope_hi": hi,
                    "bias_slope": bs["slope"],
                    "bias_slope_lo": bs["ci_lo"],
                    "bias_slope_hi": bs["ci_hi"],
                    "theory_rmse_slope": -0.5,
                })

    return Exp2Result(
        table=pd.DataFrame(rows),
        slopes=pd.DataFrame(slope_rows),
        fingerprint=config_fingerprint(cfg),
        meta={"snr_db": snr_db, "m_values": list(m_values), "j_total": j_total,
              "samples": samples, "snr_normalization": snr_normalization},
    )


def judge(res: Exp2Result) -> pd.DataFrame:
    """按 H5–H8 判定。"""
    out: list[dict[str, Any]] = []
    for _, r in res.slopes.iterrows():
        covers_zero = (r["bias_slope_lo"] <= 0.0 <= r["bias_slope_hi"])
        covers_half = (r["rmse_slope_lo"] <= -0.5 <= r["rmse_slope_hi"])
        out.append({
            "channel": r["channel"], "method": r["method"], "mode": r["mode"],
            "bias_slope": r["bias_slope"],
            "bias_slope_ci": f"[{r['bias_slope_lo']:.3f}, {r['bias_slope_hi']:.3f}]",
            "H5_偏差不随m下降": "成立" if covers_zero else "不成立",
            "rmse_slope": r["rmse_slope"],
            "rmse_slope_ci": f"[{r['rmse_slope_lo']:.3f}, {r['rmse_slope_hi']:.3f}]",
            "H6_按m^-1/2下降": "成立" if covers_half else "不成立",
        })
    return pd.DataFrame(out)


def compare_normalizations(m_max: Exp2Result, per_m: Exp2Result) -> pd.DataFrame:
    """对比两种 SNR 归一化口径下的斜率。

    这是分离"m 效应"与"有效 SNR 效应"的关键对照：

      `m_max` 口径 —— 跨 m 共用同一个 sigma，大 m 的有效 SNR 更低（衰减所致）
      `per_m` 口径 —— 每个 m 的聚合 SNR 都等于目标值，只有 m 在变

    若某个斜率在两种口径下**同号同量级**，则该结论是 m 的固有性质；
    若只出现在 `m_max` 口径下、在 `per_m` 下消失或反号，则原结论是
    **测试床衰减造成的假象**。
    """
    if m_max.slopes.empty or per_m.slopes.empty:
        return pd.DataFrame()
    keys = ["channel", "method", "mode"]
    a = m_max.slopes.set_index(keys)
    b = per_m.slopes.set_index(keys)
    out: list[dict[str, Any]] = []
    for k in a.index.intersection(b.index):
        ra, rb = a.loc[k], b.loc[k]
        for stat in ("bias_slope", "rmse_slope"):
            va, vb = float(ra[stat]), float(rb[stat])
            sign_same = (va > 0) == (vb > 0)
            out.append({
                "channel": k[0], "method": k[1], "mode": k[2], "stat": stat,
                "m_max口径": va,
                "per_m口径": vb,
                "差值": vb - va,
                "同号": bool(sign_same),
                "判定": ("m 的固有性质" if sign_same and abs(vb) > 0.5 * abs(va)
                        else "疑似测试床衰减造成的假象" if not sign_same
                        else "口径敏感，需扩大 J 复核"),
            })
    return pd.DataFrame(out)
