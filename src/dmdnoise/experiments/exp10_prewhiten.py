"""实验十 · 有色噪声下的秩判据：预白化的补救效果。

**背景（实验九）**：噪声一旦有色，**全部秩判据失效**（GD 由 1.000 掉到 0.000–0.280，
100% 秩高估）。机理是 GD 的阈值 `τ = ω(β)·median(s)` 为**固定比值**（约 1.57×median），
隐含假设噪声谱由白噪声 MP 律决定；有色噪声把"顶部/中位"由 1.25 抬到 2.2–9.3。

**补救思路**：**预白化**（见 `estimators/prewhiten.py`）。
信号是阻尼指数的和（移位算子的本征函数），故 `(I − ρS)` 把信号映为**同一组 `μ`、
只改幅度**——**子空间维数不变**；而 AR(1) 噪声恰好被白化。

**三种方案**：

    none       基线（不白化）
    oracle     用**真值** ρ/L 白化 —— 预白化能力的**上界**
    estimated  用**从数据估计**的系数白化 —— 可实现版本

⚠️ 估计器有一个已固化的陷阱：**不能用正交化后的奇异向量估计 ρ**
（正交归一化会破坏原始噪声的时间相关，实测 `ρ̂ ≈ 0`），
必须用**低秩残差矩阵本身**（见 `estimate_ar1_from_residual`）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from dmdnoise.config import Config, config_fingerprint
from dmdnoise.estimators.prewhiten import (
    estimate_ar1_from_residual as _est_ar1,
    estimate_spatial_from_residual as _est_spatial,
    whiten,
)
from dmdnoise.estimators.rank import VALID_METHODS, robust_sigma_rank
from dmdnoise.mc import derive_rng
from dmdnoise.sim import build, inject, rank_for, sigma_from_snr

LOG = logging.getLogger(__name__)

SCHEMES: tuple[str, ...] = ("none", "oracle", "estimated")
DEFAULT_COMBOS: tuple[tuple[str, int], ...] = (("complex", 8), ("real", 16))
#: (标签, 时间 ρ, 通道 ρ)
NOISE_GRID: tuple[tuple[str, float, float], ...] = (
    ("white", 0.0, 0.0),
    ("ar1-0.5", 0.5, 0.0),
    ("ar1-0.9", 0.9, 0.0),
    ("spatial-0.5", 0.0, 0.5),
    ("spatial-0.9", 0.0, 0.9),
)
SNR_DB = 10.0
M_DEFAULT = 200
J_DEFAULT = 1500
KAPPA = 3.0


@dataclass
class Exp10Result:
    table: pd.DataFrame
    summary: pd.DataFrame
    fingerprint: str
    meta: dict[str, Any] = field(default_factory=dict)


def run(cfg: Config, *, combos: Sequence[tuple[str, int]] = DEFAULT_COMBOS,
        noise_grid: Sequence[tuple[str, float, float]] = NOISE_GRID,
        snr_db: float = SNR_DB, m: int = M_DEFAULT, j_total: int = J_DEFAULT,
        progress: Any = None) -> Exp10Result:
    rows: list[dict[str, Any]] = []
    total = len(combos) * len(noise_grid)
    step = 0

    for channel, n in combos:
        sub = replace(cfg, oscillator=replace(cfg.oscillator, n=n))
        r_true = rank_for(sub, channel)

        for label, ar1, spatial in noise_grid:
            step += 1
            if progress is not None:
                progress(step, total, f"{channel} n={n} {label}")

            hist: dict[tuple[str, str], list[int]] = {
                (s, c): [] for s in SCHEMES for c in VALID_METHODS
            }
            rho_hat: list[tuple[float, float]] = []
            for j in range(j_total):
                rng_p = derive_rng(sub.run.seed, channel, snr_db, m, "e10", "phase", j)
                system = build(sub, channel, rng_p, m)
                sigma = sigma_from_snr(system.X, snr_db,
                                       convention=sub.noise.convention)
                rng_n = derive_rng(sub.run.seed, channel, snr_db, m, "e10", "noise", j)
                Xn, Yn, _ = inject(system.X, system.Y, sigma, rng_n,
                                   mode=sub.noise.mode, ar1=ar1, spatial=spatial)
                Z = np.vstack([Xn, Yn])

                for scheme in SCHEMES:
                    if scheme == "none":
                        W = Z
                    elif scheme == "oracle":
                        W = whiten(Z, ar1=ar1, spatial=spatial)
                    else:
                        rh_t = _est_ar1(Z)
                        rh_s = _est_spatial(Z) if spatial > 0.0 else 0.0
                        W = (whiten(Z, ar1=rh_t, spatial=rh_s)
                             if (rh_t > 0.0 or rh_s > 0.0) else Z)
                        if j == 0:
                            rho_hat.append((float(rh_t), float(rh_s)))
                    for crit in VALID_METHODS:
                        try:
                            _, rk = robust_sigma_rank(W, method=crit,
                                                      scheme="single")
                        except Exception:                    # noqa: BLE001
                            rk = -1
                        hist[(scheme, crit)].append(int(rk))

            for (scheme, crit), vals in hist.items():
                v = np.asarray(vals, dtype=int)
                rows.append({
                    "channel": channel, "n_axis": n, "noise": label,
                    "ar1_true": ar1, "spatial_true": spatial,
                    "scheme": scheme, "criterion": crit, "r_true": r_true,
                    "j_total": j_total,
                    "p_exact": float((v == r_true).mean()),
                    "p_over": float((v > r_true).mean()),
                    "p_under": float((v < r_true).mean()),
                    "p_invalid": float((v < 1).mean()),
                    "rho_hat_ar1": rho_hat[0][0] if rho_hat else np.nan,
                    "rho_hat_spatial": rho_hat[0][1] if rho_hat else np.nan,
                })

    table = pd.DataFrame(rows)
    return Exp10Result(
        table=table,
        summary=_summary(table),
        fingerprint=config_fingerprint(cfg),
        meta={"m": m, "j_total": j_total, "snr_db": snr_db, "kappa": KAPPA,
              "combos": [list(c) for c in combos],
              "noise_grid": [g[0] for g in noise_grid]},
    )


def _summary(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return table
    g = table.groupby(["channel", "noise", "scheme", "criterion"], dropna=False)
    out = g.agg(p_exact=("p_exact", "mean"), p_over=("p_over", "mean"),
                rho_hat_ar1=("rho_hat_ar1", "first"),
                rho_hat_spatial=("rho_hat_spatial", "first")).reset_index()
    pivot = out.pivot_table(index=["channel", "noise"],
                            columns=["criterion", "scheme"],
                            values="p_exact")
    pivot.columns = [f"{c}:{s}" for c, s in pivot.columns]
    return pivot.reset_index()


def verdict(res: Exp10Result, criterion: str = "gavish_donoho") -> pd.DataFrame:
    """逐 (通道, 噪声) 比较三方案的判对率，并给出恢复幅度。"""
    t = res.table[res.table["criterion"] == criterion]
    if t.empty:
        return t
    out: list[dict[str, Any]] = []
    for (ch, noise), g in t.groupby(["channel", "noise"]):
        d = {r["scheme"]: float(r["p_exact"]) for _, r in g.iterrows()}
        out.append({
            "channel": ch, "noise": noise,
            "ρ̂_ar1": g["rho_hat_ar1"].iloc[0],
            "ρ̂_spatial": g["rho_hat_spatial"].iloc[0],
            "none": d.get("none", np.nan),
            "oracle": d.get("oracle", np.nan),
            "estimated": d.get("estimated", np.nan),
            "恢复幅度": (d.get("estimated", np.nan) - d.get("none", np.nan)),
            "可达 oracle 的": (d.get("estimated", np.nan) / d.get("oracle", np.nan)
                            if d.get("oracle") else np.nan),
        })
    return pd.DataFrame(out)
