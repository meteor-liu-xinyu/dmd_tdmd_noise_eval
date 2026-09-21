"""实验八 · 交叉点随 `n` 的"非单调"是真实效应还是观测矩阵的实现噪声？

**背景（实验五的遗留反常）**：实值通道的交叉点随 `n` 的变化在两个模态上方向相反——

    n=8  : 模态 1、2 均在扫描上界（>2.0）之外
    n=16 : 模态 1 = 1.86，模态 2 = 1.66
    n=32 : 模态 1 = 2.00，模态 2 < 1.50

于是"交叉点 = f(n)"这一形式不成立。但**实值通道的观测矩阵 `C ∈ R^{n×4}` 是逐实现随机抽取的**，
每个 `n` 在实验五中只用了一个 `C`。因此存在一个更基本的可能：

> **所谓"随 `n` 非单调"，可能只是 `C` 的随机实现差异，而不是 `n` 的效应。**

**本实验判定这一点。** 做法：固定 `n`，抽取 `K` 个不同的 `C` 实现，
逐一测出交叉点，得到**交叉点的分布**；再比较不同 `n` 的分布是否分离。

- 若两个 `n` 的分布**大幅重叠** → "`n` 依赖"是假象，真实结论是
  **"交叉点 ≈ 某个区间，与 `n` 无关，但随 `C` 的实现而散布"**；
- 若分布**系统性分离** → `n` 依赖是真实的，需进一步找机制。

**控制技巧**：`build_real` 先抽 `C`（`n×4` 个正态数）再抽相位，抽取个数与频率无关。
故**用同一个随机种子、只改频率**，可保证 `C` 与相位逐位相同，只让模态间隔变化——
这正是交叉点定义所需要的对照。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from dmdnoise.config import Config, config_fingerprint, validate
from dmdnoise.estimators import REGISTRY, pair_to_truth
from dmdnoise.mc import derive_rng
from dmdnoise.metrics import paired_ratio
from dmdnoise.sim import build, inject, rank_for, sigma_from_snr

LOG = logging.getLogger(__name__)

N_VALUES: tuple[int, ...] = (16, 32)
#: 间隔网格：覆盖实验五观察到的交叉区间（1.66–2.00）并向外留余量。
#: 需要用 3 个点，因为交叉点在实现之间散布很广，两点常夹不住。
SEPARATIONS: tuple[float, ...] = (1.55, 1.75, 1.95)
K_CONFIGS = 15
M_DEFAULT = 200
J_DEFAULT = 800
SNR_DB = 10.0
WINDOW_DECAY = 0.70
F1_HZ = 12.0
PAIR_FAIL_GATE = 0.20


@dataclass
class Exp8Result:
    table: pd.DataFrame        # 逐 (n, 配置k, 模态, 间隔) 的 rho 与偏差
    crossings: pd.DataFrame    # 逐 (n, 配置k, 模态) 的交叉点估计
    spread: pd.DataFrame       # 逐 (n, 模态) 的交叉点分布统计与 n 间对比
    fingerprint: str
    meta: dict[str, Any] = field(default_factory=dict)


def _zeta_from_decay(f_hz: float, decay: float, m: int, dt: float) -> float:
    import math
    return -math.log(decay) / (2.0 * math.pi * f_hz * m * dt)


def _config_for(cfg: Config, n: int, sep: float, m: int) -> Config:
    from dmdnoise.config import Mode

    f2 = F1_HZ * sep
    z1 = _zeta_from_decay(F1_HZ, WINDOW_DECAY, m, cfg.oscillator.dt)
    z2 = _zeta_from_decay(f2, WINDOW_DECAY, m, cfg.oscillator.dt)
    osc = replace(cfg.oscillator, modes=(Mode(F1_HZ, z1), Mode(f2, z2)), n=n)
    grid = replace(cfg.grid, m_min=m, m_max=m, m_scan=(m,))
    return replace(cfg, oscillator=osc, grid=grid)


def run(cfg: Config, *, n_values: Sequence[int] = N_VALUES,
        k_configs: int = K_CONFIGS, separations: Sequence[float] = SEPARATIONS,
        m: int = M_DEFAULT, j_total: int = J_DEFAULT,
        progress: Any = None) -> Exp8Result:
    rows: list[dict[str, Any]] = []
    total = len(n_values) * k_configs
    step = 0

    for n in n_values:
        for k in range(k_configs):
            step += 1
            if progress is not None:
                progress(step, total, f"n={n} C#{k}")
            dt = cfg.oscillator.dt
            cfg_any = _config_for(cfg, n, separations[0], m)
            r_true = rank_for(cfg_any, "real")

            for sep in separations:
                sub = _config_for(cfg, n, sep, m)
                # 配置合法性：扫描网格必须落在设计约束内
                res = validate(sub, strict=False)
                failed = [r.name for r in res if not r.passed]
                if failed:
                    raise ValueError(f"n={n} sep={sep} 违反硬约束：{failed}")

                # 关键控制：用【同一个】种子构造系统 -> C 与相位逐位相同，只改频率
                rng_sys = np.random.default_rng(k)
                system = build(sub, "real", rng_sys, m)
                sigma = sigma_from_snr(system.X, SNR_DB,
                                       convention=cfg.noise.convention)

                est = {"dmd": [], "tdmd": []}
                ok = {"dmd": [], "tdmd": []}
                for j in range(j_total):
                    rng_n = derive_rng(cfg.run.seed, "real", n, sep, "exp8", "noise",
                                       k, j)
                    Xn, Yn, _ = inject(system.X, system.Y, sigma, rng_n,
                                       mode=cfg.noise.mode)
                    for name in ("dmd", "tdmd"):
                        try:
                            r = REGISTRY[name]().fit(Xn, Yn, r_true, dt)
                        except Exception:                       # noqa: BLE001
                            est[name].append(np.full(2, np.nan))
                            ok[name].append(False)
                            continue
                        pr = pair_to_truth(r.freqs, system.f_true)
                        est[name].append(pr.freqs if pr.ok else np.full(2, np.nan))
                        ok[name].append(bool(pr.ok))

                for mode in range(2):
                    d = np.asarray([e[mode] for e, o in zip(est["dmd"], ok["dmd"]) if o])
                    t = np.asarray([e[mode] for e, o in zip(est["tdmd"], ok["tdmd"]) if o])
                    common = min(d.size, t.size)
                    row: dict[str, Any] = {
                        "n_axis": n, "config": k, "mode": mode + 1,
                        "f2_over_f1": sep, "r_true": r_true, "j_total": j_total,
                        "j_valid_dmd": int(d.size), "j_valid_tdmd": int(t.size),
                        "pair_fail_dmd": 1.0 - d.size / j_total,
                        "pair_fail_tdmd": 1.0 - t.size / j_total,
                    }
                    if common >= 10:
                        dd = (d[:common] - system.f_true[mode]) / system.f_true[mode]
                        tt = (t[:common] - system.f_true[mode]) / system.f_true[mode]
                        bd, bt = float(np.mean(dd)), float(np.mean(tt))
                        row.update({
                            "bias_dmd": bd, "bias_tdmd": bt,
                            "rho": abs(bt) / abs(bd) if bd else np.nan,
                        })
                        pr = paired_ratio(tt, dd, stat="mse", bootstrap=1500)
                        row.update({"mse_ratio": pr.point, "mse_lo": pr.ci[0],
                                    "mse_hi": pr.ci[1],
                                    "退化显著": bool(pr.ci[0] > 1.0)})
                    rows.append(row)

    table = pd.DataFrame(rows)
    crossings = _crossings(table)
    return Exp8Result(
        table=table,
        crossings=crossings,
        spread=_spread(crossings),
        fingerprint=config_fingerprint(cfg),
        meta={"m": m, "j_total": j_total, "k_configs": k_configs,
              "n_values": list(n_values), "separations": list(separations),
              "snr_db": SNR_DB, "window_decay": WINDOW_DECAY},
    )


def _crossings(table: pd.DataFrame) -> pd.DataFrame:
    """逐 (n, C 配置, 模态) 定位 ρ 穿越 1 处；两侧均在扫描范围内才给估计。"""
    out: list[dict[str, Any]] = []
    for (n, k, mode), g in table.groupby(["n_axis", "config", "mode"]):
        g = g.sort_values("f2_over_f1")
        base = {"n_axis": n, "config": k, "mode": mode}
        usable = g[g["rho"].notna() & (g["pair_fail_dmd"] <= PAIR_FAIL_GATE)
                   & (g["pair_fail_tdmd"] <= PAIR_FAIL_GATE)]
        if len(usable) < 2:
            out.append({**base, "crossing": np.nan,
                        "note": "配对失效率过高，不可报告"})
            continue
        x = usable["f2_over_f1"].to_numpy(float)
        y = usable["rho"].to_numpy(float)
        if y.min() > 1.0:
            out.append({**base, "crossing": np.nan, "rho_min": float(y.min()),
                        "note": "ρ 全 > 1，交叉点在下界之下"})
        elif y.max() < 1.0:
            out.append({**base, "crossing": np.nan, "rho_max": float(y.max()),
                        "note": "ρ 全 < 1，交叉点在上界之上"})
        else:
            est = float(np.interp(1.0, y, x))
            out.append({**base, "crossing": est,
                        "crossing_lo": float(x.min()), "crossing_hi": float(x.max()),
                        "note": "线性内插"})
    return pd.DataFrame(out)


def _spread(crossings: pd.DataFrame) -> pd.DataFrame:
    """逐 (n, 模态) 的交叉点分布统计，并给出 `n` 之间的对比。"""
    out: list[dict[str, Any]] = []
    for (n, mode), g in crossings.groupby(["n_axis", "mode"]):
        c = g["crossing"].dropna()
        out.append({
            "n_axis": n, "mode": mode, "n_configs": int(len(g)),
            "n_crossing_inside": int(c.size),
            "median": float(c.median()) if c.size else np.nan,
            "q25": float(c.quantile(0.25)) if c.size else np.nan,
            "q75": float(c.quantile(0.75)) if c.size else np.nan,
            "min": float(c.min()) if c.size else np.nan,
            "max": float(c.max()) if c.size else np.nan,
            "below_lo": int(g["note"].str.contains("下界之下").sum()),
            "above_hi": int(g["note"].str.contains("上界之上").sum()),
        })
    return pd.DataFrame(out).sort_values(["mode", "n_axis"])


def verdict(res: Exp8Result) -> pd.DataFrame:
    """逐模态比较两个 `n` 的交叉点分布，判定"随 n 非单调"是否真实。"""
    sp = res.spread
    if sp.empty:
        return sp
    out: list[dict[str, Any]] = []
    for mode in sorted(sp["mode"].unique()):
        g = sp[sp["mode"] == mode].sort_values("n_axis")
        row: dict[str, Any] = {"mode": mode}
        for _, r in g.iterrows():
            row[f"n={int(r['n_axis'])}_中位"] = r["median"]
            row[f"n={int(r['n_axis'])}_IQR"] = (
                f"[{r['q25']:.2f}, {r['q75']:.2f}]"
                if np.isfinite(r["q25"]) else "—")
            row[f"n={int(r['n_axis'])}_区间外"] = (
                f"低{r['below_lo']}/高{r['above_hi']}")
        # 判据：两 n 的四分位区间是否重叠
        if len(g) >= 2 and g["q25"].notna().all():
            ov_lo = float(max(g["q25"].iloc[0], g["q25"].iloc[1]))
            ov_hi = float(min(g["q75"].iloc[0], g["q75"].iloc[1]))
            row["判定"] = ("分布重叠 → n 依赖不可分辨（实现噪声主导）"
                        if ov_lo <= ov_hi else "分布分离 → n 依赖真实存在")
        else:
            row["判定"] = "四分位区间不足（多数实现未落在扫描范围内）"
        out.append(row)
    return pd.DataFrame(out)
