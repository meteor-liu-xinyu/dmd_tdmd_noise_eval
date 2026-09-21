"""实验九 · 有色噪声下的稳健性。

**动机**：本项目此前**全部结论都建立在加性高斯白噪声上**（`sim/noise.py` 只有白噪声）。
但真实测量噪声普遍是**有色**的——传感器带宽有限（时间相关）、阵列串扰（通道相关）。

本实验检验两件事是否在白性假设被破坏后依然成立：

  1. **TDMD 的降偏收益**——理论基于 EIV，不要求噪声白性；但 TDMD 的投影用增广矩阵
     `Z = [X; Y]` 的右奇异向量，时间相关会改变该子空间的结构；
  2. **秩判据的可用性**——`gavish_donoho` 与 `marchenko_pastur` **都基于白噪声的谱假设**，
     有色噪声会使噪声奇异值偏离 MP 分布，可能直接失效。

**噪声模型**（均保持**逐元素单位方差**，故 `sigma` 标定不变）：

    white          白噪声（基线）
    ar1(0.5/0.9)   时间方向 AR(1)：x_t = rho x_{t-1} + sqrt(1-rho^2) w_t
    spatial(0.5/0.9) 通道方向指数相关：R_ij = rho^{|i-j|}
    ar1+spatial    两者叠加
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from dmdnoise.config import Config, config_fingerprint
from dmdnoise.estimators import REGISTRY, pair_to_truth
from dmdnoise.estimators.rank import VALID_METHODS, robust_sigma_rank
from dmdnoise.mc import derive_rng
from dmdnoise.metrics import paired_ratio, summarize
from dmdnoise.sim import build, inject, rank_for, sigma_from_snr

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class NoiseSpec:
    """一种噪声配置。"""

    label: str
    ar1: float = 0.0
    spatial: float = 0.0

    def describe(self) -> str:
        if self.ar1 == 0.0 and self.spatial == 0.0:
            return "white"
        parts = []
        if self.ar1:
            parts.append(f"ar1={self.ar1:g}")
        if self.spatial:
            parts.append(f"spatial={self.spatial:g}")
        return "+".join(parts)


#: 默认噪声网格：基线 + 单向相关 + 双向叠加
NOISE_GRID: tuple[NoiseSpec, ...] = (
    NoiseSpec("white"),
    NoiseSpec("ar1-0.5", ar1=0.5),
    NoiseSpec("ar1-0.9", ar1=0.9),
    NoiseSpec("spatial-0.5", spatial=0.5),
    NoiseSpec("spatial-0.9", spatial=0.9),
    NoiseSpec("both-0.5", ar1=0.5, spatial=0.5),
)

DEFAULT_COMBOS: tuple[tuple[str, int], ...] = (("complex", 8), ("real", 16))
SNR_DB = 10.0
M_DEFAULT = 200
J_DEFAULT = 2000
METHODS = ("dmd", "tdmd")
PAIR_FAIL_GATE = 0.20


@dataclass
class Exp9Result:
    table: pd.DataFrame        # 逐 (通道, 噪声, 方法, 模态) 的偏差-方差
    paired: pd.DataFrame       # 配对比较（降偏比 / std 比）
    ranks: pd.DataFrame        # 秩判据在有色噪声下的判对率
    fingerprint: str
    meta: dict[str, Any] = field(default_factory=dict)


def run(cfg: Config, *, combos: Sequence[tuple[str, int]] = DEFAULT_COMBOS,
        noise_grid: Sequence[NoiseSpec] = NOISE_GRID, snr_db: float = SNR_DB,
        m: int = M_DEFAULT, j_total: int = J_DEFAULT,
        progress: Any = None) -> Exp9Result:
    rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    total = len(combos) * len(noise_grid)
    step = 0

    for channel, n in combos:
        sub = replace(cfg, oscillator=replace(cfg.oscillator, n=n))
        r_true = rank_for(sub, channel)
        dt = sub.oscillator.dt

        for spec in noise_grid:
            step += 1
            if progress is not None:
                progress(step, total, f"{channel} n={n} {spec.describe()}")

            est = {k: [] for k in METHODS}
            ok = {k: [] for k in METHODS}
            rank_hist: dict[str, list[int]] = {k: [] for k in VALID_METHODS}
            f_true: NDArray | None = None
            sigma_true = float("nan")
            # 实测噪声相关（校验注入确实生效）
            ac_meas, sc_meas = [], []

            for j in range(j_total):
                rng_p = derive_rng(sub.run.seed, channel, snr_db, m, "e9", "phase", j)
                system = build(sub, channel, rng_p, m)
                if f_true is None:
                    f_true = system.f_true
                    sigma_true = sigma_from_snr(system.X, snr_db,
                                                convention=sub.noise.convention)

                rng_n = derive_rng(sub.run.seed, channel, snr_db, m, "e9", "noise", j)
                Xn, Yn, meta = inject(system.X, system.Y, sigma_true, rng_n,
                                      mode=sub.noise.mode,
                                      ar1=spec.ar1, spatial=spec.spatial)
                if j == 0:
                    d = Xn - system.X
                    if d.shape[1] > 2 and d.shape[0] > 2:
                        ac_meas.append(float(np.mean(
                            [np.corrcoef(d[i, :-1], d[i, 1:])[0, 1]
                             for i in range(d.shape[0])])))
                        sc_meas.append(float(np.corrcoef(
                            d[:-1].ravel(), d[1:].ravel())[0, 1]))

                for name in METHODS:
                    try:
                        r = REGISTRY[name]().fit(Xn, Yn, r_true, dt)
                    except Exception:                          # noqa: BLE001
                        est[name].append(np.full(int(system.f_true.size), np.nan))
                        ok[name].append(False)
                        continue
                    pr = pair_to_truth(r.freqs, system.f_true)
                    est[name].append(pr.freqs if pr.ok
                                     else np.full(int(system.f_true.size), np.nan))
                    ok[name].append(bool(pr.ok))

                # 秩判据（用校正后的 σ̂，`single` 方案）
                Z = np.vstack([Xn, Yn])
                sv = np.linalg.svd(Z, compute_uv=False)
                for crit in VALID_METHODS:
                    try:
                        _, rh = robust_sigma_rank(Z, method=crit, scheme="single",
                                                  sv=sv)
                    except Exception:                          # noqa: BLE001
                        rh = -1
                    rank_hist[crit].append(int(rh))

            assert f_true is not None
            n_modes = int(f_true.size)

            for name in METHODS:
                arr = np.asarray(est[name], dtype=float)
                okv = np.asarray(ok[name], dtype=bool)
                for mode in range(n_modes):
                    e = (arr[okv, mode] - f_true[mode]) / f_true[mode] if okv.any() \
                        else np.empty(0)
                    row: dict[str, Any] = {
                        "channel": channel, "n_axis": n, "noise": spec.label,
                        "ar1": spec.ar1, "spatial": spec.spatial, "snr_db": snr_db,
                        "method": name, "mode": mode + 1, "r_true": r_true,
                        "j_total": j_total, "j_valid": int(e.size),
                        "pair_fail_rate": 1.0 - e.size / max(j_total, 1),
                        "ac_meas": ac_meas[0] if ac_meas else np.nan,
                        "sc_meas": sc_meas[0] if sc_meas else np.nan,
                    }
                    if e.size >= 10:
                        row.update(summarize(e, bootstrap=1000).as_row())
                    rows.append(row)

            # 配对比较（公共随机数下两法逐实现对应）
            for mode in range(n_modes):
                d = np.asarray([v[mode] for v, o in zip(est["dmd"], ok["dmd"]) if o])
                t = np.asarray([v[mode] for v, o in zip(est["tdmd"], ok["tdmd"]) if o])
                kk = min(d.size, t.size)
                if kk < 10:
                    continue
                pr = paired_ratio(t[:kk], d[:kk], stat="std", bootstrap=2000)
                pair_rows.append({
                    "channel": channel, "n_axis": n, "noise": spec.label,
                    "mode": mode + 1, "std_ratio": pr.point, "std_lo": pr.ci[0],
                    "std_hi": pr.ci[1], "n_pair": pr.n,
                    "dmd_bias": float(np.mean(d[:kk])),
                    "tdmd_bias": float(np.mean(t[:kk])),
                    "bias_ratio": (abs(float(np.mean(t[:kk])))
                                   / abs(float(np.mean(d[:kk])))
                                   if np.mean(d[:kk]) else np.nan),
                })

            for crit in VALID_METHODS:
                rh = np.asarray(rank_hist[crit], dtype=int)
                rank_rows.append({
                    "channel": channel, "n_axis": n, "noise": spec.label,
                    "criterion": crit, "r_true": r_true, "j_total": j_total,
                    "p_exact": float((rh == r_true).mean()),
                    "p_over": float((rh > r_true).mean()),
                    "p_under": float((rh < r_true).mean()),
                    "p_invalid": float((rh < 1).mean()),
                })

    return Exp9Result(
        table=pd.DataFrame(rows),
        paired=pd.DataFrame(pair_rows),
        ranks=pd.DataFrame(rank_rows),
        fingerprint=config_fingerprint(cfg),
        meta={"m": m, "j_total": j_total, "snr_db": snr_db,
              "noise_grid": [s.label for s in noise_grid],
              "combos": [list(c) for c in combos]},
    )


def verdict(res: Exp9Result) -> pd.DataFrame:
    """汇总：有色噪声下 TDMD 收益与秩判据是否失效。"""
    if res.table.empty:
        return res.table
    out: list[dict[str, Any]] = []
    for (ch, noise), g in res.table.groupby(["channel", "noise"]):
        d = g[g["method"] == "dmd"]
        t = g[g["method"] == "tdmd"]
        bias_d = _safe_max(d["bias"].abs()) if "bias" in d else np.nan
        bias_t = _safe_max(t["bias"].abs()) if "bias" in t else np.nan
        fail = _safe_max(g["pair_fail_rate"])
        pr = res.paired[(res.paired["channel"] == ch) & (res.paired["noise"] == noise)]
        rk = res.ranks[(res.ranks["channel"] == ch) & (res.ranks["noise"] == noise)]
        out.append({
            "channel": ch, "noise": noise,
            "DMD偏差": bias_d, "TDMD偏差": bias_t,
            "降偏比": (bias_t / bias_d) if bias_d else np.nan,
            "std比中位": float(np.median(pr["std_ratio"])) if not pr.empty else np.nan,
            "配对失败率": fail,
            "GD判对率": _pick(rk, "gavish_donoho"),
            "MP判对率": _pick(rk, "marchenko_pastur"),
            "可用": bool(np.isfinite(bias_t) and fail <= PAIR_FAIL_GATE),
        })
    return pd.DataFrame(out).sort_values(["channel", "noise"])


def _safe_max(series) -> float:
    v = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    return float(v.max()) if v.size else float("nan")


def _pick(rk: pd.DataFrame, crit: str) -> float:
    g = rk[rk["criterion"] == crit]
    return float(g["p_exact"].mean()) if not g.empty else float("nan")
