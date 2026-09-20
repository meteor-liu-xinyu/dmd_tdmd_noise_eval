"""实验六 · 端到端评估：噪声水平未知 → 秩估计 → 频率估计 的完整链路。

**动机（本项目最大的认知缺口）**：
  - 实验一/二使用 **oracle 秩**（`r = r_true`），结论是**乐观上界**；
  - 实验三只比较了秩判据的"判对率"，**未把秩误差传导到最终的频率精度**。

两者之间缺一段：实际部署时 `σ` 未知 → 须估 `σ̂` → 定秩 `r̂(σ̂)` → 才能做 DMD/TDMD。
本实验把这段补上，并回答三个问题：

  1. **端到端误差比 oracle 秩差多少？**（用 MSE 比量化）
  2. **以最终频率精度（而非秩判对率）评判，哪个秩判据更好？**
  3. **秩误判的代价是否对称？** 高估（`r̂ > r_true`）与低估（`r̂ < r_true`）分别如何。

**设计要点**：
  - 同一实现内对**全部秩策略**共用同一条噪声与同一个 `Z`（公共随机数），
    使"策略间的差异"不被实现间波动掩盖；
  - `σ̂` 一律由**残差法自举**迭代得到（`adaptive_sigma`），不使用真值；
  - 秩越界（`r̂ > min(n, m)`）的实现记为失效，不计入偏差统计但单独报告。
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
from dmdnoise.estimators.rank import VALID_METHODS, RankError, adaptive_sigma
from dmdnoise.mc import derive_rng
from dmdnoise.metrics import paired_ratio, summarize
from dmdnoise.sim import build, inject, rank_for, sigma_from_snr

LOG = logging.getLogger(__name__)

#: 秩策略：oracle 作为上界参照，其余三个为数据驱动
STRATEGIES: tuple[str, ...] = ("oracle",) + VALID_METHODS
METHODS = ("dmd", "tdmd")
DEFAULT_COMBOS: tuple[tuple[str, int], ...] = (("complex", 8), ("real", 16))
DEFAULT_SNRS: tuple[float, ...] = (5.0, 10.0, 20.0)
M_DEFAULT = 200
J_DEFAULT = 2000
PAIR_FAIL_GATE = 0.20


@dataclass
class Exp6Result:
    table: pd.DataFrame           # 逐 (通道, SNR, 策略, 方法, 模态) 的端到端指标
    ranks: pd.DataFrame           # 逐 (通道, SNR, 策略) 的 r̂ 分布
    comparison: pd.DataFrame      # 数据驱动 vs oracle 的 MSE 比（配对 bootstrap CI）
    errors: pd.DataFrame          # 逐实现误差长表：供配对检验复用，避免重跑
    fingerprint: str
    meta: dict[str, Any] = field(default_factory=dict)


def _clamp_rank(r: int, n: int, m: int) -> int:
    return int(min(max(r, 1), min(n, m)))


def run(cfg: Config, *, combos: Sequence[tuple[str, int]] = DEFAULT_COMBOS,
        snrs: Sequence[float] = DEFAULT_SNRS, m: int = M_DEFAULT,
        j_total: int = J_DEFAULT, strategies: Sequence[str] = STRATEGIES,
        progress: Any = None) -> Exp6Result:
    rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    err_rows: list[dict[str, Any]] = []
    total = len(combos) * len(snrs)
    k = 0

    for channel, n in combos:
        sub = replace(cfg, oscillator=replace(cfg.oscillator, n=n))
        r_true = rank_for(sub, channel)
        dt = sub.oscillator.dt

        for snr in snrs:
            k += 1
            if progress is not None:
                progress(k, total, f"{channel} n={n} SNR={snr:g}dB")

            # 每个 (通道, SNR) 单元内，全部策略共用同一批实现
            est: dict[tuple[str, str], list[NDArray]] = {
                (s, meth): [] for s in strategies for meth in METHODS
            }
            ok: dict[tuple[str, str], list[bool]] = {
                (s, meth): [] for s in strategies for meth in METHODS
            }
            rank_hist: dict[str, list[int]] = {s: [] for s in strategies}
            sigma_hist: dict[str, list[float]] = {s: [] for s in strategies}
            f_true: NDArray | None = None
            sigma_true = float("nan")

            for j in range(j_total):
                rng_phase = derive_rng(sub.run.seed, channel, snr, m, "e2e", "phase", j)
                system = build(sub, channel, rng_phase, m)
                if f_true is None:
                    f_true = system.f_true
                    sigma_true = sigma_from_snr(system.X, snr,
                                                convention=sub.noise.convention)

                rng_noise = derive_rng(sub.run.seed, channel, snr, m, "e2e", "noise", j)
                Xn, Yn, _ = inject(system.X, system.Y, sigma_true, rng_noise,
                                   mode=sub.noise.mode)
                Z = np.vstack([Xn, Yn])
                sv = np.linalg.svd(Z, compute_uv=False)

                for strat in strategies:
                    if strat == "oracle":
                        r_hat, sigma_hat = r_true, sigma_true
                    else:
                        try:
                            sigma_hat, r_hat = adaptive_sigma(Z, method=strat, sv=sv)
                        except RankError:
                            r_hat, sigma_hat = 0, float("nan")
                    rank_hist[strat].append(int(r_hat))
                    sigma_hist[strat].append(float(sigma_hat))

                    r_use = _clamp_rank(int(r_hat), n, m)
                    for meth in METHODS:
                        try:
                            res = REGISTRY[meth]().fit(Xn, Yn, r_use, dt)
                        except Exception:                       # noqa: BLE001
                            est[(strat, meth)].append(np.full(int(system.f_true.size), np.nan))
                            ok[(strat, meth)].append(False)
                            continue
                        pr = pair_to_truth(res.freqs, system.f_true)
                        if pr.ok:
                            est[(strat, meth)].append(pr.freqs)
                            ok[(strat, meth)].append(True)
                        else:
                            est[(strat, meth)].append(np.full(int(system.f_true.size), np.nan))
                            ok[(strat, meth)].append(False)

            assert f_true is not None
            n_modes = int(f_true.size)

            # 保存逐实现误差（公共随机数下各策略逐实现对应，可直接做配对检验）
            for strat in strategies:
                for meth in METHODS:
                    arr = np.asarray(est[(strat, meth)], dtype=float)
                    okv = np.asarray(ok[(strat, meth)], dtype=bool)
                    for jj in range(j_total):
                        if not okv[jj]:
                            continue
                        rec: dict[str, Any] = {
                            "channel": channel, "n_axis": n, "snr_db": snr,
                            "strategy": strat, "method": meth, "j": jj,
                        }
                        for mode in range(n_modes):
                            rec[f"err{mode + 1}"] = (
                                arr[jj, mode] - f_true[mode]) / f_true[mode]
                        err_rows.append(rec)

            for strat in strategies:
                rh = np.asarray(rank_hist[strat], dtype=int)
                sh = np.asarray(sigma_hist[strat], dtype=float)
                rank_rows.append({
                    "channel": channel, "n_axis": n, "snr_db": snr, "strategy": strat,
                    "r_true": r_true, "j_total": j_total,
                    "r_hat_mean": float(rh.mean()), "r_hat_std": float(rh.std(ddof=1)),
                    "p_exact": float((rh == r_true).mean()),
                    "p_over": float((rh > r_true).mean()),
                    "p_under": float((rh < r_true).mean()),
                    "p_invalid": float((rh < 1).mean()),
                    "sigma_hat_over_true": float(np.nanmean(sh) / sigma_true)
                    if sigma_true else float("nan"),
                })

            for strat in strategies:
                for meth in METHODS:
                    key = (strat, meth)
                    arr = np.asarray(est[key], dtype=float)
                    okv = np.asarray(ok[key], dtype=bool)
                    for mode in range(n_modes):
                        err = (arr[okv, mode] - f_true[mode]) / f_true[mode] \
                            if okv.any() else np.empty(0)
                        row: dict[str, Any] = {
                            "channel": channel, "n_axis": n, "snr_db": snr,
                            "strategy": strat, "method": meth, "mode": mode + 1,
                            "r_true": r_true, "j_total": j_total,
                            "j_valid": int(err.size),
                            "pair_fail_rate": 1.0 - err.size / max(j_total, 1),
                        }
                        if err.size >= 10:
                            row.update(summarize(err, bootstrap=1000).as_row())
                        rows.append(row)

    table = pd.DataFrame(rows)
    errors = pd.DataFrame(err_rows)
    return Exp6Result(
        table=table,
        ranks=pd.DataFrame(rank_rows),
        comparison=_compare(errors),
        errors=errors,
        fingerprint=config_fingerprint(cfg),
        meta={"m": m, "j_total": j_total, "snrs": list(snrs),
              "combos": [list(c) for c in combos],
              "strategies": list(strategies)},
    )


def _compare(errors: pd.DataFrame) -> pd.DataFrame:
    """端到端退化：数据驱动秩 vs oracle 秩的 MSE 比，**配对 bootstrap** 给 CI。

    公共随机数下各策略逐实现对应（同一 `j`），故差值方差远小于各自方差——
    这正是能否判定"退化显著"的关键。
    """
    if errors.empty:
        return errors
    out: list[dict[str, Any]] = []
    keys = ["channel", "n_axis", "snr_db"]
    for k, g in errors.groupby(keys):
        modes = [c for c in g.columns if c.startswith("err")]
        for strat in [s for s in g["strategy"].unique() if s != "oracle"]:
            for meth in METHODS:
                base = g[(g["strategy"] == "oracle") & (g["method"] == meth)]
                other = g[(g["strategy"] == strat) & (g["method"] == meth)]
                merged = base.merge(other, on="j", suffixes=("_o", "_s"))
                if len(merged) < 20:
                    continue
                for mode in modes:
                    a = merged[f"{mode}_o"].to_numpy(float)
                    bb = merged[f"{mode}_s"].to_numpy(float)
                    pr = paired_ratio(bb, a, stat="mse", bootstrap=2000)
                    out.append({
                        "channel": k[0], "n_axis": k[1], "snr_db": k[2],
                        "mode": int(mode[-1]), "strategy": strat, "method": meth,
                        "n_pair": pr.n, "mse_oracle": float(np.mean(a ** 2)),
                        "mse_data_driven": float(np.mean(bb ** 2)),
                        "mse_ratio": pr.point, "mse_lo": pr.ci[0], "mse_hi": pr.ci[1],
                        "退化显著": bool(pr.ci[0] > 1.0),
                    })
    return pd.DataFrame(out)


def _safe_max(series) -> float:
    """全 NaN 时返回 NaN，不触发 RuntimeWarning。"""
    v = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    return float(v.max()) if v.size else float("nan")


def verdict(res: Exp6Result) -> pd.DataFrame:
    """按 (通道, SNR) 汇总裁决：端到端可用性与最优判据。"""
    if res.table.empty:
        return res.table
    # ranks 表用 `n` 表示通道数，table 表用 `n_axis`；此处统一为 `n_axis`，
    # 避免与 Summary.as_row() 的 `n`（样本数）语义混淆。
    ranks = res.ranks.rename(columns={"n": "n_axis"})
    out: list[dict[str, Any]] = []
    for (ch, n, snr), g in res.table.groupby(["channel", "n_axis", "snr_db"]):
        for strat in g["strategy"].unique():
            gg = g[g["strategy"] == strat]
            fail = _safe_max(gg["pair_fail_rate"])
            res_any = bool(gg["resolvable"].eq(True).any()) if "resolvable" in gg else False
            td = gg[gg["method"] == "tdmd"]
            dm = gg[gg["method"] == "dmd"]
            mse_t = _safe_max(td["mse_decomp"]) if "mse_decomp" in td else np.nan
            mse_d = _safe_max(dm["mse_decomp"]) if "mse_decomp" in dm else np.nan
            rk = ranks[(ranks["channel"] == ch) & (ranks["n_axis"] == n)
                       & (ranks["snr_db"] == snr)
                       & (ranks["strategy"] == strat)]
            out.append({
                "channel": ch, "n_axis": n, "snr_db": snr, "strategy": strat,
                "秩判对率": float(rk["p_exact"].iloc[0]) if not rk.empty else np.nan,
                "配对失败率": fail,
                "可分辨": res_any,
                "TDMD_vs_DMD_MSE": (mse_t / mse_d) if (mse_d and np.isfinite(mse_d)) else np.nan,
                "端到端可用": bool(fail <= PAIR_FAIL_GATE and res_any),
            })
    return pd.DataFrame(out).sort_values(["channel", "snr_db", "strategy"])
