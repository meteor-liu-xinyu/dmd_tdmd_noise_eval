"""实验三 · 噪声估计敏感性（方案 3A：秩估计对 σ̂ 的敏感性）。

规格依据：docs/experiment-protocol.md §4.2；决策见 ADR-002。

问题背景：原版 TDMD **没有**噪声水平输入参数，因此"噪声估计不准时还能不能用"
在原算法上没有挂钩。σ̂ 在 TDMD 流程中**唯一自然且必经**的入口是**秩选择 `r̂(σ̂)`**。

本实验的核心对照（由 `estimators/rank.py` 的规格决定）：

    判据               是否依赖 σ̂
    energy             依赖（阈值含 (σ̂/s_1)²）
    marchenko_pastur   依赖（噪声地板上沿 ∝ σ̂）
    gavish_donoho      **不依赖**（阈值由奇异值中位数自标定）

预期结论：只有 σ̂ 依赖型判据会出现鲁棒边界；GD 判据天然免疫。
这把问题从"σ̂ 估不准会怎样"细化为"**哪些秩判据会受影响**"。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from dmdnoise.config import Config, config_fingerprint
from dmdnoise.estimators import DMD, TDMD, pair_to_truth
from dmdnoise.estimators.rank import (
    VALID_METHODS,
    RankError,
    adaptive_sigma,
    estimate_rank,
)
from dmdnoise.mc import derive_rng
from dmdnoise.metrics import summarize
from dmdnoise.sim import build, inject, rank_for, sigma_from_snr

LOG = logging.getLogger(__name__)

K_GRID: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
SNR_DB = 5.0
M_DEFAULT = 200
J_DEFAULT = 2000
PAIR_FAIL_GATE = 0.20


@dataclass
class Exp3Result:
    table: pd.DataFrame           # 逐 (channel, method, k, estimator, mode) 的偏差-方差
    rank_table: pd.DataFrame      # 逐 (channel, estimator, k) 的秩误判率
    robust: pd.DataFrame          # 鲁棒区间
    adaptive: pd.DataFrame        # 数据驱动 σ̂（自举）的表现
    fingerprint: str
    meta: dict[str, Any] = field(default_factory=dict)


def run(cfg: Config, *, channels: Sequence[str] = ("complex", "real"),
        samples: dict[str, int] | None = None, snr_db: float = SNR_DB,
        m: int = M_DEFAULT, j_total: int = J_DEFAULT,
        k_grid: Sequence[float] = K_GRID,
        estimators: Sequence[str] = VALID_METHODS,
        progress: Any = None) -> Exp3Result:
    samples = samples or {"complex": 8, "real": 16}
    rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    adapt_rows: list[dict[str, Any]] = []

    for channel in channels:
        n = samples.get(channel, cfg.oscillator.n)
        sub = replace(cfg, oscillator=replace(cfg.oscillator, n=n))
        r_true = rank_for(sub, channel)
        dt = sub.oscillator.dt

        # 系统实例固定（相位按实现随机），σ_true 由聚合 SNR 反解
        rng0 = derive_rng(sub.run.seed, channel, snr_db, m, "exp3", "phase", 0)
        system = build(sub, channel, rng0, m)
        sigma_true = sigma_from_snr(system.X, snr_db, convention=sub.noise.convention)
        LOG.info("实验三 [%s] n=%d SNR=%g sigma_true=%.4e r_true=%d",
                 channel, n, snr_db, sigma_true, r_true)

        acc: dict[tuple[str, str, float], list[NDArray]] = {}
        rank_mismatch: dict[tuple[str, float], dict[str, int]] = {
            (e, k): {"over": 0, "under": 0, "exact": 0}
            for e in estimators for k in k_grid
        }
        adapt_sigma_ratio: list[float] = []
        adapt_rank_ok = 0

        for j in range(j_total):
            if progress is not None and j % max(1, j_total // 10) == 0:
                progress(j, j_total)
            # 同一噪声实现下扫描全部 k —— 公共随机数，保证 k 之间可比
            rng = derive_rng(sub.run.seed, channel, snr_db, m, "exp3", "noise", j)
            Xn, Yn, _ = inject(system.X, system.Y, sigma_true, rng, mode="trajectory")
            sv = np.linalg.svd(np.vstack([Xn, Yn]), compute_uv=False)
            m2, n_col = 2 * system.n, system.m
            # 估计器截断秩的上界是 min(n, m)（X 的维数），不是 min(2n, m)
            r_max_fit = min(system.n, system.m)
            for est_name in estimators:
                for k in k_grid:
                    sigma_hat = k * sigma_true
                    try:
                        e = estimate_rank(sv, sigma_hat=sigma_hat, m=m2, n=n_col,
                                          method=est_name)
                    except RankError:
                        continue
                    rh = int(e.rank)
                    if rh < 1 or rh > r_max_fit:
                        # 秩越界：记为该实现的失效（不计入偏差统计）
                        rank_mismatch[(est_name, k)]["over" if rh > r_max_fit else "under"] += 1
                        continue
                    for name, cls in (("dmd", DMD), ("tdmd", TDMD)):
                        r = cls().fit(Xn, Yn, rh, dt)
                        pr = pair_to_truth(r.freqs, system.f_true)
                        acc.setdefault((est_name, name, k), []).append(
                            pr.freqs if pr.ok else np.full(system.f_true.size, np.nan))
                    mm = rank_mismatch[(est_name, k)]
                    if rh > r_true:
                        mm["over"] += 1
                    elif rh < r_true:
                        mm["under"] += 1
                    else:
                        mm["exact"] += 1

            # 数据驱动 σ̂（自举）
            try:
                sh, rh = adaptive_sigma(np.vstack([Xn, Yn]), method="gavish_donoho")
                adapt_sigma_ratio.append(sh / sigma_true)
                if rh == r_true:
                    adapt_rank_ok += 1
            except RankError:
                pass

        # 汇总
        for (est_name, name, k), seq in acc.items():
            arr = np.asarray(seq, dtype=float)
            if arr.ndim != 2:
                continue
            for mode in range(arr.shape[1]):
                e = (arr[:, mode] - system.f_true[mode]) / system.f_true[mode]
                good = e[np.isfinite(e)]
                row: dict[str, Any] = {
                    "channel": channel, "n": n, "snr_db": snr_db, "m": m,
                    "estimator": est_name, "method": name, "k": k, "mode": mode + 1,
                    "j_total": j_total, "j_valid": int(good.size),
                    "pair_fail_rate": 1.0 - good.size / max(j_total, 1),
                    "r_true": r_true,
                }
                if good.size >= 10:
                    row.update(summarize(good, bootstrap=1000).as_row())
                rows.append(row)

        for est_name in estimators:
            for k in k_grid:
                mm = rank_mismatch[(est_name, k)]
                tot = sum(mm.values())
                rank_rows.append({
                    "channel": channel, "estimator": est_name, "k": k,
                    "n": n, "snr_db": snr_db, "j_total": j_total,
                    "p_exact": mm["exact"] / max(tot, 1),
                    "p_over": mm["over"] / max(tot, 1),
                    "p_under": mm["under"] / max(tot, 1),
                    "depends_on_sigma": est_name != "gavish_donoho",
                })
        adapt_rows.append({
            "channel": channel, "n": n, "j_total": j_total,
            "sigma_hat_over_true_mean": float(np.mean(adapt_sigma_ratio))
            if adapt_sigma_ratio else float("nan"),
            "sigma_hat_over_true_std": float(np.std(adapt_sigma_ratio, ddof=1))
            if len(adapt_sigma_ratio) > 1 else float("nan"),
            "rank_exact_rate": adapt_rank_ok / max(len(adapt_sigma_ratio), 1),
        })

    table = pd.DataFrame(rows)
    return Exp3Result(
        table=table,
        rank_table=pd.DataFrame(rank_rows),
        robust=_robust_interval(table),
        adaptive=pd.DataFrame(adapt_rows),
        fingerprint=config_fingerprint(cfg),
        meta={"snr_db": snr_db, "m": m, "j_total": j_total, "k_grid": list(k_grid),
              "samples": samples},
    )


def _safe_max(series) -> float:
    """全 NaN 时返回 NaN，不触发 RuntimeWarning。"""
    v = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    return float(v.max()) if v.size else float("nan")


def _robust_interval(table: pd.DataFrame) -> pd.DataFrame:
    """鲁棒区间：在哪些 k 上 TDMD(r̂) 的 MSE 仍优于 DMD(r̂)。"""
    if table.empty:
        return table
    out: list[dict[str, Any]] = []
    for (ch, est_name, k), g in table.groupby(["channel", "estimator", "k"]):
        d = g[g["method"] == "dmd"]
        t = g[g["method"] == "tdmd"]
        if d.empty or t.empty:
            continue
        mse_d = _safe_max(d["mse_decomp"])
        mse_t = _safe_max(t["mse_decomp"])
        fail = _safe_max(g["pair_fail_rate"])
        ok = bool(np.isfinite(mse_d) and np.isfinite(mse_t)
                  and mse_t < mse_d and fail <= PAIR_FAIL_GATE)
        out.append({
            "channel": ch, "estimator": est_name, "k": k,
            "mse_dmd": mse_d, "mse_tdmd": mse_t,
            "mse_ratio": mse_t / mse_d if mse_d else float("nan"),
            "pair_fail_rate": fail,
            "tdmd_仍占优": bool(ok),
        })
    return pd.DataFrame(out)
