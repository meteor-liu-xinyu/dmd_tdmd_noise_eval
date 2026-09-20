"""实验七 · 秩估计方案的补救效果。

**背景（实验六暴露的问题）**：`σ` 未知时用"残差自举 + 交替迭代"定秩，
`σ̂` 依赖型判据会收敛到**虚假不动点**——`σ̂` 偏低 → 阈值偏低 → 秩高估 →
残差偏小 → `σ̂` 更低 → …… 迭代**收敛**但结果全错，且无法自我察觉。
实测 `marchenko_pastur` 的秩从真值 2 跑到 16，端到端 MSE 退化最多 2333 倍。

**本实验检验四种补救方案**（定义见 `estimators/rank.py`）：

| 方案 | 机理 | 预期 |
|---|---|---|
| `iter` | 原版交替迭代（基线） | 落在虚假不动点 |
| **`single`** | 先用 `gavish_donoho`（不依赖 σ̂）定秩，再由残差得 σ̂，最后**一次性**应用目标判据 —— **断掉反馈回路** | 应有效 |
| `multi` | 多起点迭代取中位 | 若吸引域整体偏移则无效 |
| `damped` | 阻尼迭代 | **只降低速率，不改变不动点位置** |

**评判口径**：不看秩判对率，而看**端到端频率精度**（配对 bootstrap MSE 比 vs oracle 秩）。
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
from dmdnoise.estimators.rank import (
    SCHEMES,
    VALID_METHODS,
    RankError,
    robust_sigma_rank,
)
from dmdnoise.mc import derive_rng
from dmdnoise.metrics import paired_ratio, summarize
from dmdnoise.sim import build, inject, rank_for, sigma_from_snr

LOG = logging.getLogger(__name__)

CRITERIA: tuple[str, ...] = ("gavish_donoho", "marchenko_pastur", "energy")
DEFAULT_COMBOS: tuple[tuple[str, int], ...] = (("complex", 8), ("real", 16))
DEFAULT_SNRS: tuple[float, ...] = (5.0, 10.0)
METHOD = "tdmd"          # 端到端只评 TDMD（实验一已确认其全面不劣）
M_DEFAULT = 200
J_DEFAULT = 2000
PAIR_FAIL_GATE = 0.20


def label(criterion: str, scheme: str) -> str:
    return f"{criterion}:{scheme}"


@dataclass
class Exp7Result:
    table: pd.DataFrame        # 逐 (通道, SNR, 判据, 方案) 的端到端指标
    ranks: pd.DataFrame        # 秩判定统计
    comparison: pd.DataFrame   # vs oracle 的 MSE 比（配对 bootstrap）
    errors: pd.DataFrame       # 逐实现误差长表
    fingerprint: str
    meta: dict[str, Any] = field(default_factory=dict)


def _clamp(r: int, n: int, m: int) -> int:
    return int(min(max(r, 1), min(n, m)))


def run(cfg: Config, *, combos: Sequence[tuple[str, int]] = DEFAULT_COMBOS,
        snrs: Sequence[float] = DEFAULT_SNRS, criteria: Sequence[str] = CRITERIA,
        schemes: Sequence[str] = SCHEMES, m: int = M_DEFAULT,
        j_total: int = J_DEFAULT, progress: Any = None) -> Exp7Result:
    strategies = [label(c, s) for c in criteria for s in schemes]
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

            est: dict[str, list[NDArray]] = {s: [] for s in strategies}
            okv: dict[str, list[bool]] = {s: [] for s in strategies}
            hist: dict[str, list[int]] = {s: [] for s in strategies}
            sig_hist: dict[str, list[float]] = {s: [] for s in strategies}
            f_true: NDArray | None = None
            sigma_true = float("nan")

            for j in range(j_total):
                rng_p = derive_rng(sub.run.seed, channel, snr, m, "e7", "phase", j)
                system = build(sub, channel, rng_p, m)
                if f_true is None:
                    f_true = system.f_true
                    sigma_true = sigma_from_snr(system.X, snr,
                                                convention=sub.noise.convention)
                rng_n = derive_rng(sub.run.seed, channel, snr, m, "e7", "noise", j)
                Xn, Yn, _ = inject(system.X, system.Y, sigma_true, rng_n,
                                   mode=sub.noise.mode)
                Z = np.vstack([Xn, Yn])
                sv = np.linalg.svd(Z, compute_uv=False)

                for crit in criteria:
                    for sch in schemes:
                        tag = label(crit, sch)
                        try:
                            sh, rh = robust_sigma_rank(Z, method=crit, scheme=sch,
                                                       sv=sv)
                        except RankError:
                            sh, rh = np.nan, 0
                        hist[tag].append(int(rh))
                        sig_hist[tag].append(float(sh))
                        try:
                            res = REGISTRY[METHOD]().fit(Xn, Yn, _clamp(rh, n, m), dt)
                        except Exception:                    # noqa: BLE001
                            est[tag].append(np.full(int(system.f_true.size), np.nan))
                            okv[tag].append(False)
                            continue
                        pr = pair_to_truth(res.freqs, system.f_true)
                        if pr.ok:
                            est[tag].append(pr.freqs)
                            okv[tag].append(True)
                        else:
                            est[tag].append(np.full(int(system.f_true.size), np.nan))
                            okv[tag].append(False)

                # oracle 参照
                res_o = REGISTRY[METHOD]().fit(Xn, Yn, r_true, dt)
                po = pair_to_truth(res_o.freqs, system.f_true)
                est.setdefault("oracle", []).append(
                    po.freqs if po.ok else np.full(int(system.f_true.size), np.nan))
                okv.setdefault("oracle", []).append(bool(po.ok))
                hist.setdefault("oracle", []).append(r_true)
                sig_hist.setdefault("oracle", []).append(sigma_true)

            assert f_true is not None
            n_modes = int(f_true.size)
            all_tags = strategies + ["oracle"]

            for tag in all_tags:
                arr = np.asarray(est[tag], dtype=float)
                ok = np.asarray(okv[tag], dtype=bool)
                for jj in range(j_total):
                    if not ok[jj]:
                        continue
                    rec: dict[str, Any] = {"channel": channel, "n_axis": n,
                                           "snr_db": snr, "strategy": tag, "j": jj}
                    for mode in range(n_modes):
                        rec[f"err{mode + 1}"] = (arr[jj, mode] - f_true[mode]) / f_true[mode]
                    err_rows.append(rec)

                rh_arr = np.asarray(hist[tag], dtype=int)
                sh_arr = np.asarray(sig_hist[tag], dtype=float)
                rank_rows.append({
                    "channel": channel, "n_axis": n, "snr_db": snr, "strategy": tag,
                    "criterion": tag.split(":")[0], "scheme": tag.split(":")[1]
                    if ":" in tag else "-",
                    "r_true": r_true, "j_total": j_total,
                    "r_hat_mean": float(rh_arr.mean()),
                    "p_exact": float((rh_arr == r_true).mean()),
                    "p_over": float((rh_arr > r_true).mean()),
                    "p_under": float((rh_arr < r_true).mean()),
                    "sigma_hat_over_true": float(np.nanmean(sh_arr) / sigma_true)
                    if sigma_true else float("nan"),
                })

                for mode in range(n_modes):
                    e = (arr[ok, mode] - f_true[mode]) / f_true[mode] if ok.any() \
                        else np.empty(0)
                    row: dict[str, Any] = {
                        "channel": channel, "n_axis": n, "snr_db": snr,
                        "strategy": tag, "mode": mode + 1, "r_true": r_true,
                        "j_total": j_total, "j_valid": int(e.size),
                        "pair_fail_rate": 1.0 - e.size / max(j_total, 1),
                    }
                    if e.size >= 10:
                        row.update(summarize(e, bootstrap=1000).as_row())
                    rows.append(row)

    errors = pd.DataFrame(err_rows)
    return Exp7Result(
        table=pd.DataFrame(rows),
        ranks=pd.DataFrame(rank_rows),
        comparison=_compare(errors),
        errors=errors,
        fingerprint=config_fingerprint(cfg),
        meta={"m": m, "j_total": j_total, "snrs": list(snrs),
              "combos": [list(c) for c in combos],
              "criteria": list(criteria), "schemes": list(schemes)},
    )


def _compare(errors: pd.DataFrame) -> pd.DataFrame:
    """vs oracle 的 MSE 比（配对 bootstrap）。"""
    if errors.empty:
        return errors
    out: list[dict[str, Any]] = []
    for k, g in errors.groupby(["channel", "n_axis", "snr_db"]):
        modes = [c for c in g.columns if c.startswith("err")]
        base = g[g["strategy"] == "oracle"]
        for tag in [s for s in g["strategy"].unique() if s != "oracle"]:
            other = g[g["strategy"] == tag]
            merged = base.merge(other, on="j", suffixes=("_o", "_s"))
            if len(merged) < 20:
                continue
            for mode in modes:
                a = merged[f"{mode}_o"].to_numpy(float)
                b = merged[f"{mode}_s"].to_numpy(float)
                pr = paired_ratio(b, a, stat="mse", bootstrap=2000)
                out.append({
                    "channel": k[0], "n_axis": k[1], "snr_db": k[2],
                    "mode": int(mode[-1]), "strategy": tag,
                    "criterion": tag.split(":")[0],
                    "scheme": tag.split(":")[1] if ":" in tag else "oracle",
                    "n_pair": pr.n, "mse_ratio": pr.point,
                    "mse_lo": pr.ci[0], "mse_hi": pr.ci[1],
                    "退化显著": bool(pr.ci[0] > 1.0),
                })
    return pd.DataFrame(out)


def verdict(res: Exp7Result) -> pd.DataFrame:
    """按 (判据, 方案) 汇总：秩判对率、σ̂ 准确度、端到端退化中位数。"""
    if res.ranks.empty:
        return res.ranks
    rk = res.ranks[res.ranks["strategy"] != "oracle"]
    cmp_ = res.comparison
    out: list[dict[str, Any]] = []
    for (crit, sch), g in rk.groupby(["criterion", "scheme"]):
        tag = label(crit, sch)
        c = cmp_[cmp_["strategy"] == tag] if not cmp_.empty else pd.DataFrame()
        out.append({
            "criterion": crit, "scheme": sch,
            "秩判对率": float(g["p_exact"].mean()),
            "秩高估率": float(g["p_over"].mean()),
            "秩低估率": float(g["p_under"].mean()),
            "sigma_hat/sigma_true": float(g["sigma_hat_over_true"].mean()),
            "MSE比_中位": float(np.median(c["mse_ratio"])) if not c.empty else np.nan,
            "MSE比_最大": float(np.max(c["mse_ratio"])) if not c.empty else np.nan,
            "退化显著的组合数": int(c["退化显著"].sum()) if not c.empty else 0,
            "总组合数": int(len(c)),
        })
    return pd.DataFrame(out).sort_values(["criterion", "scheme"])
