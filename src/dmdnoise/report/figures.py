"""图表生成：fig1–fig10。

规格依据：docs/experiment-protocol.md §6。

**强制约定**：
  - 只从 `results/tables/*.csv` 读数据，**禁止硬编码任何数值**；
  - 数值缺失（实验未跑）时跳过该图并给出 warning，不伪造数据；
  - 偏差不可分辨的网格点只画上界，不画点估计。
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LOG = logging.getLogger(__name__)

# 中文字体（Windows 常见可用字体；缺失时回退 DejaVu 并只影响中文标签）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

#: 两法统一样式（色盲友好、黑白打印可辨）
STYLE = {
    "dmd": dict(color="#c0392b", marker="o", ls="-", label="标准 DMD"),
    "tdmd": dict(color="#2471a3", marker="s", ls="--", label="TDMD"),
}


def _setup() -> None:
    """确保图目录存在。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]


def _load(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        LOG.warning("缺少数据文件，跳过：%s", path.name)
        return None
    df = pd.read_csv(path)
    if df.empty:
        LOG.warning("数据为空，跳过：%s", path.name)
        return None
    return df


def _log_ticks(ax, ticks, *, base2: bool = False) -> None:
    """把 x 轴设为对数刻度，只保留给定刻度标签，去掉次刻度自动标注。"""
    ax.set_xscale("log", base=2) if base2 else ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def _save(fig, fig_dir: Path, name: str) -> Path:
    fig_dir.mkdir(parents=True, exist_ok=True)
    # 有总标题时给它留出上方空间，避免与子图标题重叠
    rect = [0, 0, 1, 0.93] if fig._suptitle is not None else None
    try:
        fig.tight_layout(rect=rect) if rect else fig.tight_layout()
    except Exception:                                  # noqa: BLE001
        pass
    out = fig_dir / f"{name}.png"
    fig.savefig(out)
    plt.close(fig)
    LOG.info("已生成 %s", out.name)
    return out


def _rows(df: pd.DataFrame, **kw) -> pd.DataFrame:
    m = pd.Series(True, index=df.index)
    for k, v in kw.items():
        if k not in df.columns:
            return df.iloc[0:0]
        if isinstance(v, (tuple, list)):
            m &= df[k].isin(list(v))
        else:
            m &= df[k] == v
    return df[m]


def _errbar(ax, x, row, style, capsize=3):
    """按可分辨性决定画点估计还是上界。"""
    if bool(row.get("resolvable", False)):
        ax.errorbar(x, abs(row["bias"]), yerr=row["delta_min"], capsize=capsize, **style)
    else:
        ax.errorbar(x, row["delta_min"], yerr=None, capsize=capsize, alpha=0.45,
                    **{**style, "label": None, "ls": ":"})


# --------------------------------------------------------------------------- fig1
def fig1_bias_vs_channels(t: dict[str, pd.DataFrame], d: Path) -> Path | None:
    """fig1 偏差 vs 观测维数 n（实验一核心图）。"""
    df = t.get("exp1")
    if df is None:
        return None
    sel = _rows(df, axis="n")
    if sel.empty:
        return None
    channels = sorted(sel["channel"].unique())
    fig, axes = plt.subplots(1, len(channels), figsize=(5.2 * len(channels), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, ch in zip(axes, channels):
        for method in ("dmd", "tdmd"):
            g = _rows(sel, channel=ch, method=method)
            if g.empty:
                continue
            agg = g.groupby("n_axis").agg(
                bias=("bias", lambda s: float(np.abs(s).max())),
                delta_min=("delta_min", "max"),
                resolvable=("resolvable", lambda s: bool(s.eq(True).any())),
                pair_fail=("pair_fail_rate", "max"),
            ).reset_index().sort_values("n_axis")
            ok = agg["pair_fail"] <= 0.20
            ax.errorbar(agg.loc[ok, "n_axis"], agg.loc[ok, "bias"].abs(),
                        yerr=agg.loc[ok, "delta_min"], capsize=3, **STYLE[method])
            ax.errorbar(agg.loc[~ok, "n_axis"], agg.loc[~ok, "bias"].abs(),
                        yerr=agg.loc[~ok, "delta_min"], capsize=3, alpha=0.35,
                        **{**STYLE[method], "label": None, "ls": ":"})
        ax.set_yscale("log")
        _log_ticks(ax, [4, 8, 16, 32, 64], base2=True)
        ax.set_xlabel("观测维数 n（通道数或嵌入维数 L）")
        ax.set_title(f"通道形式：{ch}")
    axes[0].set_ylabel("相对频率偏差 |bias|")
    axes[0].legend(loc="upper right", fontsize=9)
    fig.suptitle("图 1  偏差随观测维数的变化（实线=可报告，虚线=配对失效率>20%不计）")
    return _save(fig, d, "fig1_bias_vs_channels")


# --------------------------------------------------------------------------- fig2
def fig2_bias_vs_snr(t: dict[str, pd.DataFrame], d: Path) -> Path | None:
    """fig2 偏差 vs SNR（实验一）。"""
    df = t.get("exp1")
    if df is None:
        return None
    sel = _rows(df, axis="snr")
    if sel.empty:
        return None
    channels = sorted(sel["channel"].unique())
    fig, axes = plt.subplots(1, len(channels), figsize=(5.2 * len(channels), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, ch in zip(axes, channels):
        for method in ("dmd", "tdmd"):
            g = _rows(sel, channel=ch, method=method)
            if g.empty:
                continue
            agg = g.groupby("snr_db").agg(
                bias=("bias", lambda s: float(np.abs(s).max())),
                delta_min=("delta_min", "max"),
                pair_fail=("pair_fail_rate", "max"),
            ).reset_index().sort_values("snr_db")
            ok = agg["pair_fail"] <= 0.20
            ax.errorbar(agg.loc[ok, "snr_db"], agg.loc[ok, "bias"].abs(),
                        yerr=agg.loc[ok, "delta_min"], capsize=3, **STYLE[method])
            ax.errorbar(agg.loc[~ok, "snr_db"], agg.loc[~ok, "bias"].abs(),
                        yerr=agg.loc[~ok, "delta_min"], capsize=3, alpha=0.35,
                        **{**STYLE[method], "label": None, "ls": ":"})
        ax.set_yscale("log")
        ax.set_xlabel("SNR (dB)")
        ax.set_title(f"通道形式：{ch}")
    axes[0].set_ylabel("相对频率偏差 |bias|")
    axes[0].legend(loc="lower left", fontsize=9)
    fig.suptitle("图 2  偏差随信噪比的变化（虚线段配对失效率>20%，不得报告）")
    return _save(fig, d, "fig2_bias_vs_snr")


# --------------------------------------------------------------------------- fig3
def fig3_pair_fail_vs_snr(t: dict[str, pd.DataFrame], d: Path) -> Path | None:
    """fig3 配对失败率 vs SNR —— 决定可用 SNR 窗口。"""
    df = t.get("exp1")
    if df is None:
        return None
    sel = _rows(df, axis="snr")
    if sel.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for ch, mk in zip(sorted(sel["channel"].unique()), ("o", "s", "^")):
        g = _rows(sel, channel=ch)
        agg = g.groupby(["snr_db", "method"])["pair_fail_rate"].max().reset_index()
        for method in ("dmd", "tdmd"):
            gg = agg[agg["method"] == method].sort_values("snr_db")
            if gg.empty:
                continue
            ax.plot(gg["snr_db"], 100 * gg["pair_fail_rate"], marker=mk, ls="-",
                    label=f"{ch} / {STYLE[method]['label']}")
    ax.axhline(20, color="k", ls="--", lw=1.2)
    ax.text(ax.get_xlim()[0], 21.5, "门槛 20%（超过即不得报告偏差）", fontsize=9)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("配对失败率 (%)")
    ax.set_title("图 3  配对失效率决定可用 SNR 窗口")
    ax.legend(fontsize=8, ncol=2)
    return _save(fig, d, "fig3_pair_fail_vs_snr")


# --------------------------------------------------------------------------- fig4
def fig4_amp_ratio(t: dict[str, pd.DataFrame], d: Path) -> Path | None:
    """fig4 模态幅度比的影响（实验一 amp 轴）。"""
    df = t.get("exp1")
    if df is None:
        return None
    sel = _rows(df, axis="amp")
    if sel.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4))
    for method in ("dmd", "tdmd"):
        g = _rows(sel, method=method)
        if g.empty:
            continue
        agg = g.groupby("amp_ratio").agg(
            bias=("bias", lambda s: float(np.abs(s).max())),
            delta_min=("delta_min", "max"),
            pair_fail=("pair_fail_rate", "max"),
        ).reset_index().sort_values("amp_ratio")
        axes[0].errorbar(agg["amp_ratio"], agg["bias"].abs(), yerr=agg["delta_min"],
                         capsize=3, **STYLE[method])
        axes[1].plot(agg["amp_ratio"], 100 * agg["pair_fail"], **STYLE[method])
    axes[1].axhline(20, color="k", ls="--", lw=1.2)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("相对频率偏差 |bias|")
    axes[1].set_ylabel("配对失败率 (%)")
    for ax in axes:
        ax.set_xlabel("模态可观测能量比 $E_1/E_2$")
    axes[0].legend(fontsize=9)
    fig.suptitle("图 4  模态幅度比对偏差与配对成功率的影响")
    return _save(fig, d, "fig4_amp_ratio")


# --------------------------------------------------------------------------- fig5
def fig5_variance_cost(t: dict[str, pd.DataFrame], d: Path) -> Path | None:
    """fig5 方差代价森林图：std_TDMD / std_DMD 及配对 bootstrap CI。"""
    p = t.get("exp1_paired")
    if p is None:
        return None
    std = p[p["stat"] == "std"].copy()
    std = std[std["insufficient"].eq(False)] if "insufficient" in std.columns else std
    if std.empty:
        return None
    std["tag"] = (std["channel"] + " · " + std["axis"] + " · SNR"
                  + std["snr_db"].astype(str) + " · n" + std["n_axis"].astype(str)
                  + " · r" + std["amp_ratio"].astype(str) + " · m" + std["mode"].astype(str))
    std = std.sort_values("ratio").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.8, max(3.5, 0.26 * len(std))))
    y = np.arange(len(std))
    ok = std["paired_effective"].eq(True)
    ax.errorbar(std.loc[ok, "ratio"], y[ok.values],
                xerr=[std.loc[ok, "ratio"] - std.loc[ok, "ci_lo"],
                      std.loc[ok, "ci_hi"] - std.loc[ok, "ratio"]],
                fmt="o", color="#2471a3", capsize=3, ms=4, label="配对 bootstrap 有效")
    ax.errorbar(std.loc[~ok, "ratio"], y[~ok.values],
                xerr=[std.loc[~ok, "ratio"] - std.loc[~ok, "ci_lo"],
                      std.loc[~ok, "ci_hi"] - std.loc[~ok, "ratio"]],
                fmt="o", color="#7f8c8d", alpha=0.5, capsize=3, ms=4,
                label="配对失效，仅参考")
    ax.axvline(1.0, color="k", ls="--", lw=1.2)
    ax.text(1.01, -1.2, "比值=1（无代价）", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(std["tag"], fontsize=7)
    ax.set_xlabel("$\\mathrm{std}_{TDMD} / \\mathrm{std}_{DMD}$")
    ax.set_title("图 5  方差代价：全部可用网格点的 std 比均 < 1（TDMD 方差更小）")
    ax.legend(fontsize=8, loc="lower right")
    return _save(fig, d, "fig5_variance_cost")


# --------------------------------------------------------------------------- fig6
def fig6_bias_vs_m(t: dict[str, pd.DataFrame], d: Path) -> Path | None:
    """fig6 偏差 vs m，两种 SNR 归一化口径对比（实验二核心图）。"""
    mm = t.get("exp2_m_max")
    pm = t.get("exp2_per_m")
    if mm is None or pm is None:
        return None
    channels = sorted(mm["channel"].unique())
    fig, axes = plt.subplots(1, len(channels), figsize=(5.4 * len(channels), 4.2),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, ch in zip(axes, channels):
        for method in ("dmd", "tdmd"):
            base = STYLE[method]
            for df, ls_extra, tag in ((mm, (1, (4, 2)), "sigma 由 m_max 定"),
                                      (pm, (1, (1, 1)), "逐 m 归一化 SNR")):
                g = _rows(df, channel=ch, method=method)
                if g.empty:
                    continue
                agg = g.groupby("m")["bias"].apply(
                    lambda s: float(np.abs(s).max())).reset_index()
                ax.plot(agg["m"], agg["bias"], marker=base["marker"], color=base["color"],
                        ls=base["ls"], dashes=ls_extra[1],
                        label=f"{base['label']}（{tag}）", alpha=0.9)
        ax.set_yscale("log")
        _log_ticks(ax, [50, 100, 200, 500])
        ax.set_xlabel("快照数 m")
        ax.set_title(f"通道形式：{ch}")
    axes[0].set_ylabel("相对频率偏差 |bias|")
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle("图 6  SNR 归一化口径决定了「偏差随 m 上升」是否成立")
    return _save(fig, d, "fig6_bias_vs_m")


# --------------------------------------------------------------------------- fig7
def fig7_rmse_vs_m(t: dict[str, pd.DataFrame], d: Path) -> Path | None:
    """fig7 RMSE vs m 的 log-log 图，含理论 m^-1/2 参考线。"""
    pm = t.get("exp2_per_m")
    if pm is None:
        return None
    channels = sorted(pm["channel"].unique())
    fig, axes = plt.subplots(1, len(channels), figsize=(5.4 * len(channels), 4.2))
    axes = np.atleast_1d(axes)
    for ax, ch in zip(axes, channels):
        ref = None
        for method in ("dmd", "tdmd"):
            g = _rows(pm, channel=ch, method=method)
            if g.empty:
                continue
            agg = g.groupby("m")["rmse"].apply(lambda s: float(s.max())).reset_index()
            ax.plot(agg["m"], agg["rmse"], **STYLE[method])
            if ref is None:
                ref = (agg["m"].to_numpy(), agg["rmse"].to_numpy())
        if ref is not None:      # 以首点为锚画 m^-1/2 参考线
            m0, r0 = ref[0][0], ref[1][0]
            mgrid = np.array(sorted(pm["m"].unique()), dtype=float)
            ax.plot(mgrid, r0 * (mgrid / m0) ** -0.5, color="k", ls=":", lw=1.4,
                    label="理论 $m^{-1/2}$")
        ax.set_yscale("log")
        _log_ticks(ax, [50, 100, 200, 500])
        ax.set_xlabel("快照数 m")
        ax.set_title(f"通道形式：{ch}")
    axes[0].set_ylabel("RMSE")
    axes[0].legend(fontsize=8)
    fig.suptitle("图 7  RMSE 的样本量标度律（逐 m 归一化 SNR）")
    return _save(fig, d, "fig7_rmse_vs_m")


# --------------------------------------------------------------------------- fig8
def fig8_slopes(t: dict[str, pd.DataFrame], d: Path) -> Path | None:
    """fig8 斜率汇总（两种口径）。"""
    mm = t.get("exp2_slopes_m_max")
    pm = t.get("exp2_slopes_per_m")
    if mm is None or pm is None:
        return None
    rows = []
    for tag, df in (("sigma 由 m_max 定", mm), ("逐 m 归一化 SNR", pm)):
        for _, r in df.iterrows():
            rows.append(dict(tag=tag, channel=r["channel"], method=r["method"],
                             mode=int(r["mode"]), stat="偏差",
                             slope=r["bias_slope"], lo=r["bias_slope_lo"],
                             hi=r["bias_slope_hi"]))
            rows.append(dict(tag=tag, channel=r["channel"], method=r["method"],
                             mode=int(r["mode"]), stat="RMSE",
                             slope=r["rmse_slope"], lo=r["rmse_slope_lo"],
                             hi=r["rmse_slope_hi"]))
    s = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, stat in zip(axes, ("偏差", "RMSE")):
        g = s[s["stat"] == stat].reset_index(drop=True)
        y = np.arange(len(g))
        for tag, col in (("sigma 由 m_max 定", "#c0392b"), ("逐 m 归一化 SNR", "#2471a3")):
            sel = g["tag"] == tag
            ax.errorbar(g.loc[sel, "slope"], y[sel.values],
                        xerr=[g.loc[sel, "slope"] - g.loc[sel, "lo"],
                              g.loc[sel, "hi"] - g.loc[sel, "slope"]],
                        fmt="o", color=col, capsize=3, ms=4, label=tag)
        ax.axvline(0, color="k", ls="--", lw=1.1)
        if stat == "RMSE":
            ax.axvline(-0.5, color="g", ls=":", lw=1.4)
            ax.text(-0.49, -1.0, "理论 −1/2", color="g", fontsize=9)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r.channel} · {r.method} · m{r['mode']}"
                            for _, r in g.iterrows()], fontsize=8)
        ax.set_xlabel(f"{stat} 的 log-log 斜率")
        ax.set_title(f"{stat} 斜率")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("图 8  斜率汇总：口径改变后 偏差斜率由正转负")
    return _save(fig, d, "fig8_slopes")


# --------------------------------------------------------------------------- fig9
def fig9_rank_robustness(t: dict[str, pd.DataFrame], d: Path) -> Path | None:
    """fig9 秩判据对 σ̂ 误估的鲁棒性（实验三核心图）。"""
    df = t.get("exp3_rank")
    if df is None:
        return None
    channels = sorted(df["channel"].unique())
    ests = sorted(df["estimator"].unique())
    fig, axes = plt.subplots(1, len(channels), figsize=(5.4 * len(channels), 4.2),
                             sharey=True)
    axes = np.atleast_1d(axes)
    cmap = plt.get_cmap("tab10")
    for ax, ch in zip(axes, channels):
        for i, est in enumerate(ests):
            g = _rows(df, channel=ch, estimator=est).sort_values("k")
            if g.empty:
                continue
            dep = bool(g["depends_on_sigma"].iloc[0])
            ax.plot(g["k"], 100 * g["p_exact"], marker="o", ms=4, color=cmap(i),
                    ls="-" if dep else "--",
                    label=f"{est}" + ("" if dep else "（免标定）"))
        ax.axvline(1.0, color="k", ls=":", lw=1.2)
        ax.set_xscale("log")
        ax.set_xlabel("$k = \\hat{\\sigma}/\\sigma_{true}$")
        ax.set_title(f"通道形式：{ch}")
        ax.set_xticks([0.5, 1.0, 1.5, 2.0, 3.0])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes[0].set_ylabel("秩判定正确率 (%)")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("图 9  秩判据对噪声水平误估的鲁棒性（gavish_donoho 完全免疫）")
    return _save(fig, d, "fig9_rank_robustness")


# --------------------------------------------------------------------------- fig10
def fig10_decision_map(t: dict[str, pd.DataFrame], d: Path) -> Path | None:
    """fig10 适用边界决策图：偏差可分辨性 + 配对成功率 → 四类区域。"""
    df = t.get("exp1")
    if df is None:
        return None
    sel = _rows(df, axis="n")
    if sel.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, ch in zip(axes, sorted(sel["channel"].unique())):
        g = _rows(sel, channel=ch, method="dmd")
        if g.empty:
            continue
        agg = g.groupby("n_axis").agg(
            resolvable=("resolvable", lambda s: bool(s.eq(True).any())),
            pair_fail=("pair_fail_rate", "max"),
        ).reset_index().sort_values("n_axis")
        for _, r in agg.iterrows():
            bad_fail = r["pair_fail"] > 0.20
            color = ("#e74c3c" if bad_fail else
                     "#f39c12" if not r["resolvable"] else "#27ae60")
            label = ("③ 两者都不行" if bad_fail else
                     "① 标准 DMD 够用" if not r["resolvable"] else "② TDMD 值得用")
            ax.bar(r["n_axis"], 1, width=r["n_axis"] * 0.5, color=color, alpha=0.85)
            ax.text(r["n_axis"], 0.5, f"{int(r['n_axis'])}\n{label}", ha="center",
                    va="center", fontsize=7.5, color="white", weight="bold")
        _log_ticks(ax, [4, 8, 16, 32, 64], base2=True)
        ax.set_xlabel("观测维数 n")
        ax.set_yticks([])
        ax.set_title(f"通道形式：{ch}")
    fig.suptitle("图 10  适用边界决策图（按 n；SNR=10 dB）")
    return _save(fig, d, "fig10_decision_map")



# --------------------------------------------------------------------------- fig11
def fig11_robustness(t: dict[str, pd.DataFrame], d: Path) -> Path | None:
    """fig11 参数稳健性：偏差对模态间隔与窗内衰减的敏感性（实验四）。"""
    v = t.get("exp4_verdict")
    if v is None:
        return None
    v = v[v["可用"].eq(True)] if "可用" in v.columns else v
    if v.empty:
        return None
    channels = sorted(v["channel"].unique())
    fig, axes = plt.subplots(len(channels), 2, figsize=(11, 4.0 * len(channels)),
                             squeeze=False)
    for r, ch in enumerate(channels):
        g = v[v["channel"] == ch]
        # 左：偏差 vs 模态间隔（线 = 窗内衰减）
        for i, dec in enumerate(sorted(g["window_decay"].unique(), reverse=True)):
            gg = g[g["window_decay"] == dec].sort_values("f2_over_f1")
            if gg.empty:
                continue
            axes[r][0].plot(gg["f2_over_f1"], gg["DMD_偏差"], marker="o", ms=4,
                            color=plt.get_cmap("viridis")(i / 3.0),
                            label=f"衰减={dec:.2f}")
        # 右：偏差 vs 窗内衰减（线 = 模态间隔）
        for i, sep in enumerate(sorted(g["f2_over_f1"].unique())):
            gg = g[g["f2_over_f1"] == sep].sort_values("window_decay")
            if gg.empty:
                continue
            axes[r][1].plot(gg["window_decay"], gg["DMD_偏差"], marker="s", ms=4,
                            color=plt.get_cmap("plasma")(i / 3.0),
                            label=f"$f_2/f_1$={sep:.2f}")
        for c in (0, 1):
            axes[r][c].set_yscale("log")
            axes[r][c].set_ylabel("DMD 相对偏差 |bias|")
            axes[r][c].legend(fontsize=7, title=None)
        axes[r][0].set_xlabel("模态频率间隔 $f_2/f_1$")
        axes[r][1].set_xlabel("窗内幅度衰减 $d$")
        axes[r][0].set_title(f"{ch} · 偏差 vs 模态间隔")
        axes[r][1].set_title(f"{ch} · 偏差 vs 窗内衰减")
    fig.suptitle("图 11  参数稳健性：偏差对两个物理参数的敏感性（仅列可用网格点）")
    return _save(fig, d, "fig11_robustness")



# --------------------------------------------------------------------------- fig12
def fig12_crossover(t: dict[str, pd.DataFrame], d: Path) -> Path | None:
    """fig12 降偏比 rho 随模态间隔的变化与交叉点（实验五）。"""
    df = t.get("exp5_crossover")
    if df is None:
        return None
    cr = t.get("exp5_crossing")
    modes = sorted(df["mode"].unique())
    combos = sorted({(r["channel"], int(r["n"])) for _, r in df.iterrows()})
    fig, axes = plt.subplots(1, len(modes), figsize=(5.6 * len(modes), 4.4),
                             sharey=True)
    axes = np.atleast_1d(axes)
    cmap = plt.get_cmap("tab10")
    for ax, mode in zip(axes, modes):
        for i, (ch, n) in enumerate(combos):
            g = df[(df["channel"] == ch) & (df["n"] == n) & (df["mode"] == mode)]
            g = g[(g["pair_fail_rate"] <= 0.20) & g["dmd_resolvable"]]
            if g.empty:
                continue
            g = g.sort_values("f2_over_f1")
            col = cmap(i % 10)
            x = g["f2_over_f1"].to_numpy(float)
            ax.plot(x, g["rho"], marker="o", ms=4, color=col, label=f"{ch} n={n}")
            ax.fill_between(x, g["rho_lo"], g["rho_hi"], color=col, alpha=0.15)
        if cr is not None and not cr.empty:
            cm = cr[(cr["mode"] == mode) & np.isfinite(cr["crossing"])]
            for _, r in cm.iterrows():
                ax.axvline(r["crossing"], color="k", ls=":", lw=1.0, alpha=0.6)
        ax.axhline(1.0, color="k", ls="--", lw=1.4)
        ax.set_yscale("log")
        ax.set_xlabel("模态频率间隔 $f_2/f_1$")
        ax.set_title(f"模态 {mode}")
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel("降偏比 rho = |bias_TDMD| / |bias_DMD|")
    axes[0].annotate("rho=1：两法偏差相等", xy=(0.98, 1.0),
                 xycoords=("axes fraction", "data"), ha="right", va="bottom",
                 fontsize=8, color="k")
    fig.suptitle("图 12  降偏比随模态间隔的变化与交叉点（阴影为配对 bootstrap 95% CI）")
    return _save(fig, d, "fig12_crossover")



# --------------------------------------------------------------------------- fig13
def fig13_end_to_end(t: dict[str, pd.DataFrame], d: Path) -> Path | None:
    """fig13 端到端：噪声水平未知下的 sigma_hat、秩判定与频率精度退化（实验六）。"""
    rk = t.get("exp6_rank_stats")
    cmp_ = t.get("exp6_comparison")
    if rk is None:
        return None
    rk = rk[rk["strategy"] != "oracle"]
    if rk.empty:
        return None
    rk = rk.copy()
    rk["tag"] = (rk["channel"] + " | SNR" + rk["snr_db"].astype(str))
    tags = sorted(rk["tag"].unique())
    strategies = sorted(rk["strategy"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    cmap = plt.get_cmap("Set2")

    # (1) sigma_hat / sigma_true
    w = 0.8 / len(strategies)
    for i, st in enumerate(strategies):
        g = rk[rk["strategy"] == st].set_index("tag").reindex(tags)
        axes[0].bar(np.arange(len(tags)) + i * w, g["sigma_hat_over_true"], width=w,
                    color=cmap(i), label=st)
    axes[0].axhline(1.0, color="k", ls="--", lw=1.2)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("sigma_hat / sigma_true")
    axes[0].set_title("噪声水平估计")
    axes[0].text(0.02, 1.05, "真值 1.0", fontsize=8, transform=axes[0].get_yaxis_transform())

    # (2) 秩判对率：按策略分组（聚合会把三个策略混在一起，信息被稀释）
    for i, st in enumerate(strategies):
        g = rk[rk["strategy"] == st].set_index("tag").reindex(tags)
        axes[1].bar(np.arange(len(tags)) + i * w, g["p_exact"], width=w,
                    color=cmap(i), label=st)
    axes[1].set_xticks(np.arange(len(tags)) + w)
    axes[1].set_xticklabels(tags, fontsize=7, rotation=20, ha="right")
    axes[1].set_ylim(0, 1.08)
    axes[1].set_ylabel("秩判对率 P(r_hat = r_true)")
    axes[1].set_title("秩判定正确率（按策略）")
    axes[1].legend(fontsize=7, ncol=1, loc="upper right")

    # 
    # (3) 端到端 MSE 比 vs oracle
    if cmp_ is not None and not cmp_.empty:
        c = cmp_[cmp_["method"] == "tdmd"].copy()
        c["tag"] = c["channel"] + " SNR" + c["snr_db"].astype(str) + " m" + c["mode"].astype(str)
        c = c[np.isfinite(c["mse_ratio"])].sort_values("mse_ratio")
        y = np.arange(len(c))
        ok = c["退化显著"].eq(True)
        axes[2].errorbar(c.loc[~ok, "mse_ratio"], y[~ok.values],
                         xerr=[c.loc[~ok, "mse_ratio"] - c.loc[~ok, "mse_lo"],
                               c.loc[~ok, "mse_hi"] - c.loc[~ok, "mse_ratio"]],
                         fmt="o", color="#2471a3", capsize=3, ms=4, label="退化不显著")
        axes[2].errorbar(c.loc[ok, "mse_ratio"], y[ok.values],
                         xerr=[c.loc[ok, "mse_ratio"] - c.loc[ok, "mse_lo"],
                               c.loc[ok, "mse_hi"] - c.loc[ok, "mse_ratio"]],
                         fmt="o", color="#c0392b", capsize=3, ms=4, label="退化显著")
        axes[2].axvline(1.0, color="k", ls="--", lw=1.2)
        axes[2].set_yticks(y)
        axes[2].set_yticklabels(c["tag"] + " / " + c["strategy"], fontsize=6)
        axes[2].set_xlabel("MSE(数据驱动秩) / MSE(oracle 秩)")
        axes[2].set_title("端到端精度退化（TDMD）")
        axes[2].legend(fontsize=8, loc="lower right")
    fig.suptitle("图 13  端到端评估：噪声水平未知下的完整链路（gavish_donoho 零退化）")
    return _save(fig, d, "fig13_end_to_end")



# --------------------------------------------------------------------------- fig14
def fig14_rank_schemes(t: dict[str, pd.DataFrame], d: Path) -> Path | None:
    """fig14 秩估计方案的补救效果（实验七）。"""
    v = t.get("exp7_verdict")
    if v is None or v.empty:
        return None
    v = v[v["总组合数"] > 0]
    if v.empty:
        return None
    criteria = sorted(v["criterion"].unique())
    schemes = [x for x in ("iter", "single", "multi", "damped")
               if x in set(v["scheme"])]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    cmap = plt.get_cmap("Set2")
    x = np.arange(len(criteria))
    w = 0.8 / max(len(schemes), 1)

    def _vals(col, agg="first"):
        out = []
        for c in criteria:
            row = v[v["criterion"] == c].set_index("scheme").reindex(schemes)
            out.append(row[col].to_numpy(float))
        return np.array(out)          # (n_criteria, n_schemes)

    p_exact = _vals("秩判对率")
    for i, sch in enumerate(schemes):
        axes[0].bar(x + i * w, p_exact[:, i], width=w, color=cmap(i), label=sch)
    axes[0].set_xticks(x + w)
    axes[0].set_xticklabels(criteria, fontsize=8, rotation=12)
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("秩判对率")
    axes[0].set_title("秩判定正确率")
    axes[0].legend(fontsize=8)

    sh = _vals("sigma_hat/sigma_true")
    for i, sch in enumerate(schemes):
        axes[1].bar(x + i * w, sh[:, i], width=w, color=cmap(i), label=sch)
    axes[1].axhline(1.0, color="k", ls="--", lw=1.2)
    axes[1].set_yscale("log")
    axes[1].set_xticks(x + w)
    axes[1].set_xticklabels(criteria, fontsize=8, rotation=12)
    axes[1].set_ylabel("sigma_hat / sigma_true")
    axes[1].set_title("噪声水平估计（真值 1.0）")

    med = _vals("MSE比_中位")
    mx = _vals("MSE比_最大")
    for i, sch in enumerate(schemes):
        axes[2].bar(x + i * w, np.nan_to_num(med[:, i], nan=0.0), width=w,
                    color=cmap(i), label=sch)
        axes[2].plot(x + i * w, np.nan_to_num(mx[:, i], nan=np.nan), "k_", ms=8)
    axes[2].axhline(1.0, color="k", ls="--", lw=1.2)
    axes[2].set_yscale("log")
    axes[2].set_xticks(x + w)
    axes[2].set_xticklabels(criteria, fontsize=8, rotation=12)
    axes[2].set_ylabel("端到端 MSE 比 vs oracle")
    axes[2].set_title("端到端退化（柱=中位，横线=最大）")
    fig.suptitle("图 14  秩估计方案对比：single 是唯一有效补救，damped 与 multi 均无效")
    return _save(fig, d, "fig14_rank_schemes")


# --------------------------------------------------------------------------- 驱动
FIGURES = (fig1_bias_vs_channels, fig2_bias_vs_snr, fig3_pair_fail_vs_snr,
           fig4_amp_ratio, fig5_variance_cost, fig6_bias_vs_m, fig7_rmse_vs_m,
           fig8_slopes, fig9_rank_robustness, fig10_decision_map,
           fig11_robustness, fig12_crossover, fig13_end_to_end,
           fig14_rank_schemes)

_FILES = {
    "exp1": "exp1_bias_variance.csv",
    "exp1_paired": "exp1_paired.csv",
    "exp2_m_max": "exp2_sample_scan_m_max.csv",
    "exp2_per_m": "exp2_sample_scan_per_m.csv",
    "exp2_slopes_m_max": "exp2_slopes_m_max.csv",
    "exp2_slopes_per_m": "exp2_slopes_per_m.csv",
    "exp3_rank": "exp3_rank_mismatch.csv",
    "exp4_verdict": "exp4_verdict.csv",
    "exp5_crossover": "exp5_crossover.csv",
    "exp5_crossing": "exp5_crossing.csv",
    "exp6_rank_stats": "exp6_rank_stats.csv",
    "exp6_comparison": "exp6_comparison.csv",
    "exp7_verdict": "exp7_verdict.csv",
    "exp3_robust": "exp3_robust.csv",
}


def load_tables(tables_dir: Path) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for key, fname in _FILES.items():
        df = _load(tables_dir / fname)
        if df is not None:
            out[key] = df
    return out


def make_all(tables_dir: Path, fig_dir: Path) -> list[Path]:
    _setup()
    t = load_tables(tables_dir)
    if not t:
        LOG.error("未找到任何结果表，请先运行 scripts/run_experiment.py")
        return []
    made: list[Path] = []
    for fn in FIGURES:
        try:
            p = fn(t, fig_dir)
        except Exception as exc:                       # noqa: BLE001
            LOG.error("生成 %s 失败：%s", fn.__name__, exc)
            continue
        if p is not None:
            made.append(p)
    LOG.info("共生成 %d 张图", len(made))
    return made
