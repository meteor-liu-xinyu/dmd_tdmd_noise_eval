"""实验运行入口。

用法：
    python scripts/run_experiment.py --exp exp1 [--smoke]
    python scripts/run_experiment.py --exp exp2 [--smoke]
    python scripts/run_experiment.py --exp exp3 [--smoke]
    python scripts/run_experiment.py --exp all   [--smoke]

⚠️ 必须在 import numpy 之前锁定 BLAS 线程数（见 ADR-010 与 algorithm-spec §12.1）。
"""

from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dmdnoise.config import (  # noqa: E402
    Config,
    config_fingerprint,
    load_config,
    save_fingerprint,
    validate,
)
from dmdnoise.experiments import (  # noqa: E402
    Slice,
    annotate_reportable,
    headline,
    judge_exp2,
    run_exp1,
    run_exp2,
    run_exp3,
)
from dmdnoise.experiments import (  # noqa: E402
    run_exp4,
    run_exp5,
    run_exp6,
    run_exp7,
    run_exp8,
    run_exp9,
    run_exp10,
    run_exp11,
    summary_exp11,
    _jsonable,
    summary_exp5,
    verdict_exp4,
    verdict_exp6,
    verdict_exp7,
    verdict_exp8,
    verdict_exp9,
    verdict_exp10,
)
from dmdnoise.report import make_all as make_figures  # noqa: E402

LOG = logging.getLogger("run_experiment")

SMOKE_SLICES = (
    Slice("complex", 10.0, 4, 1.0, label="n"),
    Slice("complex", 10.0, 8, 1.0, label="n"),
    Slice("complex", -5.0, 8, 1.0, label="snr"),
    Slice("hankel", 10.0, 8, 1.0, label="embed"),
)


def make_progress(tag: str):
    def cb(done: int, total: int, desc: str = "") -> None:
        LOG.info("%s: %d/%d %s", tag, done, total, desc)

    return cb


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- exp1
def run_exp1_cli(cfg: Config, out: Path, *, smoke: bool) -> int:
    from dmdnoise.experiments import default_slices

    if smoke:
        slices, j, m = list(SMOKE_SLICES), 400, 200
        LOG.warning("SMOKE 模式：仅验证流水线，数值不可用于结论")
    else:
        slices, j, m = None, cfg.grid.j_main, 200

    n_slices = len(slices) if slices is not None else len(default_slices(cfg))
    LOG.info("实验一：%d 个切片，J=%d，m=%d", n_slices, j, m)

    res = run_exp1(cfg, slices=slices, m=m, j_total=j, progress=make_progress("exp1"))
    suffix = "_smoke" if smoke else ""
    out.mkdir(parents=True, exist_ok=True)
    res.rows.to_csv(out / f"exp1_bias_variance{suffix}.csv", index=False, encoding="utf-8")
    res.paired.to_csv(out / f"exp1_paired{suffix}.csv", index=False, encoding="utf-8")
    annotate_reportable(res).to_csv(out / f"exp1_reportable{suffix}.csv", index=False,
                                    encoding="utf-8")
    headline(res).to_csv(out / f"exp1_headline{suffix}.csv", index=False, encoding="utf-8")
    write_json(out / f"exp1_meta{suffix}.json",
               {"experiment": "exp1", "smoke": smoke,
                "fingerprint": res.fingerprint, **res.meta})
    LOG.info("已写出 exp1_*.csv（%d 行）", len(res.rows))
    print_summary_exp1(res)
    return 0


def print_summary_exp1(res) -> None:
    df = headline(res)
    if df.empty:
        LOG.warning("结果为空")
        return
    print("\n=== 实验一 · 各网格点汇总（相对偏差）===", file=sys.stdout)
    print(df.to_string(index=False), file=sys.stdout)
    if res.paired is not None and not res.paired.empty:
        std = res.paired[res.paired["stat"] == "std"]
        if not std.empty:
            print("\n=== 实验一 · 方差代价（std_TDMD / std_DMD）===", file=sys.stdout)
            cols = ["channel", "axis", "snr_db", "n_axis", "amp_ratio", "mode",
                    "ratio", "ci_lo", "ci_hi", "paired_effective"]
            print(std[cols].to_string(index=False), file=sys.stdout)


# --------------------------------------------------------------------------- exp2
def run_exp2_cli(cfg: Config, out: Path, *, smoke: bool, snr_norm: str = "both") -> int:
    """跑实验二。snr_norm="both" 时两种 SNR 归一化口径都跑并给出对比。"""
    from dmdnoise.experiments.exp2_sample_scan import compare_normalizations

    if smoke:
        j, kw = 400, {"m_values": (50, 200, 500)}
        LOG.warning("SMOKE 模式：仅验证流水线，数值不可用于结论")
    else:
        j, kw = 10000, {}

    norms = ["m_max", "per_m"] if snr_norm == "both" else [snr_norm]
    suffix = "_smoke" if smoke else ""
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for norm in norms:
        LOG.info("--- exp2 口径 %s ---", norm)
        res = run_exp2(cfg, j_total=j, snr_normalization=norm,
                       progress=make_progress("exp2[" + norm + "]"), **kw)
        results[norm] = res
        tag = suffix if len(norms) == 1 else "_" + norm + suffix
        res.table.to_csv(out / ("exp2_sample_scan" + tag + ".csv"), index=False,
                         encoding="utf-8")
        res.slopes.to_csv(out / ("exp2_slopes" + tag + ".csv"), index=False,
                          encoding="utf-8")
        judge_exp2(res).to_csv(out / ("exp2_judgement" + tag + ".csv"), index=False,
                               encoding="utf-8")
        write_json(out / ("exp2_meta" + tag + ".json"),
                   {"experiment": "exp2", "smoke": smoke,
                    "fingerprint": res.fingerprint, **res.meta})
        LOG.info("已写出 exp2_*%s.csv（%d 行明细）", tag, len(res.table))
        print()
        print("=== 实验二 · 斜率与判定（口径 " + norm + "）===")
        print(judge_exp2(res).to_string(index=False))

    if len(norms) == 2:
        cmp = compare_normalizations(results["m_max"], results["per_m"])
        cmp.to_csv(out / ("exp2_norm_compare" + suffix + ".csv"), index=False,
                   encoding="utf-8")
        print()
        print("=== 实验二 · SNR 归一化口径对比（分离 m 效应与有效 SNR 效应）===")
        print(cmp.to_string(index=False))
    return 0



# --------------------------------------------------------------------------- exp3
def run_exp3_cli(cfg: Config, out: Path, *, smoke: bool) -> int:
    if smoke:
        LOG.warning("SMOKE 模式：仅验证流水线，数值不可用于结论")
    res = run_exp3(cfg, j_total=300 if smoke else 2000,
                   progress=make_progress("exp3"))
    suffix = "_smoke" if smoke else ""
    out.mkdir(parents=True, exist_ok=True)
    res.table.to_csv(out / f"exp3_rank_sensitivity{suffix}.csv", index=False,
                     encoding="utf-8")
    res.rank_table.to_csv(out / f"exp3_rank_mismatch{suffix}.csv", index=False,
                          encoding="utf-8")
    res.robust.to_csv(out / f"exp3_robust{suffix}.csv", index=False, encoding="utf-8")
    res.adaptive.to_csv(out / f"exp3_adaptive_sigma{suffix}.csv", index=False,
                        encoding="utf-8")
    write_json(out / f"exp3_meta{suffix}.json",
               {"experiment": "exp3", "smoke": smoke,
                "fingerprint": res.fingerprint, **res.meta})
    LOG.info("已写出 exp3_*.csv（%d 行明细）", len(res.table))
    print("\n=== 实验三 · 秩误判率（k = sigma_hat / sigma_true）===", file=sys.stdout)
    print(res.rank_table.to_string(index=False), file=sys.stdout)
    print("\n=== 实验三 · 鲁棒区间 ===", file=sys.stdout)
    print(res.robust.to_string(index=False), file=sys.stdout)
    print("\n=== 实验三 · 数据驱动 sigma_hat 表现 ===", file=sys.stdout)
    print(res.adaptive.to_string(index=False), file=sys.stdout)
    return 0


def run_exp4_cli(cfg: Config, out: Path, *, smoke: bool, **_) -> int:
    """实验四：参数稳健性复核（模态间隔 x 窗内衰减）。"""
    if smoke:
        from dmdnoise.experiments import RobustPoint

        pts = [RobustPoint("complex", s, d, 8, 10.0)
               for s in (1.25, 1.83) for d in (0.90, 0.50)]
        j = 400
        LOG.warning("SMOKE 模式：仅验证流水线，数值不可用于结论")
    else:
        pts, j = None, cfg.grid.j_main

    res = run_exp4(cfg, points=pts, j_total=j, progress=make_progress("exp4"))
    suffix = "_smoke" if smoke else ""
    out.mkdir(parents=True, exist_ok=True)
    res.table.to_csv(out / ("exp4_robustness" + suffix + ".csv"), index=False,
                     encoding="utf-8")
    res.paired.to_csv(out / ("exp4_paired" + suffix + ".csv"), index=False,
                      encoding="utf-8")
    res.constraints.to_csv(out / ("exp4_constraints" + suffix + ".csv"), index=False,
                           encoding="utf-8")
    v = verdict_exp4(res)
    v.to_csv(out / ("exp4_verdict" + suffix + ".csv"), index=False, encoding="utf-8")
    write_json(out / ("exp4_meta" + suffix + ".json"),
               {"experiment": "exp4", "smoke": smoke,
                "fingerprint": res.fingerprint, **res.meta})
    LOG.info("已写出 exp4_*.csv（%d 行明细）", len(res.table))
    print()
    print("=== 实验四 · 参数稳健性判定（模态间隔 x 窗内衰减）===")
    print(v.to_string(index=False))
    return 0


def run_exp5_cli(cfg: Config, out: Path, *, smoke: bool, **_) -> int:
    """实验五：交叉点定位（降偏比 rho 穿越 1 的位置）。"""
    if smoke:
        combos = (("real", 16),)
        seps = (1.70, 1.80, 1.90)
        j = 800
        LOG.warning("SMOKE 模式：仅验证流水线，数值不可用于结论")
    else:
        combos, seps = None, None
        j = 3000

    kw = {} if seps is None else {"combos": combos, "separations": seps}
    res = run_exp5(cfg, j_total=j, progress=make_progress("exp5"), **kw)
    suffix = "_smoke" if smoke else ""
    out.mkdir(parents=True, exist_ok=True)
    res.table.to_csv(out / ("exp5_crossover" + suffix + ".csv"), index=False,
                     encoding="utf-8")
    res.crossing.to_csv(out / ("exp5_crossing" + suffix + ".csv"), index=False,
                        encoding="utf-8")
    write_json(out / ("exp5_meta" + suffix + ".json"),
               {"experiment": "exp5", "smoke": smoke,
                "fingerprint": res.fingerprint, **res.meta})
    LOG.info("已写出 exp5_*.csv（%d 行明细）", len(res.table))
    print()
    print("=== 实验五 · 交叉点定位 ===")
    print(summary_exp5(res).to_string(index=False))
    return 0


def run_exp6_cli(cfg: Config, out: Path, *, smoke: bool, **_) -> int:
    """实验六：端到端评估（噪声水平未知 -> 秩估计 -> 频率估计）。"""
    j = 300 if smoke else 2000
    if smoke:
        LOG.warning("SMOKE 模式：仅验证流水线，数值不可用于结论")
    res = run_exp6(cfg, j_total=j, progress=make_progress("exp6"))
    suffix = "_smoke" if smoke else ""
    out.mkdir(parents=True, exist_ok=True)
    for name, df in (("exp6_rank_stats", res.ranks),
                     ("exp6_table", res.table),
                     ("exp6_comparison", res.comparison),
                     ("exp6_verdict", verdict_exp6(res))):
        df.to_csv(out / (name + suffix + ".csv"), index=False, encoding="utf-8")
    write_json(out / ("exp6_meta" + suffix + ".json"),
               {"experiment": "exp6", "smoke": smoke,
                "fingerprint": res.fingerprint, **res.meta})
    LOG.info("已写出 exp6_*.csv（%d 行明细）", len(res.table))
    print()
    print("=== 实验六 · 端到端裁决 ===")
    print(verdict_exp6(res).to_string(index=False))
    print()
    print("=== 实验六 · 数据驱动秩 vs oracle 秩（配对 bootstrap）===")
    if not res.comparison.empty:
        c = res.comparison[res.comparison["method"] == "tdmd"]
        cols = ["channel", "snr_db", "mode", "strategy", "mse_ratio",
                "mse_lo", "mse_hi", "退化显著"]
        print(c[cols].to_string(index=False))
    return 0


def run_exp7_cli(cfg: Config, out: Path, *, smoke: bool, **_) -> int:
    """实验七：秩估计方案的补救效果（规避虚假不动点）。"""
    j = 300 if smoke else 2000
    if smoke:
        LOG.warning("SMOKE 模式：仅验证流水线，数值不可用于结论")
    res = run_exp7(cfg, j_total=j, progress=make_progress("exp7"))
    suffix = "_smoke" if smoke else ""
    out.mkdir(parents=True, exist_ok=True)
    for name, df in (("exp7_rank_stats", res.ranks),
                     ("exp7_table", res.table),
                     ("exp7_comparison", res.comparison),
                     ("exp7_verdict", verdict_exp7(res))):
        df.to_csv(out / (name + suffix + ".csv"), index=False, encoding="utf-8")
    write_json(out / ("exp7_meta" + suffix + ".json"),
               {"experiment": "exp7", "smoke": smoke,
                "fingerprint": res.fingerprint, **res.meta})
    LOG.info("已写出 exp7_*.csv（%d 行明细）", len(res.table))
    print()
    print("=== 实验七 · 各判据 x 方案的补救效果 ===")
    print(verdict_exp7(res).to_string(index=False))
    return 0


def run_exp8_cli(cfg: Config, out: Path, *, smoke: bool, **_) -> int:
    """实验八：交叉点的实现间散布（判定"随 n 非单调"是否为真）。"""
    if smoke:
        kw = {"n_values": (16,), "k_configs": 3, "j_total": 300}
        LOG.warning("SMOKE 模式：仅验证流水线，数值不可用于结论")
    else:
        kw = {"j_total": 800}
    res = run_exp8(cfg, progress=make_progress("exp8"), **kw)
    suffix = "_smoke" if smoke else ""
    out.mkdir(parents=True, exist_ok=True)
    for name, df in (("exp8_table", res.table), ("exp8_crossings", res.crossings),
                     ("exp8_spread", res.spread),
                     ("exp8_verdict", verdict_exp8(res))):
        df.to_csv(out / (name + suffix + ".csv"), index=False, encoding="utf-8")
    write_json(out / ("exp8_meta" + suffix + ".json"),
               {"experiment": "exp8", "smoke": smoke,
                "fingerprint": res.fingerprint, **res.meta})
    LOG.info("已写出 exp8_*.csv（%d 行明细）", len(res.table))
    print()
    print("=== 实验八 · 交叉点分布的 n 间对比 ===")
    print(verdict_exp8(res).to_string(index=False))
    print()
    print("=== 逐 (n, 模态) 分布统计 ===")
    print(res.spread.to_string(index=False))
    return 0


def run_exp9_cli(cfg: Config, out: Path, *, smoke: bool, **_) -> int:
    """实验九：有色噪声下的稳健性。"""
    j = 400 if smoke else 2000
    if smoke:
        LOG.warning("SMOKE 模式：仅验证流水线，数值不可用于结论")
    res = run_exp9(cfg, j_total=j, progress=make_progress("exp9"))
    suffix = "_smoke" if smoke else ""
    out.mkdir(parents=True, exist_ok=True)
    for name, df in (("exp9_table", res.table), ("exp9_paired", res.paired),
                     ("exp9_ranks", res.ranks), ("exp9_verdict", verdict_exp9(res))):
        df.to_csv(out / (name + suffix + ".csv"), index=False, encoding="utf-8")
    write_json(out / ("exp9_meta" + suffix + ".json"),
               {"experiment": "exp9", "smoke": smoke,
                "fingerprint": res.fingerprint, **res.meta})
    LOG.info("已写出 exp9_*.csv（%d 行明细）", len(res.table))
    print()
    print("=== 实验九 · 有色噪声下的裁决 ===")
    print(verdict_exp9(res).to_string(index=False))
    return 0


def run_exp10_cli(cfg: Config, out: Path, *, smoke: bool, **_) -> int:
    """实验十：有色噪声下的秩判据 —— 预白化的补救效果。"""
    j = 300 if smoke else 1500
    if smoke:
        LOG.warning("SMOKE 模式：仅验证流水线，数值不可用于结论")
    res = run_exp10(cfg, j_total=j, progress=make_progress("exp10"))
    suffix = "_smoke" if smoke else ""
    out.mkdir(parents=True, exist_ok=True)
    res.table.to_csv(out / ("exp10_table" + suffix + ".csv"), index=False,
                     encoding="utf-8")
    res.summary.to_csv(out / ("exp10_summary" + suffix + ".csv"), index=False,
                       encoding="utf-8")
    verdict_exp10(res).to_csv(out / ("exp10_verdict" + suffix + ".csv"), index=False,
                              encoding="utf-8")
    write_json(out / ("exp10_meta" + suffix + ".json"),
               {"experiment": "exp10", "smoke": smoke,
                "fingerprint": res.fingerprint, **res.meta})
    LOG.info("已写出 exp10_*.csv（%d 行明细）", len(res.table))
    print()
    print("=== 实验十 · 预白化的补救效果（GD 判据）===")
    print(verdict_exp10(res).to_string(index=False))
    return 0


def run_exp11_cli(cfg: Config, out: Path, *, smoke: bool, **_) -> int:
    """实验十一：原文 Figure 1 复现（外部一致性验证）。"""
    if smoke:
        kw = {"ms": (100,), "sigma2_grid": (0.05, 0.8, 3.2), "j_total": 120,
              "j_controls": 40, "j_samples": 50,
              "noise_modes": ("trajectory", "independent")}
        LOG.warning("SMOKE 模式：仅验证流水线，数值不可用于结论")
    else:
        kw = {"j_total": 1500, "j_controls": 200, "j_samples": 300,
              "noise_modes": ("trajectory", "independent")}
    res = run_exp11(cfg, progress=make_progress("exp11"), **kw)
    suffix = "_smoke" if smoke else ""
    out.mkdir(parents=True, exist_ok=True)
    for name, df in (("exp11_table", res.table),
                     ("exp11_controls", res.controls),
                     ("exp11_sigma_scan", res.sigma_scan),
                     ("exp11_samples", res.samples),
                     ("exp11_verdict", res.verdict)):
        df.to_csv(out / (name + suffix + ".csv"), index=False, encoding="utf-8")
    write_json(out / ("exp11_meta" + suffix + ".json"),
               {"smoke": smoke, "fingerprint": res.fingerprint,
                **{k: _jsonable(v) for k, v in res.meta.items()}})
    LOG.info("已写出 exp11_*.csv（原文算例 %d 行、对照 %d 行、扫描 %d 行）",
             len(res.table), len(res.controls), len(res.sigma_scan))
    print()
    print("=== 实验十一 · 原文 Figure 1 复现裁决 ===")
    cols = ["来源", "噪声口径", "m", "sigma2", "DMD偏差", "TDMD偏差", "偏差比",
            "DMD可分辨", "TDMD可分辨", "方差比_TDMD_over_DMD", "MAE差",
            "MAE差下界", "MAE差上界", "TDMD更接近真值", "公共随机数生效", "无噪退化门"]
    print(res.verdict[[c for c in cols if c in res.verdict.columns]].to_string(index=False))
    print()
    print("=== 交付代码 vs 原文公式 / 负对照 ===")
    c = res.controls[res.controls["method"].str.startswith(("ref_vs", "diff_vs"))]
    print(c[["m", "method", "bias_complex_abs"]].to_string(index=False))
    print()
    print("=== 偏差标度指数（预期 DMD~1、TDMD~2；仅用偏差可分辨的点拟合）===")
    print(f"{'判据':>8} {'模态':>4} {'a':>7} {'用点数':>6} {'可分辨':>6} {'总点数':>6}")
    for sl in res.meta["slopes"]:
        print(f"{sl['method']:>8} {sl['mode']:>4} "
              f"{sl['slope_log_bias_vs_log_sigma2']:>7.2f} {sl['n_sigma2']:>6} "
              f"{sl.get('n_resolvable', 0):>6} {sl.get('n_sigma2_total', 0):>6}")
    print()
    print(summary_exp11(res))
    return 0


def run_figures_cli(cfg: Config, out: Path, *, smoke: bool, **_) -> int:
    """由 results/tables 下的 CSV 生成 fig1-fig10。只读 CSV，不重跑实验。"""
    fig_dir = ROOT / cfg.run.out_dir / "figures"
    made = make_figures(out, fig_dir)
    if not made:
        LOG.error("未生成任何图；请先运行 --exp exp1 / exp2 / exp3")
        return 1
    print()
    print(f"已生成 {len(made)} 张图 -> {fig_dir}")
    for p in made:
        print("  " + p.name)
    return 0


RUNNERS = {"exp1": run_exp1_cli, "exp2": run_exp2_cli, "exp3": run_exp3_cli,
           "exp4": run_exp4_cli, "exp5": run_exp5_cli, "exp6": run_exp6_cli,
           "exp7": run_exp7_cli, "exp8": run_exp8_cli, "exp9": run_exp9_cli, "exp10": run_exp10_cli,
           "exp11": run_exp11_cli,
           "figures": run_figures_cli}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DMD/TDMD 偏差-方差实验入口")
    ap.add_argument("--exp", default="exp1",
                    choices=["exp1", "exp2", "exp3", "exp4", "exp5", "exp6",
                             "exp7", "exp8", "exp9", "exp10", "exp11", "figures", "all"])
    ap.add_argument("--config", default=None, help="YAML 配置路径；缺省用内置默认值")
    ap.add_argument("--out", default=None, help="输出目录；缺省 results/tables")
    ap.add_argument("--smoke", action="store_true", help="小规模冒烟运行")
    ap.add_argument("--snr-norm", default="both", choices=["m_max", "per_m", "both"],
                    help="exp2 的 SNR 归一化口径；默认两种都跑并对比")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.config:
        cfg, results = load_config(args.config)
    else:
        cfg = Config()
        results = validate(cfg)

    out = Path(args.out) if args.out else ROOT / cfg.run.out_dir / "tables"
    if args.smoke and args.out is None:
        # 冒烟产物与正式结果**物理分离**：其数值由 --smoke 明确声明不可用于结论，
        # 若与正式结果同目录混放（仅差后缀），浏览时极易误取。
        out = out.parent / "smoke"
    LOG.info("配置指纹 %s", config_fingerprint(cfg))
    LOG.info("约束：通过=%s，警戒=%s",
             all(r.passed for r in results), [r.name for r in results if r.warning])
    LOG.info("BLAS 线程 OPENBLAS_NUM_THREADS=%s", os.environ["OPENBLAS_NUM_THREADS"])
    save_fingerprint(cfg, results, out / "config_meta.json")

    targets = (["exp1", "exp2", "exp3", "exp4", "exp5", "exp6", "exp7",
                "exp8", "exp9", "exp10", "exp11", "figures"] if args.exp == "all"
               else [args.exp])
    rc = 0
    for name in targets:
        t0 = time.perf_counter()
        LOG.info("===== 开始 %s =====", name)
        kw = {"snr_norm": args.snr_norm} if name == "exp2" else {}
        rc |= RUNNERS[name](cfg, out, smoke=args.smoke, **kw)
        LOG.info("===== %s 完成，耗时 %.1f s =====", name, time.perf_counter() - t0)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
