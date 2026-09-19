"""实验运行入口。

用法：
    python scripts/run_experiment.py --exp exp1 --smoke
    python scripts/run_experiment.py --exp exp1

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
    run_exp1,
)

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


def run_exp1_cli(cfg: Config, out: Path, *, smoke: bool) -> int:
    if smoke:
        slices = list(SMOKE_SLICES)
        j, m = 400, 200
        LOG.warning("SMOKE 模式：仅验证流水线，数值不可用于结论")
        n_slices = len(slices)
    else:
        slices = None
        j, m = cfg.grid.j_main, 200
        n_slices = len(__import__("dmdnoise.experiments", fromlist=["x"]).default_slices(cfg))

    LOG.info("实验一：%d 个切片，J=%d，m=%d", n_slices, j, m)

    res = run_exp1(cfg, slices=slices, m=m, j_total=j,
                   progress=make_progress("exp1"))

    out.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if smoke else ""
    res.rows.to_csv(out / f"exp1_bias_variance{suffix}.csv", index=False, encoding="utf-8")
    res.paired.to_csv(out / f"exp1_paired{suffix}.csv", index=False, encoding="utf-8")
    annotate_reportable(res).to_csv(out / f"exp1_reportable{suffix}.csv",
                                    index=False, encoding="utf-8")
    headline(res).to_csv(out / f"exp1_headline{suffix}.csv", index=False, encoding="utf-8")

    meta = {"experiment": "exp1", "smoke": smoke, "fingerprint": res.fingerprint, **res.meta}
    (out / f"exp1_meta{suffix}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    LOG.info("已写出 exp1_*.csv（%d 行明细）", len(res.rows))
    print_summary(res)
    return 0


def print_summary(res) -> None:
    df = headline(res)
    if df.empty:
        LOG.warning("结果为空")
        return
    print("\n=== 各网格点汇总（相对偏差）===", file=sys.stdout)
    print(df.to_string(index=False), file=sys.stdout)

    pr = res.paired
    if pr is not None and not pr.empty:
        std = pr[pr["stat"] == "std"]
        if not std.empty:
            print("\n=== 方差代价（std_TDMD / std_DMD，配对 bootstrap）===", file=sys.stdout)
            cols = ["channel", "axis", "snr_db", "n_axis", "amp_ratio", "mode",
                    "ratio", "ci_lo", "ci_hi", "paired_effective"]
            print(std[cols].to_string(index=False), file=sys.stdout)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DMD/TDMD 偏差-方差实验入口")
    ap.add_argument("--exp", default="exp1", choices=["exp1"])
    ap.add_argument("--config", default=None, help="YAML 配置路径；缺省用内置默认值")
    ap.add_argument("--out", default=None, help="输出目录；缺省 results/tables")
    ap.add_argument("--smoke", action="store_true", help="小规模冒烟运行")
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

    LOG.info("配置指纹 %s", config_fingerprint(cfg))
    LOG.info("约束：通过=%s，警戒=%s",
             all(r.passed for r in results), [r.name for r in results if r.warning])
    LOG.info("BLAS 线程 OPENBLAS_NUM_THREADS=%s", os.environ["OPENBLAS_NUM_THREADS"])
    save_fingerprint(cfg, results, out / "config_meta.json")

    t0 = time.perf_counter()
    rc = run_exp1_cli(cfg, out, smoke=args.smoke)
    LOG.info("总耗时 %.1f s", time.perf_counter() - t0)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
