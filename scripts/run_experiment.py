"""实验运行入口。

用法：
    python scripts/run_experiment.py --exp exp1 --smoke
    python scripts/run_experiment.py --exp exp1
    python scripts/run_experiment.py --exp exp1 --config configs/exp1.yaml

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

import numpy as np  # noqa: E402

from dmdnoise.config import Config, config_fingerprint, save_fingerprint, validate  # noqa: E402
from dmdnoise.experiments import decision_notes, run_exp1  # noqa: E402

LOG = logging.getLogger("run_experiment")


def _make_progress(tag: str):
    state = {"t0": time.perf_counter()}

    def cb(i: int, n: int) -> None:
        if i == 0:
            state["t0"] = time.perf_counter()
            return
        el = time.perf_counter() - state["t0"]
        LOG.info("%s: %d/%d (%.1fs, 预计总 %.1fs)", tag, i, n, el, el * n / max(i, 1))

    return cb


def _run_exp1(cfg: Config, out: Path, *, smoke: bool) -> int:
    if smoke:
        channels = ("real",)
        snr_res = (0.0, -5.0)
        snr_up = (10.0,)
        j = 600
        m = 200
        LOG.warning("SMOKE 模式：仅验证流水线，数值不可用于结论")
    else:
        channels = ("real", "complex")
        snr_res = (0.0, -5.0, -10.0)
        snr_up = (20.0, 10.0, 5.0)
        j = cfg.grid.j_main
        m = 200

    LOG.info("实验一：channels=%s, A组=%s, B组=%s, m=%d, J=%d",
             channels, snr_res, snr_up, m, j)
    res = run_exp1(cfg, channels=channels, snr_resolvable=snr_res,
                   snr_upper=snr_up, m=m, j_total=j,
                   progress=_make_progress("exp1"))

    out.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if smoke else ""
    p_rows = out / f"exp1_bias_variance{suffix}.csv"
    p_paired = out / f"exp1_paired{suffix}.csv"
    p_notes = out / f"exp1_reportable{suffix}.csv"
    res.rows.to_csv(p_rows, index=False, encoding="utf-8")
    res.paired.to_csv(p_paired, index=False, encoding="utf-8")
    decision_notes(res).to_csv(p_notes, index=False, encoding="utf-8")

    meta = {
        "experiment": "exp1",
        "smoke": smoke,
        "fingerprint": res.fingerprint,
        **res.meta,
    }
    (out / f"exp1_meta{suffix}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    LOG.info("已写出：%s", p_rows.name)
    _print_headline(res)
    return 0


def _print_headline(res) -> None:
    df = decision_notes(res)
    if df.empty:
        LOG.warning("结果为空")
        return
    print("\n=== 可报告口径 ===", file=sys.stdout)
    with_ = df.to_string(index=False, max_colwidth=22)
    print(with_, file=sys.stdout)

    if not res.paired.empty:
        pr = res.paired[res.paired["stat"] == "std"]
        if not pr.empty:
            print("\n=== 方差代价（std_TDMD / std_DMD）===", file=sys.stdout)
            cols = ["channel", "snr_db", "mode", "std_tdmd_over_dmd",
                    "std_ci_lo", "std_ci_hi", "paired_effective"]
            print(pr[cols].to_string(index=False), file=sys.stdout)


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
        from dmdnoise.config import load_config

        cfg, results = load_config(args.config)
    else:
        cfg = Config()
        results = validate(cfg)

    out = Path(args.out) if args.out else ROOT / cfg.run.out_dir / "tables"

    LOG.info("配置指纹 %s", config_fingerprint(cfg))
    LOG.info("约束状态：通过=%s，警戒=%s",
             all(r.passed for r in results), [r.name for r in results if r.warning])
    save_fingerprint(cfg, results, out / "config_meta.json")

    LOG.info("BLAS 线程：OPENBLAS_NUM_THREADS=%s", os.environ["OPENBLAS_NUM_THREADS"])

    t0 = time.perf_counter()
    rc = _run_exp1(cfg, out, smoke=args.smoke)
    LOG.info("总耗时 %.1f s", time.perf_counter() - t0)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
