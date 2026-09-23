"""核对文档里的数字与仓库实际产物是否一致。

两类数字都核
------------

**① 规模数字** —— 图表张数、测试项数（含逐文件分解）、代码行数、数据行数、
ADR 编号范围、配置指纹、主结果表个数。它们分散在多处、由不同轮次的改动引入，
**只有在人工复核时才会被发现**。本项目已发生过两次同类事故：

- 完整 review 时手工查出 4 处不一致（其中 §2.13 的「9/12」是真错）；
- 新增 exp11 后，报告页头的 ``fig1–fig15``、``ADR-001 ~ ADR-018``、``187 项``、
  ``2 040 行``、``1 170 行`` 同时失效（实际为 16 张 / 019 / 210 项 / 2 425 行 / 1 190 行），
  而报告 §6 的测试分解里**整条 `test_paper_replication` 23 项缺失**。

**② 结论性数值** —— 从 ``results/tables/*.csv`` **重算**报告中的关键结论数字
（偏差降幅、`std` 比区间、交叉点、秩判对率、端到端 MSE 比、实现间散布极差…）
后与报告比对。这一类比规模数字更重要：规模数字错了读者看得出来，
而**结论数字错了读不出来** —— §2.13 的「9/12」正是这一类。

设计要点
--------

**找不到声明时报 WARN，而不是静默通过。** 如果文档措辞改变导致正则失配，
脚本必须让人看见 —— 否则会制造"检查通过"的假象，比不检查更危险。

**同一数字的所有出现都要检查。** 报告常把同一个数字写进摘要、正文与表格；
只查第一处（或用 ``re.search``）会让其余的不一致漏网 —— 已用注入错误验证过。

**必须排除「历史记录区」（``## 附录``及之后）。** 该区记录的是**当时的数字**
（如"§6 称 1 026 行 → 改为 1 170 行"），必然与现值不同；把它纳入核对会误判。

用法::

    python scripts/check_report_consistency.py           # 完整检查
    python scripts/check_report_consistency.py --quiet   # 只输出问题
    python scripts/check_report_consistency.py --no-pytest   # 跳过 pytest 收集（更快）

依赖 ``pip install pandas``（数值重算用）；缺结果 CSV 时自动跳过数值部分。

退出码：0 = 全部一致，1 = 存在不一致或未能核对。
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------- 运行环境
# BLAS 单线程：本项目约定，小矩阵上多线程开销远大于计算（见 MEMORY.md）。
# 必须在 import numpy 之前设置。
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
REPORT = DOCS / "final-report.md"
README = ROOT / "README.md"
PLAN = DOCS / "implementation-plan.md"
FIGDIR = ROOT / "results" / "figures"
TABLEDIR = ROOT / "results" / "tables"

#: 报告 §6 声明为「主结果表」的 CSV。**新增实验时必须同步这里**，
#: 否则本脚本会漏检该实验的数据行数（曾漏掉 exp11 的 20 行）。
MAIN_TABLES = (
    "exp1_bias_variance.csv", "exp2_sample_scan_m_max.csv",
    "exp2_sample_scan_per_m.csv", "exp3_rank_sensitivity.csv",
    "exp3_rank_mismatch.csv", "exp3_robust.csv", "exp4_robustness.csv",
    "exp5_crossover.csv", "exp6_table.csv", "exp7_table.csv",
    "exp8_table.csv", "exp9_table.csv", "exp10_table.csv", "exp11_table.csv",
)


@dataclass
class Check:
    """一条「文档声明 vs 实测」的对照。"""

    name: str
    claimed: str
    actual: str
    ok: bool | None          # None = 未在文档中找到声明（格式可能已变）
    where: str = ""
    numeric: bool = True     # 展示时是否把 "6 727" 归一化显示为 6727

    @property
    def status(self) -> str:
        return {True: "OK  ", False: "FAIL", None: "WARN"}[self.ok]

    def show_claimed(self) -> str:
        if self.numeric and self.claimed not in ("-", ""):
            try:
                return str(_num(self.claimed))
            except ValueError:
                pass
        return self.claimed


def _num(s: str) -> int:
    """把 ``6 727`` / ``6,727`` 这类带分隔符的数字转为 int。"""
    return int(re.sub(r"[^0-9]", "", s))


def _loc(text: str, pos: int) -> str:
    """把字符偏移转成行号。"""
    return f"L{text[:pos].count(chr(10)) + 1}"


# ---------------------------------------------------------------- 实测
def measure(*, use_pytest: bool = True) -> dict[str, object]:
    """从仓库实际产物测量各项规模数字。"""
    f: dict[str, object] = {}

    # ---- 图表
    pngs = sorted(FIGDIR.glob("fig*.png"))
    f["n_png"] = len(pngs)
    f["max_fig"] = max(
        (int(m.group(1)) for p in pngs if (m := re.match(r"fig(\d+)", p.name))),
        default=0)

    # ---- 代码行数
    def lines(pattern: str) -> int:
        return sum(len(p.read_text(encoding="utf-8").splitlines())
                   for p in ROOT.glob(pattern))

    f["src_lines"] = lines("src/**/*.py")
    f["tests_lines"] = lines("tests/**/*.py")

    # ---- 测试项数
    if use_pytest:
        out = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q"],
            cwd=str(ROOT), capture_output=True, text=True,
            encoding="utf-8", errors="replace")
        by_file = {m.group(1): int(m.group(2))
                   for m in re.finditer(r"^(tests/[\w/]+\.py): (\d+)$",
                                        out.stdout, re.M)}
        f["pytest_by_file"] = by_file
        f["pytest_total"] = sum(by_file.values())
    else:
        f["pytest_by_file"] = {}
        f["pytest_total"] = None

    # ---- ADR 编号
    if PLAN.exists():
        adrs = re.findall(r"^### ADR-(\d+)", PLAN.read_text(encoding="utf-8"), re.M)
        f["max_adr"] = max((int(a) for a in adrs), default=0)
        f["n_adr"] = len(adrs)

    # ---- 主结果表行数
    import csv
    total = 0
    for name in MAIN_TABLES:
        p = TABLEDIR / name
        if p.exists():
            with p.open(encoding="utf-8", newline="") as fh:
                total += max(sum(1 for _ in csv.reader(fh)) - 1, 0)
    f["main_rows"] = total

    # ---- 配置指纹
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from dmdnoise.config import Config, config_fingerprint
        f["fingerprint"] = config_fingerprint(Config())
    except Exception:
        f["fingerprint"] = None
    return f


#: 历史记录区的起始标题。该区**记录的是当时的数字**（"§6 称 X → 改为 Y"），
#: 必然与现值不同，故核对时必须排除 —— 否则会把正确的历史记录判为不一致。
HISTORY_HEADING = r"^## 附录"


def _mask_history(text: str) -> str:
    """把「历史记录」区替换成**等量换行**后返回。

    只替换成换行（而非删除）是为了**保持行号不变** —— ``_loc()`` 报告的行号
    是基于全文的，删掉一段会让其后所有位置的行号整体偏移。
    """
    m = re.search(HISTORY_HEADING, text, re.M)
    if not m:
        return text
    head, tail = text[:m.start()], text[m.start():]
    return head + "\n" * tail.count("\n")


# ---------------------------------------------------------------- 检查
def check_doc(doc: Path, f: dict[str, object], *, detailed: bool) -> list[Check]:
    """对单个文档做**该文档应有的**对照检查。

    ``detailed=True`` 用于 ``final-report.md``（含交付物清单的完整声明）；
    ``README.md`` 只声明了其中一部分，故传 ``False`` —— 否则会为它没有的那些项
    刷出一片 WARN，把真实问题淹没在噪声里。
    """
    if not doc.exists():
        return [Check(f"{doc.name}", "-", "文档不存在", False)]
    text = _mask_history(doc.read_text(encoding="utf-8"))
    out: list[Check] = []

    def add(name: str, pattern: str, actual: object, cast=_num,
            numeric: bool = True) -> None:
        """在文档里找 ``pattern`` 的捕获组，与 ``actual`` 比对。

        ⚠️ **必须检查所有出现，而不是第一处**：同一个规模数字常在多处出现
        （如「210 项」在报告与 README 各一处），只查第一处会让其余的不一致漏网。
        数值版同理 —— 自检时注入错误只改到了第二处，正是因为只看第一处才漏检。
        """
        hits = list(re.finditer(pattern, text))
        if not hits:
            out.append(Check(name, "-", str(actual), None,
                             f"{doc.name}（未找到声明）", numeric))
            return
        bad = 0
        for m in hits:
            try:
                if cast(m.group(1)) != actual:
                    bad += 1
            except (TypeError, ValueError):
                bad += 1
        out.append(Check(name, hits[0].group(1).strip(), str(actual), bad == 0,
                         f"{doc.name}:{_loc(text, hits[0].start())}"
                         + (f"，其中 {bad} 处不符" if bad else ""), numeric))

    # ---- 两份文档都声明的项
    add("图表最大编号", r"fig1[–\-]fig(\d+)", f["max_fig"])
    add("ADR 最大编号", r"ADR-001\s*~\s*\*{0,2}ADR-0(\d+)", f["max_adr"])
    add("测试项数", r"(\d+)\s*项全部通过", f["pytest_total"])

    if not detailed:
        # README 的代码规模写法与报告不同（总数 + src/tests 分解）
        add("代码总行数", r"实际代码规模\*{0,2}[^\n]*?\*\*([\d\s]+)\s*行",
            int(f["src_lines"]) + int(f["tests_lines"]))
        add("代码行数 src", r"`src/`\s*([\d\s]+)\s*\+", f["src_lines"])
        add("代码行数 tests", r"`tests/`\s*([\d\s]+)\s*）", f["tests_lines"])
        return out

    # ---- 仅报告：交付物清单的完整声明
    add("图表张数", r"共 \*{0,2}(\d+) 张", f["n_png"])
    add("代码行数 src", r"`src/dmdnoise/`[^\n]*?\*\*([\d\s]+)\s*行\*\*", f["src_lines"])
    add("代码行数 tests", r"`tests/`（\*\*([\d\s]+)\s*行", f["tests_lines"])
    add("主结果表个数", r"(\d+)\s*个主结果表共", len(MAIN_TABLES))
    add("主结果表行数", r"个主结果表共\s*\*{0,2}([\d\s]+)\s*行", f["main_rows"])
    add("配置指纹", r"配置指纹 `([0-9a-f]{16})`", f["fingerprint"],
        cast=str, numeric=False)

    # ---- 测试文件分解：逐文件核对
    # 完整 review 时漏掉过整条 `test_paper_replication`（23 项），故逐文件比对
    for path, n in sorted((f.get("pytest_by_file") or {}).items()):
        stem = Path(path).stem
        add(f"分解 {stem}", rf"`{re.escape(stem)}`\s+(\d+)", n)

    return out


# ---------------------------------------------------------------- 数值核对
#
# 上一节核对的只是**规模数字**（图表数、测试项数、行数）。
# 报告里还有一类更重要的数字：**结论性数值**（偏差降幅、降偏比、判对率…），
# 它们是从 CSV 算出来的。§2.13 的「9/12」就是这类错误 —— 手写了但没人守。
#
# 做法：每个数值给出「从哪个 CSV 怎么算」的重算函数 + 「在报告里长什么样」的正则。
# 正则失配 -> WARN（措辞可能改了），数值不符 -> FAIL。

def _read(name: str):
    import pandas as pd
    return pd.read_csv(TABLEDIR / name)


def compute_numbers() -> dict[str, object]:
    """从 CSV 重算报告中的关键结论性数值。"""
    import numpy as np
    out: dict[str, object] = {}

    # ---- exp1：n 由 4 增到 16 的偏差降幅（复值 / n 轴 / SNR=10dB / 标准 DMD）
    d = _read("exp1_bias_variance.csv")
    g = d[(d.channel == "complex") & (d.axis == "n") & (d.snr_db == 10.0)
          & (d.method == "dmd")]
    out["exp1_fold"] = (float(g[g.n_axis == 4]["bias"].abs().max())
                        / float(g[g.n_axis == 16]["bias"].abs().max()))

    # ---- exp1：`std` 比的可用区间（配对失败率 ≤ 20% 的点）
    p = _read("exp1_paired.csv")
    pf = d.groupby(["channel", "axis", "snr_db", "n_axis", "amp_ratio"])[
        "pair_fail_rate"].max().to_dict()
    s = p[p.stat == "std"].copy()
    s["pair_fail"] = [pf.get((r.channel, r.axis, r.snr_db, r.n_axis, r.amp_ratio),
                             np.nan) for r in s.itertuples()]
    usable = s[s.pair_fail <= 0.20]["ratio"]
    out["exp1_std_min"] = float(usable.min())
    out["exp1_std_max"] = float(usable.max())
    # 全 58 个点中 CI 下界 > 1 的个数（报告称 0）
    out["exp1_std_ci_above1"] = int((s.ci_lo > 1).sum())

    # ---- exp3：σ̂/σ 的校正后均值（复值）
    a = _read("exp3_adaptive_sigma.csv")
    out["exp3_sigma_complex"] = float(
        a[a.channel == "complex"]["sigma_hat_over_true_mean"].iloc[0])

    # ---- exp4：f2/f1 由 1.25 拉到 1.83 时 DMD 偏差的降幅（复值）
    #
    # ⚠️ 口径：**按窗内衰减分组各求比值、再取中位数**。模态间隔与窗内衰减是两个
    # 独立轴（实验四的结论正是"间隔是强杠杆、衰减几乎无影响"），跨衰减取 max/min
    # 会把两个轴混在一起 —— 那样得 3.91，与报告的 3.7 不符，而中位数正好是 3.70。
    e4 = _read("exp4_verdict.csv")
    e4c = e4[(e4.channel == "complex") & e4["可用"].astype(bool)]
    lo = (e4c[np.isclose(e4c.f2_over_f1, e4c.f2_over_f1.min())]
          .set_index("window_decay")["DMD_偏差"])
    hi = (e4c[np.isclose(e4c.f2_over_f1, e4c.f2_over_f1.max())]
          .set_index("window_decay")["DMD_偏差"])
    out["exp4_fold"] = float((lo / hi).median())
    # 端点本身也要守：报告曾把「4.30e-3 → 1.10e-3」并列，但那两个值来自**不同的
    # 窗内衰减曲线**（4.30e-3 是 d=0.35、1.10e-3 是 d=0.50/0.70 附近），
    # 相除得 3.91，与同一句里的「3.7 倍」不符。改用**同口径的中位端点**。
    # 以 1e-3 为单位存放，便于与报告的 `4.05e-3` 写法直接比较。
    out["exp4_lo_x1e3"] = float(lo.median()) * 1e3
    out["exp4_hi_x1e3"] = float(hi.median()) * 1e3

    # ---- exp5：实值 n=16 处两个模态的交叉点
    e5 = _read("exp5_crossing.csv")
    e5r = e5[(e5.channel == "real") & (e5.n == 16) & e5.crossing.notna()]
    out["exp5_cross_mode1"] = float(e5r[e5r["mode"] == 1]["crossing"].iloc[0])
    out["exp5_cross_mode2"] = float(e5r[e5r["mode"] == 2]["crossing"].iloc[0])

    # ---- exp6：gavish_donoho 的端到端 MSE 比（应恒为 1）
    c = _read("exp6_comparison.csv")
    gd = c[c.strategy == "gavish_donoho"]
    out["exp6_gd_ratio_max"] = float(gd.mse_ratio.max())

    # ---- exp7：marchenko_pastur 的秩判对率（校正后）
    v = _read("exp7_verdict.csv")
    mp = v[(v.criterion == "marchenko_pastur") & (v["总组合数"] > 0)]
    out["exp7_mp_exact"] = float(mp["秩判对率"].max())

    # ---- exp8：降偏比的实现间散布（极差的最小与最大值）
    t = _read("exp8_table.csv")
    fold = (t.groupby(["n_axis", "mode", "f2_over_f1"])["rho"]
            .agg(lambda x: x.max() / x.min()))
    out["exp8_fold_min"] = float(fold.min())
    out["exp8_fold_max"] = float(fold.max())

    # ---- exp9：复值 / 白噪声下的降偏比
    e9 = _read("exp9_verdict.csv")
    w = e9[(e9.channel == "complex") & (e9.noise == "white")]
    out["exp9_ratio_white"] = float(w["降偏比"].iloc[0])

    # ---- exp10：预白化「完全达到 oracle」的配置数
    e10 = _read("exp10_table.csv")
    piv = (e10[e10.criterion == "gavish_donoho"]
           .pivot_table(index=["channel", "noise"], columns="scheme", values="p_exact"))
    out["exp10_n_config"] = int(len(piv))
    out["exp10_n_oracle_ok"] = int(((piv["estimated"] > 0.99) & (piv["oracle"] > 0.99)).sum())

    # ---- exp11：TDMD 偏差不可分辨的个数、方差比 < 1 的个数
    e11 = _read("exp11_verdict.csv")
    out["exp11_n_total"] = int(len(e11))
    out["exp11_n_unresolvable"] = int((~e11["TDMD可分辨"].astype(bool)).sum())
    out["exp11_n_var_lt1"] = int((e11["方差比_TDMD_over_DMD"] < 1).sum())
    return out


def check_numbers(doc: Path, f: dict[str, object]) -> list[Check]:
    """把报告里的**结论性数值**与从 CSV 重算的结果比对。"""
    if not doc.exists():
        return []
    text = _mask_history(doc.read_text(encoding="utf-8"))
    out: list[Check] = []

    def add(name: str, pattern: str, *values: float, tol: float = 0.02) -> None:
        """把 ``pattern`` 的捕获组与 ``values`` 比对。

        ⚠️ **检查所有出现**（理由见 ``check_doc`` 里同名函数的说明）：报告常把同一个
        数字写进摘要、正文与表格，只查第一处会漏掉其余。任一不符即 FAIL。
        """
        hits = list(re.finditer(pattern, text, re.S))
        if not hits:
            out.append(Check(name, "-", _fmt(values), None,
                             f"{doc.name}（未找到声明）", numeric=False))
            return
        first, bad = "", 0
        for m in hits:
            try:
                got = tuple(float(g.replace(" ", "")) for g in m.groups())
            except (ValueError, AttributeError):
                bad += 1
                continue
            if not first:
                first = _fmt(got)
            if not (len(got) == len(values)
                    and all(abs(g - v) <= tol * max(abs(v), 1e-9)
                            for g, v in zip(got, values))):
                bad += 1
        out.append(Check(
            name, first + (f"（共 {len(hits)} 处）" if len(hits) > 1 else ""),
            _fmt(values), bad == 0,
            f"{doc.name}:{_loc(text, hits[0].start())}"
            + (f"，其中 {bad} 处不符" if bad else ""), numeric=False))

    # 每项数字都对应报告中的一处**具体措辞**；措辞改了就 WARN（提醒更新正则）
    add("数值 n=4→16 降幅", r"偏差(?:降|下降)\s*\*{0,2}(\d+)\s*倍",
        round(float(f["exp1_fold"])), tol=0.01)
    add("数值 std 比区间", r"`std` 比落在 \*{0,2}([\d.]+)–([\d.]+)\*{0,2}",
        float(f["exp1_std_min"]), float(f["exp1_std_max"]))
    add("数值 std CI>1 个数", r"`std` 比 CI 下界 > 1 的为 \*{0,2}(\d+)\*{0,2} 个",
        float(f["exp1_std_ci_above1"]), tol=0.0)
    add("数值 σ̂/σ 校正后", r"σ_true` 由 0\.931 → \*{0,2}([\d.]+)\*{0,2}",
        float(f["exp3_sigma_complex"]))
    # ⚠️ 锚点必须带上「1.25 … 1.83」这个上下文。曾用 `DMD 偏差降 ([\d.]+) 倍`，
    # 它会**误抓 §8.1 的「DMD 偏差降 113 倍」**（那是 `n` 的降幅，不是间隔的），
    # 把正确的报告判成不一致。摘要与 §2.6 表格措辞不同，故分两条。
    add("数值 间隔倍率(摘要)", r"由 1\.25 拉到 1\.83 时 DMD 偏差降 ([\d.]+) 倍",
        float(f["exp4_fold"]), tol=0.03)
    add("数值 间隔倍率(§2.6)", r"由 1\.25 → 1\.83 \| \*\*降 ([\d.]+) 倍\*\*",
        float(f["exp4_fold"]), tol=0.03)
    # 端点（单位 1e-3）：与倍率同口径，否则两者互相矛盾
    add("数值 间隔端点(×1e-3)", r"偏差中位端点 ([\d.]+)e-3 → ([\d.]+)e-3",
        float(f["exp4_lo_x1e3"]), float(f["exp4_hi_x1e3"]), tol=0.03)
    # 交叉点：报告以表格形式给出（`| 1 | **1.86** | [1.85, 1.90] |`），
    # 用区间值作唯一性锚点，避免与其它表格的同形行撞车
    add("数值 交叉点 模态1", r"\| 1 \| \*\*([\d.]+)\*\* \| \[1\.85",
        float(f["exp5_cross_mode1"]))
    add("数值 交叉点 模态2", r"\| 2 \| \*\*([\d.]+)\*\* \| \[1\.60",
        float(f["exp5_cross_mode2"]))
    add("数值 GD MSE 比恒为 1", r"MSE 比在全部 \d+ 个组合上都恰为 `([\d.]+)`",
        float(f["exp6_gd_ratio_max"]), tol=1e-6)
    add("数值 MP 判对率", r"秩判对率由 0\.112 → \*{0,2}([\d.]+)\*{0,2}",
        float(f["exp7_mp_exact"]))
    add("数值 实现散布极差", r"散布达 \*{0,2}([\d.]+)–([\d.]+) 倍\*{0,2}",
        float(f["exp8_fold_min"]), float(f["exp8_fold_max"]), tol=0.10)
    add("数值 预白化达 oracle", r"(\d+) 个配置中的 (\d+) 个上完全达到 oracle",
        float(f["exp10_n_config"]), float(f["exp10_n_oracle_ok"]), tol=0.0)
    # exp11：报告表格里的引号是 **ASCII `"`**（U+0022），不是弯引号 ——
    # 正则同时容忍两种，避免换引号就失配
    add("数值 exp11 不可分辨",
        r'TDMD 偏差不可分辨\*{0,2}（即["\u201c\u201d]基本无偏["\u201c\u201d]）\s*\|\s*\*{0,2}(\d+)/(\d+)',
        float(f["exp11_n_unresolvable"]), float(f["exp11_n_total"]), tol=0.0)
    add("数值 exp11 方差比<1",
        r"方差比 < 1\*{0,2}（原文.*?）\s*\|\s*\*{0,2}(\d+)/(\d+)",
        float(f["exp11_n_var_lt1"]), float(f["exp11_n_total"]), tol=0.0)
    return out


def _fmt(vals: tuple[float, ...]) -> str:
    """把一组数值格式化成便于比对的短串。"""
    parts = []
    for v in vals:
        if v == int(v) and abs(v) < 1e6:
            parts.append(str(int(v)))
        else:
            parts.append(f"{v:.4g}")
    return " / ".join(parts)


# ---------------------------------------------------------------- 输出
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="核对文档数字与仓库产物是否一致")
    ap.add_argument("--quiet", action="store_true", help="只输出非 OK 项")
    ap.add_argument("--no-pytest", action="store_true",
                    help="跳过 pytest 收集（更快，但无法核对测试项数）")
    args = ap.parse_args(argv)

    print("=== 实测产物 ===")
    f = measure(use_pytest=not args.no_pytest)
    for k, v in f.items():
        if k != "pytest_by_file":
            print(f"  {k:16s} {v}")
    print()

    checks: list[Check] = []
    checks += check_doc(REPORT, f, detailed=True)     # 交付物清单（完整声明）
    checks += check_doc(README, f, detailed=False)    # 只核它实际声明的那几项

    # ---- 结论性数值：从 CSV 重算后与报告比对（这是比规模数字更重要的一类）
    try:
        checks += check_numbers(REPORT, compute_numbers())
    except FileNotFoundError as e:
        print(f"  (跳过数值核对：缺少结果 CSV {e.filename} —— 先跑 --exp all)")

    n_ok = sum(c.ok is True for c in checks)
    n_fail = sum(c.ok is False for c in checks)
    n_warn = sum(c.ok is None for c in checks)

    print("=== 声明 vs 实测 ===")
    print(f"  {'':4} {'检查项':22s} {'文档声明':>14s} {'实测':>14s}   位置")
    for c in checks:
        if args.quiet and c.ok is True:
            continue
        print(f"  {c.status} {c.name:22s} {c.show_claimed():>14s} "
              f"{c.actual:>14s}   {c.where}")

    print()
    print(f"一致 {n_ok} 项；不一致 {n_fail} 项；未能核对 {n_warn} 项")
    if n_warn:
        print("  注意：WARN 表示文档中**找不到**该声明 —— 措辞可能已改，"
              "需更新脚本正则；不要当作通过。")
    if n_fail:
        print("  不一致项已在上表标 FAIL，请修正文档或产物。")
    return 0 if (n_fail == 0 and n_warn == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
