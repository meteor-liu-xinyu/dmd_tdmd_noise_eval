"""核对文档里的「规模数字」与仓库实际产物是否一致。

为什么需要这个脚本
------------------

本项目的报告与 README 里有大量**容易过期**的数字：图表张数、测试项数、代码行数、
数据行数、ADR 编号范围、配置指纹。它们分散在多处、由不同轮次的改动引入，
**只有在人工复核时才会被发现**。本项目已经发生过两次同类事故：

- 完整 review 时手工查出 4 处不一致（其中 §2.13 的「9/12」是真错）；
- 新增 exp11 后，报告页头的 ``fig1–fig15``、``ADR-001 ~ ADR-018``、``187 项``、
  ``2 040 行``、``1 170 行`` 同时失效（实际为 16 张 / 019 / 210 项 / 2 425 行 / 1 190 行），
  而报告 §6 的测试分解里**整条 `test_paper_replication` 23 项缺失**。

这类错误的共性是：**文档改动与产物改动不同步**，且没有任何自动机制会发现。
本脚本把上述数字全部改成"从产物实测"，与文档声明逐一比对。

设计要点
--------

**找不到声明时报 WARN，而不是静默通过。** 如果文档措辞改变导致正则失配，
脚本必须让人看见 —— 否则会制造"检查通过"的假象，比不检查更危险。

用法::

    python scripts/check_report_consistency.py           # 完整检查
    python scripts/check_report_consistency.py --quiet   # 只输出问题
    python scripts/check_report_consistency.py --no-pytest   # 跳过 pytest 收集（更快）

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


# ---------------------------------------------------------------- 检查
def check_doc(doc: Path, f: dict[str, object], *, detailed: bool) -> list[Check]:
    """对单个文档做**该文档应有的**对照检查。

    ``detailed=True`` 用于 ``final-report.md``（含交付物清单的完整声明）；
    ``README.md`` 只声明了其中一部分，故传 ``False`` —— 否则会为它没有的那些项
    刷出一片 WARN，把真实问题淹没在噪声里。
    """
    if not doc.exists():
        return [Check(f"{doc.name}", "-", "文档不存在", False)]
    text = doc.read_text(encoding="utf-8")
    out: list[Check] = []

    def add(name: str, pattern: str, actual: object, cast=_num,
            numeric: bool = True) -> None:
        """在文档里找 ``pattern`` 的第 1 个捕获组，与 ``actual`` 比对。"""
        m = re.search(pattern, text)
        if not m:
            out.append(Check(name, "-", str(actual), None,
                             f"{doc.name}（未找到声明）", numeric))
            return
        out.append(Check(name, m.group(1).strip(), str(actual),
                         cast(m.group(1)) == actual,
                         f"{doc.name}:{_loc(text, m.start())}", numeric))

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
