"""在最终报告中插入 / 重排 fig1–fig16 的图片引用。

背景
----

报告正文此前只以行内代码提到图名（如「见图 `fig10`」），**从不嵌入图片**；
且 `exp4`–`exp11` 的图（`fig11`–`fig16`）**完全没有被正文引用**。结果是
18 页纯文字报告配 16 张独立 PNG，对评审不友好。

本脚本做两件事：

1. 对**正文已引用**的图，插图位置 = 首次引用所在块之后（读者读到引用即见图）；
2. 对**未被引用**的图，插图位置 = 其所属章节的末尾（不让图悬空在无关处）。

段落锚点规则
------------

从锚点行向后找**第一个空行**，插在其后 —— markdown 的块由空行分隔，该规则对
段落、引用块、表格、列表都安全（不会插进表格中间）。另有三条避让：

- **避让表格**（`avoid_table`）：锚点句常以「：」引出紧随的表格，把图插在
  句子与表格之间会打断句式；
- **避让分隔线**（`avoid_hr`）：`---` 后跟图片行会被 pandoc 按 **YAML metadata
  block** 解析，图片行以 `!` 开头 → `!` 在 YAML 里是标签指示符 → 构建失败；
- **图片前后各留一个空行**：pandoc 只把「独占一段」的图片渲染成 `figure` 环境；
  紧贴前一段落会退化为**行内图片**，从而不受 `max width` 约束、按原始尺寸
  溢出页面（实测 `fig2` 因此溢出 122.9pt ≈ 4.3cm，内容被裁）。

caption 与编号
--------------

caption **自带「图 N ·」**，并由 `scripts/export_pdf.py` 用
`\\captionsetup{labelformat=empty}` **关掉 LaTeX 自动编号** —— 因为自动编号按
**出现顺序**计数，而本报告的图并非按 1–16 出现（`fig10` 属 §3、`fig16` 属 §2.15），
实测会把 `fig16` 标成「图 15」、`fig10` 标成「图 16」，与正文引用的 `figN` 对不上。

用法::

    python scripts/insert_figures.py            # dry-run，打印插入计划
    python scripts/insert_figures.py --apply    # 写入

若报告已含图片引用，脚本会拒绝执行（避免重复插入）；确需重做请先
`git checkout docs/final-report.md` 或加 `--force`。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "final-report.md"

#: 图片在 md 中的相对路径（md 位于 docs/，图片位于 results/figures/）
REL = "../results/figures"

#: 每张图的 caption 与**默认章节锚点**（当正文未引用该图时使用）。
#: caption 取自 `report/figures.py` 各图 docstring 的权威说明。
FIGS: dict[int, tuple[str, str, str]] = {
    # n: (文件名, caption, 未引用时的锚点小节标题)
    1: ("fig1_bias_vs_channels.png", "偏差随观测维数 `n` 的变化（实验一）", "2.1 观测维数是第一因素"),
    2: ("fig2_bias_vs_snr.png", "偏差随 SNR 的变化（实验一）", "2.2 可用 SNR 窗口由配对失败率决定"),
    3: ("fig3_pair_fail_vs_snr.png", "配对失败率随 SNR 的变化 —— 决定可用 SNR 窗口", "2.2 可用 SNR 窗口由配对失败率决定"),
    4: ("fig4_amp_ratio.png", "模态幅度比的影响（实验一 amp 轴）", "2.2 可用 SNR 窗口由配对失败率决定"),
    5: ("fig5_variance_cost.png", "方差代价森林图：`std_TDMD / std_DMD` 及配对 bootstrap CI", "2.3 TDMD 无方差代价"),
    6: ("fig6_bias_vs_m.png", "偏差随快照数 `m` 的变化，两种 SNR 归一化口径对比（实验二）", "2.4 偏差地板"),
    7: ("fig7_rmse_vs_m.png", "RMSE 随 `m` 的 log-log 图，含理论 `m^-1/2` 参考线", "2.4 偏差地板"),
    8: ("fig8_slopes.png", "斜率汇总（两种口径）", "2.4 偏差地板"),
    9: ("fig9_rank_robustness.png", "秩判据对 `σ̂` 误估的鲁棒性（实验三）", "2.5 秩判据的鲁棒性"),
    10: ("fig10_decision_map.png", "适用边界决策图：偏差可分辨性 + 配对成功率 → 四类区域", "3. 适用边界决策表"),
    11: ("fig11_robustness.png", "参数稳健性：偏差对模态间隔与窗内衰减的敏感性（实验四）", "2.6 参数稳健性"),
    12: ("fig12_crossover.png", "降偏比 `rho` 随模态间隔的变化与交叉点（实验五）", "2.7 交叉点的精确位置"),
    13: ("fig13_end_to_end.png", "端到端：`σ̂`、秩判定与频率精度退化（实验六）", "2.8 ★ 端到端评估"),
    14: ("fig14_rank_schemes.png", "秩估计方案的补救效果（实验七）", "2.10 秩估计方案对比"),
    15: ("fig15_crossover_scatter.png", "交叉点的实现间散布（实验八）", "2.11 ★ 交叉点的实现间散布"),
    16: ("fig16_paper_replication.png", "原文 Figure 1 复现（Hemati et al. 2017 §3）—— 外部一致性验证", "2.15 ★ 原文 Figure 1 复现"),
}

#: 页头那行含「图表 fig1–fig16」，会让 fig1/fig16 被误判为"已引用" → 排除
HEADER_LINE = 3

#: 「图表 fig1–fig16」这类**范围表述**不是引用，且会把 fig16 误判为已引用
RANGE_PAT = re.compile(r"fig1\s*[–\-]\s*fig\d+")


def load() -> list[str]:
    return REPORT.read_text(encoding="utf-8").split("\n")


def find_ref(lines: list[str], n: int) -> int | None:
    """返回 `figN` 首次被引用的行号（1-based），无则 None。

    需排除三种**非引用**的匹配，否则会插到错误位置：页头的「图表 fig1–fig16」、
    §6 交付物清单里的同名范围表述，以及其它表格行（都不是"讨论图"的正文）。
    """
    pat = re.compile(rf"fig{n}(?![0-9])")
    for i, ln in enumerate(lines, 1):
        if i == HEADER_LINE or RANGE_PAT.search(ln):
            continue
        if ln.lstrip().startswith("|"):        # 表格行
            continue
        if pat.search(ln):
            return i
    return None


def _level(ln: str) -> int:
    """返回 markdown 标题级别（``#`` 个数）；非标题返回 0。"""
    m = re.match(r"^(#+) ", ln)
    return len(m.group(1)) if m else 0


def find_section(lines: list[str], title: str) -> int | None:
    """返回 `title`（小节标题的子串）所在行的行号。"""
    for i, ln in enumerate(lines, 1):
        if ln.startswith("#") and title in ln:
            return i
    return None


def section_end(lines: list[str], start: int) -> int:
    """返回 `start` 行所属章节的**最后一行**（1-based）。

    ⚠️ 必须按**标题级别**判定结束：`### 2.x` 章节的结束是下一个 `###` 或 `##`，
    而不是下一个 `##` —— 只在 `##` 处结束会把所有 `### 2.x` 的末尾都算到 §3 之前，
    使未引用的图全部挤到同一处（实测 7 张图都落到了 L825）。
    """
    lvl = _level(lines[start - 1]) if 0 < start <= len(lines) else 0
    if lvl == 0:
        return start
    for i in range(start + 1, len(lines) + 1):
        L = _level(lines[i - 1])
        if L and L <= lvl:
            j = i - 1
            while j > start and not lines[j - 1].strip():
                j -= 1
            return j
    return len(lines)


def block_end(lines: list[str], start: int) -> int:
    """返回从 `start` 行（1-based）起的块之后的插入位置（1-based 行号）。

    规则：向后找**第一个空行**，插在其后 —— markdown 的块由空行分隔，
    故对段落/引用块/表格/列表都安全（不会插进表格中间）。
    """
    for i in range(start - 1, len(lines)):
        if not lines[i].strip():
            return i + 2                      # 空行之后（1-based）
    return len(lines) + 1


def avoid_table(lines: list[str], pos: int) -> int:
    """若插入点紧接着一个表格，把插入点移到**表格之后**。

    理由：锚点句常以「：」引出紧随的表格（如「据此划出四类区域（见图 `fig10`）：」），
    把图插在句子与表格之间会**打断这一句式**。
    """
    while 0 < pos <= len(lines):
        i = pos - 1
        while i < len(lines) and not lines[i].strip():      # 跳过空行
            i += 1
        if i < len(lines) and lines[i].lstrip().startswith("|"):
            pos = block_end(lines, i + 1)
        else:
            break
    return pos


def avoid_hr(lines: list[str], pos: int) -> int:
    """若插入点前的最近非空行是 ``---`` 分隔线，把插入点移到**该线之前**。

    ⚠️ 这是必须的：图片行以 ``!`` 开头，而 pandoc 会把 ``---`` 与其后的行
    按 **YAML metadata block** 解析 —— ``!`` 在 YAML 里是标签指示符，于是报

        YAML parse exception ... while scanning a tag

    整个 PDF 构建失败（实测 fig13 因落在 §2.8 末尾的 ``---`` 之后而触发）。
    """
    i = pos - 2                      # 插入位置前一行的索引（0-based）
    while i >= 0 and not lines[i].strip():
        i -= 1
    if i >= 0 and lines[i].strip() == "---":
        return i + 1                 # 插到该分隔线之前（1-based）
    return pos


def figure_md(n: int) -> str:
    """生成图片的 markdown 行。

    ⚠️ caption **必须自带「图 N ·」编号** —— 理由见模块 docstring（自动编号会错位）。
    """
    fname, cap, _ = FIGS[n]
    return f"![图 {n} · {cap}]({REL}/{fname})"


def plan(lines: list[str]) -> list[tuple[int, int, str, str]]:
    """生成插入计划：``(图号, 插入行号, 插入内容, 依据)``。"""
    out: list[tuple[int, int, str, str]] = []
    for n in sorted(FIGS):
        ref = find_ref(lines, n)
        if ref is not None:
            pos = avoid_hr(lines, avoid_table(lines, block_end(lines, ref)))
            why = f"引用处 L{ref}"
        else:
            sec = find_section(lines, FIGS[n][2])
            if sec is None:
                print(f"  !! fig{n}: 找不到锚点小节 {FIGS[n][2]!r}，跳过")
                continue
            pos = avoid_hr(lines, section_end(lines, sec) + 1)
            why = f"章节末尾（{FIGS[n][2]}）"
        out.append((n, pos, figure_md(n), why))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="在报告中插入 fig1–fig16")
    ap.add_argument("--apply", action="store_true", help="实际写入（默认 dry-run）")
    ap.add_argument("--force", action="store_true",
                    help="即使报告已含图片引用也执行（会重复插入，慎用）")
    args = ap.parse_args(argv)

    lines = load()
    existing = sum(1 for ln in lines if ln.startswith("!["))
    if existing and not args.force:
        print(f"!! 报告中已存在 {existing} 个图片引用，拒绝重复插入。")
        print("   如需重做：git checkout docs/final-report.md 后重跑，或加 --force。")
        return 1

    items = plan(lines)

    print(f"=== 插入计划（{len(items)} 张图）===")
    for n, pos, body, why in items:
        nxt = lines[pos - 1].strip()[:50] if 0 < pos - 1 < len(lines) else "(文末)"
        print(f"  fig{n:<2} 前插于 L{pos:<5} {why:34s} 其后为: {nxt or '(空行)'}")

    if not args.apply:
        print("\n(dry-run —— 未写入。加 --apply 执行)")
        return 0

    # 从后往前插入，避免行号偏移。
    # ⚠️ 图片**前后各需一个空行**：pandoc 只把「独占一段」的图片渲染成 figure
    # 环境；紧贴前一段落会退化为**行内图片**，从而不受尺寸约束、按原始尺寸溢出
    # 页面（实测 fig2 因此溢出 122.9pt ≈ 4.3cm，内容被裁）。
    # ⚠️ 同一位置的多张图必须按图号**降序**处理：每次 insert 都会把先前插入的
    # 内容往后推，故先插图号大的、后插小的，最终才是 fig6→fig7→fig8 的正序。
    for n, pos, body, _ in sorted(items, key=lambda t: (-t[1], -t[0])):
        lines.insert(pos - 1, "")          # 尾空行
        lines.insert(pos - 1, body)
        lines.insert(pos - 1, "")          # 首空行

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n已插入 {len(items)} 张图片引用 -> {REPORT}")
    print("重新导出 PDF：python scripts/export_pdf.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
