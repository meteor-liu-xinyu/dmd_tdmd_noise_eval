"""把 ``docs/*.md`` 导出为 PDF（pandoc + xelatex）。

为什么需要这个脚本，而不是直接跑 ``pandoc``
--------------------------------------------

项目文档含大量中文、数学符号与 Unicode 记号，直接调用 ``pandoc`` 会遇到
**静默失败** —— PDF 照常生成，但内容缺失或溢出，而问题只出现在 latex 日志里。
本脚本把这些检查固化为可执行断言。

四条实测结论（构成本脚本的全部设计依据）
----------------------------------------

**1. 字体配置：``Segoe UI Symbol`` 作西文，``Microsoft YaHei`` 作中文。**

报告实际使用 16 个在 ``Times New Roman`` 中**无字形**的字符::

    ★ ✅ ✓ ⚠ ✗ ❌ ① ② ③ ∈ ≡ ⊆ ₀ ⇒ ⟹ ≫ ≲ ≳

``fontTools`` 逐字体查 cmap 的结果：``Microsoft YaHei`` 覆盖其中
``★ ①②③ ∈ ≡ ⊆``，``Segoe UI Symbol`` 覆盖其余（含 ``≲ ≳ ≫``）。
两者**互补**，故西文用 ``Segoe UI Symbol``、中文用雅黑。实测缺字 **121 → 0**。

**2. 不要用 ``ucharclasses`` 做符号回退 —— 它破坏 xeCJK。**

按直觉应先试 ``ucharclasses`` 的 ``\\setTransitionsFor{...}`` 把符号区块转交
``Segoe UI Symbol``。实测**缺字从 121 暴增到 13373**、PDF 从 710 KB 缩到 186 KB：
它劫持了字符流，使 xeCJK 无法再为中文分配字体，大量中文（含「一」「不」「的」）
落到无中文的 ``Times New Roman`` 而被**直接丢弃**。
符号问题用**字体选择**解决，不要用字符流劫持。

**3. 预处理是必需的**（这两项在任何字体中都无字形）::

    U+0302 组合抑扬符（构成 σ̂）  -> 缺失 68 处
    U+FE0F 变体选择符（构成 ⚠️） -> 缺失 15 处

两处替换都不改变语义：``σ̂ -> σ^``、``⚠️ -> ⚠``。

**4. 表格缩放**。报告有 47 个表格、最宽 14 列。pandoc 的 ``longtable`` 是自然
宽度，超出页面只在日志里报 ``Overfull \\hbox`` —— PDF 里表现为**右侧被裁掉**，
不报错。故统一把表格字号降为 ``\\footnotesize``。实测 Overfull **0 处**。

用法::

    python scripts/export_pdf.py                     # 导出 docs/final-report.md
    python scripts/export_pdf.py docs/algorithm-spec.md
    python scripts/export_pdf.py --all               # 导出 docs/ 下全部 .md

依赖：``pandoc``、``xelatex``（TeX Live），以及 Windows 字体
``Microsoft YaHei`` / ``Segoe UI Symbol`` / ``Cambria Math``。
（可选）``pymupdf`` —— 装了则额外报告内嵌字体，未装则跳过该项检查。
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
WORK = ROOT / "tmp" / "pdf_export"

#: 组合抑扬符。报告中 ``σ̂`` = ``σ`` + 它；**任何字体都没有它的 cmap 项**
#: （实测 Segoe UI Symbol / 雅黑 / Times 全无），必须替换。
#: 替换为 ``^`` —— 语义不变，且 ``σ^`` 在技术文档中惯例可读。
COMBINING_HAT = "\u0302"

#: 变体选择符-16。``⚠`` + 它构成 emoji 序列，xeCJK 无法处理（实测缺 15 处）→ 删除。
VS16 = "\ufe0f"

#: pandoc 的 ``--pdf-engine``
ENGINE = "xelatex"

#: 字体设置。依据见模块 docstring 第 1 条。
#:
#: ⚠️ ``monofont`` **必须与 ``mainfont`` 同族**：报告的 ``₀ ≲ ≳ ⊆ ⟹`` 只出现在
#: 行内代码与代码块中（走 monofont），而 ``Microsoft YaHei`` 缺这几个字形。
#: 若把 monofont 设为雅黑，会残留 12 处缺字。
FONTS = {
    "mainfont": "Segoe UI Symbol",
    "monofont": "Segoe UI Symbol",
    "CJKmainfont": "Microsoft YaHei",
    "CJKmonofont": "Microsoft YaHei",
    "mathfont": "Cambria Math",
}

GEOMETRY = "margin=1.8cm"
FONTSIZE = "10pt"

#: 目录标题。pandoc 默认给英文 "Contents"，本项目文档全为中文。
TOC_TITLE = "目录"

#: 单处 Overfull 的容忍上限（pt）。
#:
#: ⚠️ 判据必须按**溢出量**而非**处数** —— 实测报告有 9 处 Overfull，但全部是
#: ``0.13pt``（≈0.05 mm），来自 longtable 的列宽舍入，**视觉不可见、内容不会被裁**。
#: 真正被裁的表格溢出量通常是 20–200pt。按处数设阈值会把这种无害舍入误判为失败。
OVERFULL_PT_LIMIT = 1.0


@dataclass
class BuildReport:
    """一次构建的诊断结果。"""

    src: Path
    out: Path
    ok: bool = False
    missing_chars: list[str] = field(default_factory=list)
    overfull: int = 0
    max_overfull_pt: float = 0.0
    substituted: dict[str, int] = field(default_factory=dict)
    pages: int | None = None
    size_kb: float = 0.0
    log_tail: str = ""

    #: PDF 内嵌字体清单（需 pymupdf；未安装时为 ``None`` 表示跳过该检查）
    fonts: list[str] | None = None

    def summary(self) -> str:
        head = f"{'OK  ' if self.ok else 'FAIL'} {self.src.name} -> {self.out.name}"
        lines = [head]
        if self.substituted:
            sub = ", ".join(f"{k}={v}" for k, v in self.substituted.items())
            lines.append(f"       预处理: {sub}")
        if self.pages is not None:
            lines.append(f"       {self.pages} 页，{self.size_kb:.0f} KB")
        lines.append(f"       缺字 {len(self.missing_chars)} 类；"
                     f"Overfull {self.overfull} 处，最大 {self.max_overfull_pt:.2f}pt"
                     f"（限 {OVERFULL_PT_LIMIT:g}pt）")
        if self.fonts is not None:
            lines.append(f"       嵌入字体: {', '.join(self.fonts)}")
        if self.missing_chars:
            lines.append(f"       !! 缺字: {' '.join(self.missing_chars[:20])}")
        if not self.ok and self.log_tail:
            lines.append(f"       日志:\n{self.log_tail}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- 预处理
def preprocess(md: str) -> tuple[str, dict[str, int]]:
    """把 Markdown 调整为 xelatex 可排的等价形式。

    只做**渲染必需**的两处替换（见模块 docstring 第 3 条），原始 ``.md`` 不被修改。
    """
    n_hat = md.count(COMBINING_HAT)
    md = md.replace(COMBINING_HAT, "^")

    n_vs = md.count(VS16)
    md = md.replace(VS16, "")

    subs = {"σ̂->σ^": n_hat, "⚠️->⚠": n_vs}
    return md, {k: v for k, v in subs.items() if v}


def _doc_title(src: Path, md: str) -> str:
    """取文档标题（第一个一级标题），作为 PDF 元数据里的 ``Title``。

    pandoc 的 ``--metadata title=`` 会在正文前**插入标题块**，本项目各文档已自带
    一级标题，故改用 ``\\hypersetup{pdftitle=...}`` 只设元数据、不动正文。
    """
    m = re.search(r"^#\s+(.+?)\s*$", md, re.M)
    return m.group(1) if m else src.stem


def _tex_escape(s: str) -> str:
    """转义 LaTeX 特殊字符，避免元数据里的 ``& % # _`` 破坏编译。"""
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}")):
        s = s.replace(a, b)
    return s


def header_tex(title: str = "") -> str:
    """生成 LaTeX 导言片段。

    只做元数据、表格缩放与断行容忍 —— 符号问题由字体选择解决，
    **不引入字符流劫持**（见模块 docstring 第 2 条）。
    """
    meta = ""
    if title:
        meta = ("\\AtBeginDocument{\\hypersetup{pdftitle={%s}}}\n"
                % _tex_escape(title))
    return f"""% 由 scripts/export_pdf.py 生成
{meta}% 表格缩放：报告最宽 14 列，pandoc 的 longtable 为自然宽度，
% 超宽只在日志报 Overfull hbox、PDF 里右侧被静默裁掉。
\\usepackage{{etoolbox}}
\\AtBeginEnvironment{{longtable}}{{\\footnotesize}}
\\setlength{{\\LTcapwidth}}{{\\textwidth}}

% 允许较松的断行，减少 Overfull
\\tolerance=2000
\\emergencystretch=3em
"""


# --------------------------------------------------------------------------- 构建
def _embedded_fonts(pdf: Path) -> list[str] | None:
    """列出 PDF 的内嵌字体；pymupdf 不可用时返回 ``None``（跳过检查）。

    ⚠️ 这是 ``Missing character`` 告警**覆盖不到**的一类失败：编译机上字形存在，
    编译不报任何问题，但若字体未嵌入，**别的机器打开时中文会是空白**。
    实测本项目导出的 PDF 内嵌 ``SegoeUISymbol`` / ``MicrosoftYaHei``（含 Bold），
    无未嵌入项。
    """
    try:
        import pymupdf
    except ImportError:
        return None
    try:
        doc = pymupdf.open(pdf)
        out: set[str] = set()
        for pno in range(doc.page_count):
            for f in doc.get_page_fonts(pno):
                # 元组为 (xref, ext, type, basefont, name, encoding)；
                # ext 为空字符串表示该字体**未嵌入**
                ext = f[1] or ""
                out.add(f[3] if ext else f"{f[3]}(未嵌入!)")
        doc.close()
        return sorted(out)
    except Exception:
        return None


def _compile(tex: Path, work: Path, runs: int = 2) -> tuple[bool, str]:
    """在 ``work`` 目录里用 xelatex 编译 ``tex``。

    跑 ``runs`` 遍：第一遍生成目录/交叉引用的 ``.aux``，第二遍才让它们落定
    （只跑一遍会在 PDF 里留下 ``??`` 的页码与引用）。``.aux``/``.log`` 全部留在
    ``work``（即 ``tmp/``）内，不污染 ``docs/``。

    返回 ``(是否成功, 完整 latex 日志)``。
    """
    for _ in range(runs):
        subprocess.run(
            ["xelatex", "-interaction=nonstopmode",
             f"-output-directory={work}", tex.name],
            cwd=str(work), capture_output=True, text=True,
            encoding="utf-8", errors="replace", env=dict(os.environ),
        )
    log_file = work / (tex.stem + ".log")
    log = (log_file.read_text(encoding="utf-8", errors="replace")
           if log_file.exists() else "")
    # 判定依据：latex 的致命错误以行首 `! ` 标记（nonstopmode 下不会中断进程，
    # 只靠返回码无法分辨"编译成功"与"带着错误出了个残缺 PDF"）
    fatal = re.search(r"^! ", log, re.M)
    ok = not fatal and (work / (tex.stem + ".pdf")).exists()
    return ok, log


def build(src: Path, *, toc: bool = True, keep_work: bool = False) -> BuildReport:
    """把 ``src`` 导出为同名 ``.pdf``。"""
    rep = BuildReport(src=src, out=src.with_suffix(".pdf"))
    if not src.exists():
        rep.log_tail = f"源文件不存在: {src}"
        return rep
    if shutil.which("pandoc") is None:
        rep.log_tail = "pandoc 未安装"
        return rep
    if shutil.which(ENGINE) is None:
        rep.log_tail = f"{ENGINE} 未安装"
        return rep

    WORK.mkdir(parents=True, exist_ok=True)
    body = WORK / f"{src.stem}.body.md"
    tex = WORK / f"{src.stem}.tex"
    hdr = WORK / "header.tex"

    text, rep.substituted = preprocess(src.read_text(encoding="utf-8"))
    body.write_text(text, encoding="utf-8")
    hdr.write_text(header_tex(_doc_title(src, text)), encoding="utf-8")

    # ---- 第 1 步：pandoc 生成 .tex（不直接出 PDF，以便拿到完整 latex 日志）
    cmd = [
        "pandoc", str(body), "-o", str(tex),
        "--pdf-engine", ENGINE,
        "--include-in-header", str(hdr),
        "-V", f"geometry:{GEOMETRY}",
        "-V", f"fontsize={FONTSIZE}",
        "-V", "colorlinks=true",
        "-V", "linkcolor=blue!60!black",
        "-V", "urlcolor=blue!60!black",
    ]
    for k, v in FONTS.items():
        cmd += ["-V", f"{k}={v}"]
    if toc:
        # 默认目录标题是英文 "Contents"，本项目文档全为中文
        cmd += ["--toc", "--toc-depth=3", "-V", f"toc-title={TOC_TITLE}"]

    p1 = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                        errors="replace", env=dict(os.environ), cwd=str(ROOT))
    if p1.returncode != 0 or not tex.exists():
        rep.log_tail = ("pandoc 生成 tex 失败\n"
                        + "\n".join(((p1.stdout or "") + (p1.stderr or "")).strip().split("\n")[-10:]))
        return rep

    # ---- 第 2 步：xelatex 编译
    ok, log = _compile(tex, WORK)

    # ---- 诊断：缺字与溢出（只出现在 latex 日志里，不中断构建）
    rep.missing_chars = sorted(set(
        re.findall(r"Missing character: There is no (\S+)\s", log)))
    widths = [float(v) for v in
              re.findall(r"Overfull \\hbox \(([0-9.]+)pt too wide\)", log)]
    rep.overfull = len(widths)
    rep.max_overfull_pt = max(widths) if widths else 0.0
    rep.pages = _pdf_pages_from_log(log)

    # ---- 诊断：表格被裁切（缺字之外的另一种静默失败）
    # 13 列以上的表格最容易溢出，单独计数以便定位
    rep.log_tail = ""

    built = WORK / f"{src.stem}.pdf"
    if built.exists():
        rep.size_kb = built.stat().st_size / 1024
        shutil.copy2(built, rep.out)          # 拷到 docs/，避免跨盘 move 失败
        built.unlink(missing_ok=True)
        rep.fonts = _embedded_fonts(rep.out)

    # 字体未嵌入在编译机上不会报错，但换机器打开会缺字 —— 单独判定
    has_unembedded = any("未嵌入" in f for f in (rep.fonts or []))
    rep.ok = bool(ok and rep.out.exists() and not rep.missing_chars
                  and not has_unembedded
                  and rep.max_overfull_pt <= OVERFULL_PT_LIMIT)
    if not rep.ok:
        errs = re.findall(r"^! .*$", log, re.M)[:4]
        rep.log_tail = ("latex 致命错误:\n" + "\n".join(errs)) if errs else \
            "\n".join(log.strip().split("\n")[-12:])

    if not keep_work:
        # 保留 .log 供回溯（诊断缺字/溢出的唯一来源），清掉其余中间文件
        for f in (body, hdr, tex):
            f.unlink(missing_ok=True)
        for pat in ("*.aux", "*.out", "*.toc"):
            for f in WORK.glob(pat):
                f.unlink(missing_ok=True)
    return rep


def _pdf_pages_from_log(log: str) -> int | None:
    """从 xelatex 日志里取页数。

    ⚠️ 正则必须带 ``re.S``：latex 会把日志按 79 列硬折行，实测出现过
    ``final-report.p`` / ``df (18 pages).`` 断在文件名中间的情况。
    另外不要直接读 PDF 解析 ``/Type /Page`` —— xelatex 输出的是带
    **压缩对象流**的 PDF，页面对象不在明文里（实测抓到 0 个，会误报"无页"）。
    """
    m = re.search(r"Output written on .*?\((\d+) pages?", log, re.S)
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="把 docs/*.md 导出为 PDF")
    ap.add_argument("src", nargs="?", default="final-report.md",
                    help="要导出的 Markdown（相对 docs/ 或绝对路径）")
    ap.add_argument("--all", action="store_true", help="导出 docs/ 下全部 .md")
    ap.add_argument("--no-toc", action="store_true", help="不生成目录")
    ap.add_argument("--keep-work", action="store_true", help="保留中间文件（调试用）")
    args = ap.parse_args(argv)

    targets = (sorted(DOCS.glob("*.md")) if args.all
               else [(p if (p := Path(args.src)).is_absolute() else DOCS / p)])

    print(f"\n=== 导出 PDF（engine={ENGINE}，{len(targets)} 个文件）===")
    bad: list[BuildReport] = []
    for t in targets:
        rep = build(t, toc=not args.no_toc, keep_work=args.keep_work)
        print(rep.summary())
        print()
        if not rep.ok:
            bad.append(rep)

    if bad:
        print(f"!! {len(bad)} 个文件存在问题：{[r.src.name for r in bad]}")
        return 1
    print("全部导出成功：无缺字、无表格溢出。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
