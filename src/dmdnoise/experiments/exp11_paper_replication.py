"""实验十一：原文 Figure 1 复现 —— 外部一致性验证。

**动机**：本项目此前全部结论都是"内部自洽"的（五次推翻/更正自己、全部测试、
失败记录、ADR）。但项目自称的目的是"检验 TDMD **强收敛定理**的有限样本表现"，
而做到这一点的唯一证据（能复现原文算例）此前只存在于一个被 gitignore 的
临时脚本里，最终报告只字未提。本实验把它提升为正式实验。

**原文设定**（Hemati et al. 2017, *TCFD* 31(4):349-368, §3 "DMD on a linear system"，
即 Figure 1 的算例）：

  - 状态空间 `C^2`，动力学 `(x1,x2) -> (lam1 x1, lam2 x2)`
  - `(lam1, lam2) = (1.02 e^{0.1i}, 1.04 e^{0.3i})` —— **非共轭**，两者模均略大于 1
  - 观测为随机线性映射 `C^2 -> C^250`（`n = 250`，`r = 2`）
  - 快照对 `m = {100, 200, 500}`
  - **构造**：拼接 `{5, 10, 25}` 段、每段 20 个快照的 ensemble run，
    每段由**不同初态**初始化。原文明确说这是为了"保持各数据集的信噪比一致"。
  - 噪声：加性圆对称复高斯 `(dX, dY) ~ CN(0, 0.05)`
  - 每个 `m` 重复 **200 次**独立噪声实现
  - 结论：TDMD 的谱**基本无偏**、方差**略紧**于标准 DMD

⚠️ **三处原文未规定、但会改变结论的自由度**（详见 `docs/final-report.md` §2.15）：

  1. **随机观测矩阵的尺度**。噪声 `CN(0, 0.05)` 是**绝对**口径，而信号幅度由观测
     矩阵尺度决定 —— 故**有效 SNR 是一个自由参数**。本项目据此把噪声水平做成
     显式扫描轴（`SIGMA2_GRID`），并报告偏差的**标度指数**而非单点数值。
  2. **`dX` 与 `dY` 是否相关**。原文写 `(dX, dY) ~ CN(0, 0.05)`，**未说明两者
     之间的相关性**，而这恰恰决定 TDMD 是否有偏 —— 实测（`m=200`、`sigma^2=0.8`、
     `J=600`）降偏比在 `trajectory`（X/Y 噪声重叠）下为 **0.084**、
     在 `independent`（各自独立）下为 **0.549**，**相差 6.5 倍**。
     只有 `trajectory` 口径与原文"TDMD 基本无偏"的表述一致
     （TDMD 的偏差/MCSE = 2.6，**不可分辨**；`independent` 下为 7.8，可分辨）。
     这也是本项目 ADR-005 采纳的噪声模型，故本实验以 `trajectory` 为主口径。
  3. **拼接方式**。必须**按块**拼接：`X = [X_1|...|X_A]`、`Y = [Y_1|...|Y_A]`。
     若先把各段轨迹首尾相接、再整体切片（`Y = full[:, 1:m+1]`），段边界会产生
     `4%` 左右的"缝接"列对 —— 前一段的**最大**列（`lam^20 x0`）与后一段的
     **最小**列（`x0`）配成一对，`||X_k||/||Y_k||` 由 ~0.97 散到 0.59–4.78。
     实测把**无噪**恢复误差由 `1e-16` 抬到 `7.5e-2`、含噪偏差**放大 360 倍**。
     （`col(Y) ⊆ col(X)` 这个判据**识别不出**它 —— 两种构造下所有列都落在同一
     `r` 维空间里，总能拟合出精确的 `A`。真正的判据是**无噪恢复误差**。）

**本实验的三部分**：

  A. **原文算例复现**：`m in {100,200,500}`（主口径）+ `m=200` 上两种噪声口径的
     对照；报 DMD/TDMD 的谱偏差、离散度、可分辨性与**配对 MAE 差**。
  B. **噪声扫描（机制验证）**：扫 `sigma^2`，拟合 `log|bias| ~ a*log(sigma^2)`。
     **只在偏差可分辨的点上拟合** —— 不可分辨时点估计由 MCSE 主导，斜率纯属噪声。
  C. **交付代码 ≡ 原文公式**：在同一份数据上，把随包交付的 `DMD`/`TDMD`
     与**按原文公式内联实现**的参考算子逐位比对；
     并验证负对照 `tdmd_left_star` 在**原文自身算例**上确实失败、
     `tdmd_left_pinv` 与正确版等价。
     （后一条正是 `docs/algorithm-spec.md` §8.2 要求的"用原文自身算例标定负对照"。）
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from dmdnoise.config import Config, config_fingerprint
from dmdnoise.estimators import DMD, TDMD
from dmdnoise.estimators.variants import tdmd_left_pinv, tdmd_left_star
from dmdnoise.metrics import RESOLVE_COEF, paired_ratio

# --------------------------------------------------------------------------- 原文参数
#: 原文 §3 的离散特征值（**非共轭**，模分别为 1.02 与 1.04）
PAPER_LAMBDA: NDArray = np.array([1.02 * np.exp(0.1j), 1.04 * np.exp(0.3j)])

#: 原文的观测维数（`C^2 -> C^250`）
N_OBS = 250

#: 原文的截断秩
RANK = 2

#: 每段 ensemble run 的快照数
ENSEMBLE = 20

#: `m -> 拼接段数`（原文为保持 SNR 一致而这样选）
M_TO_RUNS = {100: 5, 200: 10, 500: 25}

#: 原文的噪声方差：`E|dX|^2 = E|dY|^2 = 0.05`
NOISE_VAR = 0.05

#: 采样间隔（原文用离散映射 `lam^k`，故取 `dt = 1`）
DT = 1.0

#: 默认的 `m` 网格
M_GRID: tuple[int, ...] = (100, 200, 500)

#: 噪声口径。原文未说明 `dX` 与 `dY` 的相关性 —— 这里两种都测，第一个为**主口径**。
#: `trajectory` 与项目 ADR-005 一致（单条噪声轨迹），且与原文"TDMD 基本无偏"一致。
NOISE_MODES: tuple[str, ...] = ("trajectory", "independent")
NOISE_MODE_PRIMARY = NOISE_MODES[0]

#: 噪声口径对照所用的 `m`
M_MODE_COMPARE = 200


#: 噪声扫描网格（`E|d|^2`）。含原文值 0.05 并向两侧展开。
#: 上界取 3.2 是为了让**两法**在多数点上偏差可分辨（低 `sigma^2` 处 TDMD 的
#: 偏差低于检测限，这本身是"基本无偏"的体现，但不能用于拟合斜率）。
SIGMA2_GRID: tuple[float, ...] = (0.05, 0.2, 0.8, 3.2)

#: 噪声口径对照所用的 `sigma^2`。
#: ⚠️ **必须含一个高噪声点**：在原文噪声水平（0.05）处，两种口径下 TDMD 的偏差
#: **都不可分辨**，因此**判不出口径差异**——不可分辨时只有比值的**上界**有意义。
#: 高噪声点才能给出严格比较：那里 `independent` 的 TDMD 偏差已可分辨，
#: 而 `trajectory` 的比值上界仍远低于它。
MODE_COMPARE_SIGMA2: tuple[float, ...] = (NOISE_VAR, max(SIGMA2_GRID))

#: 噪声扫描所用的 `m`
M_SIGMA_SCAN = 200

#: 默认实现次数
J_DEFAULT = 1500

#: 负对照/等价性检查所用的实现次数
J_CONTROLS = 200

#: 存下的逐实现特征值样本数（供谱散点图 `fig16` 的第 1 面板）
J_SAMPLES = 300

#: 偏差可分辨判据：`|bias| > RESOLVE_COEF * MCSE`（与 `metrics.summarize` 一致）
RESOLVED_THRESHOLD = RESOLVE_COEF


# --------------------------------------------------------------------------- 数据构造
def build_paper_data(m: int, rng: np.random.Generator) -> tuple[NDArray, NDArray]:
    """按原文 §3 构造一份**无噪**快照对 `(X, Y)`。

    **按块拼接**：逐段生成 20 列的 `(X_a, Y_a)`，再横向拼成 `X`、`Y`。
    这样**每一列**都精确满足 `Y[:, k] = A X[:, k]`（`A = C diag(lam) C^+`）。

    ⚠️ 不要改成"先把各段轨迹首尾相接再做整体切片"——那会在段边界产生
    约 4% 的缝接列对，使 `||X_k||/||Y_k||` 由 ~0.97 散到 0.59–4.78，
    把无噪恢复误差抬到 `7.5e-2`、含噪偏差放大 360 倍。
    （见模块文档第 3 条与 `test_seam_construction_breaks_operator_relation`。）
    """
    if m not in M_TO_RUNS:
        raise ValueError(f"原文只给出 m in {sorted(M_TO_RUNS)}，收到 {m}")
    n_runs = M_TO_RUNS[m]
    C = rng.standard_normal((N_OBS, RANK)) + 1j * rng.standard_normal((N_OBS, RANK))
    k = np.arange(ENSEMBLE)
    x_blocks: list[NDArray] = []
    y_blocks: list[NDArray] = []
    for _ in range(n_runs):
        # 每段由【不同初态】初始化（原文原话：different state）
        x0 = rng.standard_normal(RANK) + 1j * rng.standard_normal(RANK)
        B = C * x0[None, :]                                   # (n, r)
        x_blocks.append(B @ (PAPER_LAMBDA[:, None] ** k[None, :]))
        y_blocks.append(B @ (PAPER_LAMBDA[:, None] ** (k[None, :] + 1)))
    X = np.hstack(x_blocks)
    Y = np.hstack(y_blocks)
    if X.shape != (N_OBS, m) or Y.shape != (N_OBS, m):
        raise AssertionError(f"形状异常：{X.shape} / {Y.shape}，期望 {(N_OBS, m)}")
    return X, Y


def _unit_noise(shape: tuple[int, ...], rng: np.random.Generator) -> NDArray:
    """单位方差圆对称复高斯：`E|z|^2 = 1`。"""
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / math.sqrt(2.0)


def noise_pair(X: NDArray, Y: NDArray, m: int, noise_mode: str, sigma2: float,
               rng: np.random.Generator) -> tuple[NDArray, NDArray]:
    """按指定口径生成加噪快照对（**返回噪声本身**，由调用方加到真值上）。

    `independent` —— `dX` 与 `dY` 各自独立抽取
    `trajectory`  —— `dY` 是 `dX` 沿列方向**右移一格**（同一段噪声轨迹）：
                     逐段生成 21 列噪声，`X` 取前 20 列、`Y` 取后 20 列。
                     这与项目 ADR-005 的单条轨迹模型一致。

    两者都保持 `E|d|^2 = sigma2`，故 SNR 标定不受影响。
    """
    root = math.sqrt(sigma2)
    if noise_mode == "independent":
        return root * _unit_noise(X.shape, rng), root * _unit_noise(Y.shape, rng)
    if noise_mode == "trajectory":
        out_x: list[NDArray] = []
        out_y: list[NDArray] = []
        for _ in range(M_TO_RUNS[m]):
            e = _unit_noise((N_OBS, ENSEMBLE + 1), rng)
            out_x.append(e[:, :ENSEMBLE])
            out_y.append(e[:, 1:])
        return root * np.hstack(out_x), root * np.hstack(out_y)
    raise ValueError(f"未知噪声口径 {noise_mode!r}，可选 {NOISE_MODES}")


def true_lambda() -> NDArray:
    """真值特征值，按**角度升序**（等价于按频率升序，即项目的配对顺序）。"""
    return PAPER_LAMBDA[np.argsort(np.angle(PAPER_LAMBDA))]


def _jsonable(obj: Any) -> Any:
    """把 meta 里的复杂对象转成可 JSON 序列化的形式。

    ⚠️ 特征值是复数，`json.dump` 不接受 —— 直接写 meta 会在**跑完全部
    计算之后**抛 `TypeError`（CSV 已落盘而 meta 缺失），故必须在写出前转换。
    复数统一转 `{"re": .., "im": ..}`，保持精度且无歧义。
    """
    if isinstance(obj, complex):
        return {"re": float(obj.real), "im": float(obj.imag)}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _order_by_angle(v: NDArray) -> NDArray:
    """按角度升序排列。

    角度升序与频率升序一致（`dt = 1` 且角度落在 `[0, pi)`），
    故与项目的 `pair_to_truth`（按频率升序按下标配对）等价。
    """
    v = np.asarray(v).ravel()
    return v[np.argsort(np.angle(v))]


# --------------------------------------------------------------------------- 参考实现
def reference_operators(X: NDArray, Y: NDArray, r: int) -> dict[str, NDArray]:
    """**按原文公式内联**实现的算子（用于与随包交付的估计器逐位比对）。

    两阶段写法与原文 Eq. (2)/(9) 一致：先做子空间投影，再做算子识别。

      DMD  ：投影用 `X` 的右奇异向量 `V_r`（只纠 `Y`）—— 偏差来源
      TDMD ：投影用 `Z = [X;Y]` 的右奇异向量（`X`、`Y` 同等对待）
    """
    # --- 标准 DMD ---
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    Vr = Vt[:r, :].conj().T
    Xb = X @ Vr @ Vr.conj().T
    Yb = Y @ Vr @ Vr.conj().T
    Ub, Sb, Vtb = np.linalg.svd(Xb, full_matrices=False)
    A_dmd = (Ub[:, :r].conj().T @ Yb @ Vtb[:r, :].conj().T) @ np.diag(1.0 / Sb[:r])

    # --- TDMD（增广矩阵） ---
    Z = np.vstack([X, Y])
    _, _, Vtz = np.linalg.svd(Z, full_matrices=False)
    Vrz = Vtz[:r, :].conj().T
    Xb2 = X @ Vrz @ Vrz.conj().T
    Yb2 = Y @ Vrz @ Vrz.conj().T
    Ub2, Sb2, Vtb2 = np.linalg.svd(Xb2, full_matrices=False)
    A_tdmd = (Ub2[:, :r].conj().T @ Yb2 @ Vtb2[:r, :].conj().T) @ np.diag(1.0 / Sb2[:r])
    return {"dmd": A_dmd, "tdmd": A_tdmd}


# --------------------------------------------------------------------------- 结果结构
@dataclass
class Exp11Result:
    table: pd.DataFrame          # A：原文算例的偏差/离散度明细
    controls: pd.DataFrame       # C：负对照与"交付代码 vs 原文公式"的等价性
    sigma_scan: pd.DataFrame     # B：噪声扫描明细
    samples: pd.DataFrame        # 逐实现特征值样本（供谱散点图，只读 CSV）
    verdict: pd.DataFrame        # 原文主张是否复现
    fingerprint: str
    meta: dict[str, Any] = field(default_factory=dict)


def _summarise_lambda(evals: NDArray) -> dict[str, float]:
    """把 `(J, RANK)` 的特征值样本汇总为偏差/离散度。"""
    mu = evals.mean(axis=0)
    lam = true_lambda()
    diff = mu - lam
    bias_complex_abs = np.abs(diff)
    bias_abs_signed = np.abs(mu) - np.abs(lam)
    bias_arg = np.angle(mu) - np.angle(lam)
    # 离散度用复数的均方根偏差（含实虚两方向），其 1/sqrt(J) 即 MCSE
    std_complex = np.sqrt(np.mean(np.abs(evals - mu) ** 2, axis=0))
    mcse = std_complex / math.sqrt(evals.shape[0])
    return {
        "mean_abs": np.abs(mu), "mean_arg": np.angle(mu),
        "bias_complex_abs": bias_complex_abs, "bias_abs_signed": bias_abs_signed,
        "bias_arg": bias_arg, "std_complex": std_complex, "mcse": mcse,
        "bias_over_mcse": np.where(mcse > 0, bias_complex_abs / np.maximum(mcse, 1e-300), np.inf),
    }


# --------------------------------------------------------------------------- 主流程
def run(cfg: Config, *, ms: Sequence[int] | None = None,
        sigma2_grid: Sequence[float] | None = None,
        noise_modes: Sequence[str] | None = None,
        j_total: int = J_DEFAULT, j_controls: int = J_CONTROLS,
        j_samples: int = J_SAMPLES,
        progress: Callable[[int, int], None] | None = None) -> Exp11Result:
    """跑实验十一。

    `progress(done, total)` 在每完成一个实现后调用（用于后台进度打印）。
    `j_samples` 控制存下多少个**逐实现特征值样本**（供谱散点图，不额外增加计算）。
    """
    m_list = tuple(ms) if ms is not None else M_GRID
    s2_list = tuple(sigma2_grid) if sigma2_grid is not None else SIGMA2_GRID
    modes = tuple(noise_modes) if noise_modes is not None else NOISE_MODES
    primary = modes[0]
    seed = int(cfg.run.seed)
    lam = true_lambda()

    # 单元格：(m, 噪声口径, sigma^2)。
    # 主口径跑全部 m（在原文噪声水平上）；其余口径在 M_MODE_COMPARE 与
    # MODE_COMPARE_SIGMA2 的组合上跑（含一个高噪声点，用于严格比较口径差异）。
    cells: list[tuple[int, str, float]] = [
        (int(m), primary, NOISE_VAR) for m in m_list]
    cells += [(M_MODE_COMPARE, nm, float(s2))
              for nm in modes[1:] for s2 in MODE_COMPARE_SIGMA2]

    rows: list[dict[str, Any]] = []
    ctrl_rows: list[dict[str, Any]] = []
    scan_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    gates: list[dict[str, Any]] = []
    # 逐实现特征值（供配对 bootstrap）：(m, sigma2, noise_mode, method) -> (J, RANK)
    paired: dict[tuple[int, float, str, str], NDArray] = {}

    # ⚠️ Part B 每个**实现**只计 1 次（内部对全部 sigma^2 各估一次），
    # 故这里是 `+ j_total` 而不是 `+ len(s2_list) * j_total`——
    # 后者会让进度永远到不了 100%（一度如此）。
    total = len(cells) * j_total + j_total
    done = 0

    # ---------------------------------------------------------------- A: 原文算例
    for m, nm, s2_cell in cells:
        X, Y = build_paper_data(m, np.random.default_rng(seed + m))

        # D1 门：无噪退化（两法都应精确复现 lam）。每个 m 只查一次。
        if not any(g["m"] == m for g in gates):
            for method, est in (("dmd", DMD()), ("tdmd", TDMD())):
                ev = _order_by_angle(est.fit(X, Y, RANK, DT).eigenvalues)
                gates.append({"m": m, "method": method,
                              "noiseless_max_err": float(np.abs(ev - lam).max())})

        acc: dict[str, list[NDArray]] = {"dmd": [], "tdmd": [], "star": [], "pinv": []}
        ss = np.random.SeedSequence([seed, m, 11])
        for j, child in enumerate(ss.spawn(j_total)):
            rng = np.random.default_rng(child)
            dX, dY = noise_pair(X, Y, m, nm, s2_cell, rng)
            Xn, Yn = X + dX, Y + dY
            acc["dmd"].append(_order_by_angle(DMD().fit(Xn, Yn, RANK, DT).eigenvalues))
            acc["tdmd"].append(_order_by_angle(TDMD().fit(Xn, Yn, RANK, DT).eigenvalues))
            if j < j_samples and nm == primary and s2_cell == NOISE_VAR:
                for method in ("dmd", "tdmd"):
                    ev = acc[method][-1]
                    for mode in range(RANK):
                        sample_rows.append({
                            "m": m, "sigma2": s2_cell, "noise_mode": nm,
                            "method": method, "mode": mode + 1, "j": j,
                            "re": float(ev[mode].real), "im": float(ev[mode].imag),
                        })
            if j < j_controls:
                acc["star"].append(_order_by_angle(
                    tdmd_left_star(Xn, Yn, RANK, DT).eigenvalues))
                acc["pinv"].append(_order_by_angle(
                    tdmd_left_pinv(Xn, Yn, RANK, DT).eigenvalues))
            done += 1
            if progress is not None:
                progress(done, total)

        # 逐模态明细
        for method in ("dmd", "tdmd"):
            arr = np.asarray(acc[method])
            s = _summarise_lambda(arr)
            paired[(m, s2_cell, nm, method)] = arr
            for mode in range(RANK):
                rows.append({
                    "m": m, "n_runs": M_TO_RUNS[m], "sigma2": s2_cell,
                    "noise_mode": nm, "method": method, "mode": mode + 1,
                    "j_total": j_total,
                    "lam_abs_true": float(np.abs(lam[mode])),
                    "lam_arg_true": float(np.angle(lam[mode])),
                    "mean_abs": float(s["mean_abs"][mode]),
                    "mean_arg": float(s["mean_arg"][mode]),
                    "bias_complex_abs": float(s["bias_complex_abs"][mode]),
                    "bias_abs_signed": float(s["bias_abs_signed"][mode]),
                    "bias_arg": float(s["bias_arg"][mode]),
                    "std_complex": float(s["std_complex"][mode]),
                    "mcse": float(s["mcse"][mode]),
                    "bias_over_mcse": float(s["bias_over_mcse"][mode]),
                    "resolvable": bool(s["bias_over_mcse"][mode] > RESOLVED_THRESHOLD),
                })

        # ------------------------------------------------------------ C: 负对照
        for method in ("star", "pinv"):
            arr = np.asarray(acc[method])
            s = _summarise_lambda(arr)
            # "最优配对"下的偏差：每个估计取到真值的最近距离（对变体最宽容）
            nn = np.mean(np.min(np.abs(arr[:, :, None] - lam[None, None, :]), axis=2), axis=0)
            for mode in range(RANK):
                ctrl_rows.append({
                    "m": m, "noise_mode": nm, "method": method, "mode": mode + 1,
                    "j_total": len(arr),
                    "bias_complex_abs": float(s["bias_complex_abs"][mode]),
                    "bias_over_mcse": float(s["bias_over_mcse"][mode]),
                    "nearest_truth_dist": float(nn[mode]),
                    "std_complex": float(s["std_complex"][mode]),
                })
        # 交付代码 vs 原文公式：逐位比对（用无噪数据，避免噪声放大数值差）
        ref = reference_operators(X, Y, RANK)
        for method, est in (("dmd", DMD()), ("tdmd", TDMD())):
            mine = _order_by_angle(est.fit(X, Y, RANK, DT).eigenvalues)
            theirs = _order_by_angle(np.linalg.eigvals(ref[method]))
            ctrl_rows.append({
                "m": m, "noise_mode": nm, "method": f"ref_vs_shipped[{method}]",
                "mode": 0, "j_total": 1,
                "bias_complex_abs": float(np.abs(mine - theirs).max()),
                "bias_over_mcse": float("nan"),
                "nearest_truth_dist": float("nan"),
                "std_complex": float("nan"),
            })
        # star/pinv 与正确 TDMD 的逐实现差（验证"pinv 等价、star 不等价"）。
        # 负对照只跑前 j_controls 次实现，故只与 tdmd 的前若干次对齐比较。
        n_ctrl = len(acc["star"])
        tdmd_head = np.asarray(acc["tdmd"])[:n_ctrl]
        for method in ("star", "pinv"):
            d = float(np.abs(np.asarray(acc[method]) - tdmd_head).max())
            ctrl_rows.append({
                "m": m, "noise_mode": nm, "method": f"diff_vs_tdmd[{method}]",
                "mode": 0, "j_total": len(acc[method]),
                "bias_complex_abs": d, "bias_over_mcse": float("nan"),
                "nearest_truth_dist": float("nan"), "std_complex": float("nan"),
            })

    # ---------------------------------------------------------------- B: 噪声扫描
    Xs, Ys = build_paper_data(M_SIGMA_SCAN, np.random.default_rng(seed + M_SIGMA_SCAN))
    per_sigma: dict[tuple[float, str], list[NDArray]] = {
        (s2, meth): [] for s2 in s2_list for meth in ("dmd", "tdmd")}
    ss = np.random.SeedSequence([seed, M_SIGMA_SCAN, 12])
    for j, child in enumerate(ss.spawn(j_total)):
        rng = np.random.default_rng(child)
        # 同一批**单位方差**噪声在各 sigma 间共用（公共随机数，便于比较标度）
        ux, uy = noise_pair(Xs, Ys, M_SIGMA_SCAN, primary, 1.0, rng)
        for s2 in s2_list:
            root = math.sqrt(s2)
            Xn, Yn = Xs + root * ux, Ys + root * uy
            per_sigma[(s2, "dmd")].append(
                _order_by_angle(DMD().fit(Xn, Yn, RANK, DT).eigenvalues))
            per_sigma[(s2, "tdmd")].append(
                _order_by_angle(TDMD().fit(Xn, Yn, RANK, DT).eigenvalues))
        done += 1
        if progress is not None:
            progress(done, total)

    slopes: list[dict[str, Any]] = []
    for s2 in s2_list:
        for method in ("dmd", "tdmd"):
            arr = np.asarray(per_sigma[(s2, method)])
            s = _summarise_lambda(arr)
            paired[(M_SIGMA_SCAN, s2, primary, method)] = arr
            for mode in range(RANK):
                scan_rows.append({
                    "sigma2": s2, "m": M_SIGMA_SCAN, "noise_mode": primary,
                    "method": method, "mode": mode + 1, "j_total": j_total,
                    "bias_complex_abs": float(s["bias_complex_abs"][mode]),
                    "bias_abs_signed": float(s["bias_abs_signed"][mode]),
                    "std_complex": float(s["std_complex"][mode]),
                    "mcse": float(s["mcse"][mode]),
                    "bias_over_mcse": float(s["bias_over_mcse"][mode]),
                    "resolvable": bool(s["bias_over_mcse"][mode] > RESOLVED_THRESHOLD),
                })
    # 标度指数：log|bias| = a * log(sigma^2) + b
    # ⚠️ **只可用「偏差可分辨」的点**：偏差不可分辨时其点估计由 MCSE 主导，
    # 拟合出的斜率纯属噪声（实测会退化到 ~1，掩盖真实标度）。
    scan_out = pd.DataFrame(scan_rows)
    for (method, mode), g in scan_out.groupby(["method", "mode"]):
        g = g[np.isfinite(g["bias_complex_abs"]) & (g["bias_complex_abs"] > 0)]
        gres = g[g["resolvable"].astype(bool)]
        use = gres if gres["sigma2"].nunique() >= 3 else g
        if use["sigma2"].nunique() >= 3:
            a, b = np.polyfit(np.log(use["sigma2"]), np.log(use["bias_complex_abs"]), 1)
        else:
            a, b = float("nan"), float("nan")
        slopes.append({"method": method, "mode": int(mode),
                       "slope_log_bias_vs_log_sigma2": float(a), "intercept": float(b),
                       "n_sigma2": int(use["sigma2"].nunique()),
                       "n_sigma2_total": int(g["sigma2"].nunique()),
                       "n_resolvable": int(gres["sigma2"].nunique())})

    table = pd.DataFrame(rows)
    verdict = _verdict(table, scan_out, gates, paired, lam)
    return Exp11Result(
        table=table, controls=pd.DataFrame(ctrl_rows), sigma_scan=scan_out,
        samples=pd.DataFrame(sample_rows), verdict=verdict,
        fingerprint=config_fingerprint(cfg),
        meta={
            "experiment": "exp11",
            "source": "Hemati et al. 2017 TCFD 31(4):349-368 sec.3 / Figure 1",
            # 复数不可 JSON 序列化，统一存 [re, im]
            "paper_lambda": [[float(v.real), float(v.imag)] for v in PAPER_LAMBDA],
            "n_obs": N_OBS, "rank": RANK, "ensemble": ENSEMBLE,
            "m_to_runs": {str(k): v for k, v in M_TO_RUNS.items()},
            "noise_var": NOISE_VAR, "m_grid": list(m_list),
            "noise_modes": list(modes), "noise_mode_primary": primary,
            "m_mode_compare": M_MODE_COMPARE,
            "mode_compare_sigma2": list(MODE_COMPARE_SIGMA2),
            "sigma2_grid": list(s2_list), "j_total": j_total,
            "j_controls": j_controls, "j_samples": j_samples,
            "gates": gates,
            "slopes": slopes,
        },
    )


# --------------------------------------------------------------------------- 裁决
def _verdict(table: pd.DataFrame, scan: pd.DataFrame,
             gates: list[dict[str, Any]],
             paired: dict[tuple[int, float, str, str], NDArray],
             lam: NDArray) -> pd.DataFrame:
    """逐 (m, sigma2, 噪声口径) 判定原文的两条主张是否复现。"""
    out: list[dict[str, Any]] = []
    gate_ok = all(g["noiseless_max_err"] < 1e-10 for g in gates)

    def _max_bias(df: pd.DataFrame, method: str) -> float:
        g = df[df["method"] == method]
        return float(g["bias_complex_abs"].max()) if not g.empty else float("nan")

    def _worst_over_mcse(df: pd.DataFrame, method: str) -> float:
        g = df[df["method"] == method]
        return float(g["bias_over_mcse"].min()) if not g.empty else float("nan")

    # A（原文算例）与 B（噪声扫描）在 `sigma^2 == NOISE_VAR` 处重叠。
    # 去重时**保留 A 的那一行**（同一 (m, sigma2) 上的两次独立抽样等价，
    # 但混在一起会让"取最保守偏差"变成在 4 个估计上取 max，语义不清）。
    # ⚠️ 不能用「s2 == NOISE_VAR 就去 table 里找」来选来源 —— 若该 `m` 不在
    # `m_list`（例如只跑扫描的 m=200），整行会被静默丢掉（已修过一次）。
    key = ["m", "sigma2", "noise_mode", "method", "mode"]
    base = table.assign(来源="A")
    extra = scan.merge(table[key], on=key, how="left", indicator=True)
    extra = extra[extra["_merge"] == "left_only"].drop(columns="_merge")
    frames = pd.concat([base, extra.assign(来源="B")], ignore_index=True)
    if frames.empty:
        return frames
    for (m, s2, nm), sub in frames.groupby(["m", "sigma2", "noise_mode"]):
        m, s2 = int(m), float(s2)
        b_dmd, b_tdmd = _max_bias(sub, "dmd"), _max_bias(sub, "tdmd")
        ratio = b_tdmd / b_dmd if b_dmd > 0 else float("nan")

        var_ratio = var_lo = var_hi = float("nan")
        mae_dmd = mae_tdmd = mae_diff = mae_lo = mae_hi = float("nan")
        var_delta = var_sum = float("nan")
        crn_ok = tdmd_closer = False
        a = paired.get((m, s2, nm, "dmd"))
        b = paired.get((m, s2, nm, "tdmd"))
        if a is not None and b is not None and a.shape == b.shape and a.shape[0] >= 20:
            try:
                # 口径与 exp1 一致：**std_TDMD / std_DMD**，< 1 表示 TDMD 离散度更小
                pr = paired_ratio(np.abs(b).ravel(), np.abs(a).ravel(), stat="std",
                                  bootstrap=2000)
                var_ratio, (var_lo, var_hi) = pr.point, pr.ci
                var_delta, var_sum = pr.var_delta, pr.var_sum
                crn_ok = bool(pr.paired_effective)
            except Exception:                     # 分母为零等退化情形
                pass
            # 配对 MAE 差：逐实现的 |lam_hat - lam_true| 之差。
            # 这是"谁更接近真值"最灵敏的检验 —— 比分别比较两个偏差量级稳健得多
            # （公共随机数下差值的方差远小于各自方差）。
            ea, eb = np.abs(a - lam), np.abs(b - lam)
            diff = (eb - ea).ravel()
            if diff.size >= 20:
                gen = np.random.default_rng(0)
                idx = gen.integers(0, diff.size, size=(2000, diff.size))
                boot = diff[idx].mean(axis=1)
                mae_dmd, mae_tdmd = float(ea.mean()), float(eb.mean())
                mae_diff = float(diff.mean())
                mae_lo, mae_hi = (float(v) for v in np.percentile(boot, [2.5, 97.5]))
                tdmd_closer = bool(mae_hi < 0.0)

        out.append({
            "来源": str(sub["来源"].iloc[0]),
            "噪声口径": nm, "m": m, "sigma2": s2,
            "DMD偏差": b_dmd, "TDMD偏差": b_tdmd, "偏差比": ratio,
            "DMD可分辨": bool(_worst_over_mcse(sub, "dmd") > RESOLVED_THRESHOLD),
            "TDMD可分辨": bool(_worst_over_mcse(sub, "tdmd") > RESOLVED_THRESHOLD),
            "方差比_TDMD_over_DMD": var_ratio, "方差比下界": var_lo, "方差比上界": var_hi,
            "MAE_DMD": mae_dmd, "MAE_TDMD": mae_tdmd,
            "MAE差": mae_diff, "MAE差下界": mae_lo, "MAE差上界": mae_hi,
            "TDMD更接近真值": tdmd_closer,
            # 公共随机数自检（规格 §6.2）：Var(delta) 应远小于 Var(a)+Var(b)
            "Var_delta": var_delta, "Var_sum": var_sum, "公共随机数生效": crn_ok,
            "无噪退化门": gate_ok,
        })
    df = pd.DataFrame(out)
    return df.sort_values(["m", "sigma2", "噪声口径"]).reset_index(drop=True)


def summary(res: Exp11Result) -> str:
    """人类可读的一句话小结（供入口打印）。"""
    gates = res.meta.get("gates", [])
    gate_ok = all(g["noiseless_max_err"] < 1e-10 for g in gates)
    slopes = res.meta.get("slopes", [])

    def _mean_slope(meth: str) -> float:
        v = [s["slope_log_bias_vs_log_sigma2"] for s in slopes if s["method"] == meth]
        return float(np.nanmean(v)) if v else float("nan")

    v = res.verdict
    n_unres = int((~v["TDMD可分辨"].astype(bool)).sum()) if not v.empty else 0
    return (f"无噪退化门 {'通过' if gate_ok else '未通过'}；"
            f"偏差标度指数 DMD={_mean_slope('dmd'):.2f} / TDMD={_mean_slope('tdmd'):.2f}；"
            f"TDMD 偏差不可分辨的组合 {n_unres}/{len(v)}")
