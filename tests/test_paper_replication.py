"""实验十一（原文 Figure 1 复现）的正确性测试。

核心待验证的性质：

  1. **原文参数正确**：`lam = (1.02 e^{0.1i}, 1.04 e^{0.3i})`、`n=250`、`r=2`、
     `m = {100,200,500}` 对应 `{5,10,25}` 段、`CN(0,0.05)`。
  2. ★ **拼接必须按块**：无噪时 DMD/TDMD 必须精确复现 `lam`（<1e-10）。
     若改成"先拼轨迹再切片"，段边界会产生缝接列对、破坏算子关系，
     无噪也就复现不出 `lam` —— 该反例已固化为回归。
  3. ★ **交付代码 ≡ 原文公式**：随包交付的 `DMD`/`TDMD` 与内联参考实现
     在同一数据上给出相同特征值。
  4. ★ **负对照在原文自身算例上有效**：`tdmd_left_star` 必须严重失败、
     `tdmd_left_pinv` 必须与正确版等价 —— 这正是 `algorithm-spec.md` §8.2
     要求的"用原文自身算例标定负对照"。
  5. **噪声方差标定**：`E|d|^2` 必须等于设定值（否则 SNR 口径失真）。
  6. **裁决行的来源选择**：`sigma2 == NOISE_VAR` 的行不能因为该 `m` 不在
     `ms` 中而被静默丢掉（已修过一次的缺陷）。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dmdnoise.config import Config
from dmdnoise.estimators import DMD, TDMD
from dmdnoise.estimators.variants import tdmd_left_pinv, tdmd_left_star
from dmdnoise.experiments.exp11_paper_replication import (
    ENSEMBLE,
    M_MODE_COMPARE,
    NOISE_MODE_PRIMARY,
    NOISE_MODES,
    SIGMA2_GRID,
    noise_pair,
    M_SIGMA_SCAN,
    M_TO_RUNS,
    N_OBS,
    NOISE_VAR,
    PAPER_LAMBDA,
    RANK,
    _order_by_angle,
    _unit_noise,
    build_paper_data,
    reference_operators,
    run,
    true_lambda,
)


# --------------------------------------------------------------------------- 参数保真
def test_paper_parameters_match_source() -> None:
    """原文 §3 的参数必须逐项对上（数值来自原文，不得"顺手改"）。"""
    lam = true_lambda()
    np.testing.assert_allclose(np.abs(lam), [1.02, 1.04], rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.angle(lam), [0.1, 0.3], rtol=0, atol=1e-12)
    assert (N_OBS, RANK, ENSEMBLE) == (250, 2, 20)
    assert M_TO_RUNS == {100: 5, 200: 10, 500: 25}
    assert NOISE_VAR == 0.05
    # 原文的特征值是【非共轭】的（两模态模不同）—— 若写成共轭对就错了
    assert not np.isclose(lam[0], lam[1].conj())


def test_lambda_order_is_by_angle() -> None:
    """真值的排序必须与估计器的频率升序一致（项目配对约定）。"""
    assert list(np.angle(true_lambda())) == sorted(np.angle(PAPER_LAMBDA))
    rng = np.random.default_rng(0)
    v = rng.standard_normal(4) + 1j * rng.standard_normal(4)
    assert list(np.angle(_order_by_angle(v))) == sorted(np.angle(v))


# --------------------------------------------------------------------------- 构造
def _column_space_defect(X: np.ndarray, Y: np.ndarray, rank: int) -> float:
    """`col(Y)` 到 `col(X)` 的相对投影残差。

    `Y = A X` 的充要条件是 `col(Y) ⊆ col(X)`。按其主 `rank` 维子空间做投影，
    判据良态；而 `Y @ pinv(X)` 这种写法在秩亏损矩阵上只能给到 `1e-5` 量级，
    不足以区分"有无缝接"。
    """
    Ux, _, _ = np.linalg.svd(X, full_matrices=False)
    P = Ux[:, :rank] @ Ux[:, :rank].conj().T
    return float(np.linalg.norm(Y - P @ Y) / np.linalg.norm(Y))


@pytest.mark.parametrize("m", sorted(M_TO_RUNS))
def test_build_paper_data_shape_and_rank(m: int) -> None:
    X, Y = build_paper_data(m, np.random.default_rng(0))
    assert X.shape == (N_OBS, m) and Y.shape == (N_OBS, m)
    assert np.linalg.matrix_rank(X) == RANK
    assert np.linalg.matrix_rank(Y) == RANK
    # 无噪数据必须精确满足 Y = A X（等价于 col(Y) ⊆ col(X)）
    assert _column_space_defect(X, Y, RANK) < 1e-12


def test_rejects_unknown_m() -> None:
    with pytest.raises(ValueError):
        build_paper_data(150, np.random.default_rng(0))


def test_block_construction_recovers_lambda_exactly() -> None:
    """★ 块拼接 + 无噪 ⇒ DMD/TDMD 必须**精确**复现 `lam`（D1 门）。"""
    lam = true_lambda()
    X, Y = build_paper_data(100, np.random.default_rng(0))
    for est in (DMD(), TDMD()):
        ev = _order_by_angle(est.fit(X, Y, RANK, 1.0).eigenvalues)
        assert np.abs(ev - lam).max() < 1e-10, (type(est).__name__, ev)


def test_seam_construction_breaks_operator_relation() -> None:
    """★ 反例：**先拼轨迹再整体切片**会产生缝接列对，破坏算子关系。

    该反例正是 `build_paper_data` 必须按块拼接的原因。实测（`m=100`、5 段）：

    | 构造 | 无噪恢复误差 | 含噪 `|bias|`（J=800） | `||X_k||/||Y_k||` 范围 |
    |---|---|---|---|
    | 块拼接 | `2.9e-16` | `1.5e-4`（MCSE 比 4.2） | 0.966–0.980 |
    | 缝接 | **`7.5e-2`** | **`5.5e-2`**（MCSE 比 **1568**） | **0.591–4.783** |

    **机制**：缝接把前一段的**最大**列（`lam^20 x0`）与后一段的**最小**列
    （`x0`）配成一对，列范数比从 ~0.97 散到 0.59–4.78，相当于在高杠杆位置
    掺入外点。**含噪偏差被放大两个数量级以上。**

    ⚠️ **注意判据的选择**：`col(Y) ⊆ col(X)`（列空间重合）**识别不出**缝接 ——
    两种构造下所有列都落在同一个 `r` 维空间中，总能拟合出一个精确的 `A`。
    真正的判据是**无噪恢复误差**（即项目 D1 门）。
    """
    lam = true_lambda()
    lams = PAPER_LAMBDA[np.argsort(np.angle(PAPER_LAMBDA))]
    rng = np.random.default_rng(0)
    C = rng.standard_normal((N_OBS, RANK)) + 1j * rng.standard_normal((N_OBS, RANK))
    k = np.arange(ENSEMBLE + 1)
    cols = []
    for _ in range(M_TO_RUNS[100]):
        x0 = rng.standard_normal(RANK) + 1j * rng.standard_normal(RANK)
        cols.append((C * x0[None, :]) @ (lams[:, None] ** k[None, :]))
    full = np.concatenate(cols, axis=1)
    X_seam, Y_seam = full[:, :100], full[:, 1:101]
    assert Y_seam.shape == (N_OBS, 100)

    # 两种构造下 col(Y) 都落在 col(X) 内 —— 故该判据无区分力（记录在案）
    assert _column_space_defect(X_seam, Y_seam, RANK) < 1e-12

    # 真正的判据：无噪恢复误差
    ev_seam = _order_by_angle(TDMD().fit(X_seam, Y_seam, RANK, 1.0).eigenvalues)
    assert np.abs(ev_seam - lam).max() > 1e-3, "缝接本应破坏无噪恢复，实测没有"

    X_ok, Y_ok = build_paper_data(100, np.random.default_rng(0))
    ev_ok = _order_by_angle(TDMD().fit(X_ok, Y_ok, RANK, 1.0).eigenvalues)
    assert np.abs(ev_ok - lam).max() < 1e-12

    # 机制：缝接使 ||X_k||/||Y_k|| 的散布急剧变大
    def _ratio_spread(X, Y):
        r = np.linalg.norm(X, axis=0) / np.linalg.norm(Y, axis=0)
        return float(r.max() / r.min())
    assert _ratio_spread(X_seam, Y_seam) > 3.0
    assert _ratio_spread(X_ok, Y_ok) < 1.1


# --------------------------------------------------------------------------- 噪声口径
def test_noise_modes_are_registered() -> None:
    """两种口径都在册，且主口径为 `trajectory`（与 ADR-005 及原文表述一致）。"""
    assert NOISE_MODES == ("trajectory", "independent")
    assert NOISE_MODE_PRIMARY == "trajectory"


@pytest.mark.parametrize("mode", ["trajectory", "independent"])
def test_noise_pair_variance_calibration(mode: str) -> None:
    """两种口径都必须满足 `E|d|^2 = sigma2`（否则 SNR 标定失真）。"""
    X, Y = build_paper_data(100, np.random.default_rng(0))
    for s2 in (0.05, 0.8):
        dX, dY = noise_pair(X, Y, 100, mode, s2, np.random.default_rng(1))
        assert dX.shape == X.shape and dY.shape == Y.shape
        for d in (dX, dY):
            assert abs(float(np.mean(np.abs(d) ** 2)) - s2) / s2 < 0.05


def test_trajectory_noise_is_shared_between_X_and_Y() -> None:
    """★ `trajectory` 口径下 `dY` 必须是 `dX` 沿列右移一格（同一段噪声轨迹）。

    这正是原文**未规定**、却决定 TDMD 是否有偏的那个自由度：
    `independent` 下两法降偏比 ~0.55，`trajectory` 下 ~0.08（相差约 6.5 倍）。
    此处把该口径的构造本身固定下来。
    """
    m = 100
    X, Y = build_paper_data(m, np.random.default_rng(0))
    dX, dY = noise_pair(X, Y, m, "trajectory", 0.05, np.random.default_rng(2))

    n_runs = M_TO_RUNS[m]
    keep = np.ones(m, dtype=bool)
    keep[ENSEMBLE - 1::ENSEMBLE] = False          # 段尾无"下一列"可比
    idx = np.arange(m)[keep]
    np.testing.assert_array_equal(dY[:, idx], dX[:, idx + 1])

    # 段尾处**不**相等（跨段不共享），否则说明构造退化成整条轨迹
    ends = np.arange(ENSEMBLE - 1, m, ENSEMBLE)[:-1]
    assert not np.array_equal(dY[:, ends], dX[:, ends + 1])
    assert n_runs == M_TO_RUNS[m]


def test_independent_noise_is_not_shared() -> None:
    """`independent` 口径下 `dY` 与 `dX` 的位移**不应**相关（否则口径名不符实）。"""
    m = 100
    X, Y = build_paper_data(m, np.random.default_rng(0))
    dX, dY = noise_pair(X, Y, m, "independent", 0.05, np.random.default_rng(2))
    assert not np.allclose(dY[:, :-1], dX[:, 1:])


def test_unknown_noise_mode_rejected() -> None:
    X, Y = build_paper_data(100, np.random.default_rng(0))
    with pytest.raises(ValueError):
        noise_pair(X, Y, 100, "bogus", 0.05, np.random.default_rng(0))


def test_noise_variance_calibration() -> None:
    """`_unit_noise` 必须给 `E|z|^2 = 1`；乘 `sqrt(s2)` 后即 `E|d|^2 = s2`。"""
    rng = np.random.default_rng(0)
    z = _unit_noise((200, 400), rng)
    assert abs(float(np.mean(np.abs(z) ** 2)) - 1.0) < 0.01
    for s2 in (0.0125, 0.05, 0.8):
        d = math.sqrt(s2) * z
        assert abs(float(np.mean(np.abs(d) ** 2)) - s2) / s2 < 0.02


# --------------------------------------------------------------------------- 代码路径
def test_reference_matches_shipped_estimators() -> None:
    """★ 交付代码与**按原文公式内联实现**的参考算子必须给出相同特征值。

    这是 exp11 的核心对外主张：随包交付的实现在原文算例上算的就是原文的量。
    """
    for m in (100, 200):
        X, Y = build_paper_data(m, np.random.default_rng(1))
        ref = reference_operators(X, Y, RANK)
        for name, est in (("dmd", DMD()), ("tdmd", TDMD())):
            mine = _order_by_angle(est.fit(X, Y, RANK, 1.0).eigenvalues)
            theirs = _order_by_angle(np.linalg.eigvals(ref[name]))
            assert np.abs(mine - theirs).max() < 1e-10, (m, name)


def test_left_pinv_is_equivalent_but_left_star_is_not() -> None:
    """★ 负对照必须在**原文自身算例**上标定（`algorithm-spec.md` §8.2 的要求）。

      `tdmd_left_pinv` —— 等价变形，**不能**作负对照（曾导致误判）
      `tdmd_left_star` —— 确实失败，**可以**作负对照
    """
    rng = np.random.default_rng(3)
    X, Y = build_paper_data(200, rng)
    nX, nY = _unit_noise(X.shape, rng), _unit_noise(Y.shape, rng)
    Xn, Yn = X + math.sqrt(NOISE_VAR) * nX, Y + math.sqrt(NOISE_VAR) * nY

    correct = _order_by_angle(TDMD().fit(Xn, Yn, RANK, 1.0).eigenvalues)
    pinv = _order_by_angle(tdmd_left_pinv(Xn, Yn, RANK, 1.0).eigenvalues)
    star = _order_by_angle(tdmd_left_star(Xn, Yn, RANK, 1.0).eigenvalues)

    assert np.abs(pinv - correct).max() < 1e-10, "pinv 应与正确版等价"
    lam = true_lambda()
    assert np.abs(star - lam).max() > 0.1, "star 应在原文算例上明显失败"

    # star 的失败必须是【系统性偏错】而非【方差大】：
    # 跨实现的偏差必须远超跨实现的离散度（否则只是噪声大，不构成坏实现）
    rng2 = np.random.default_rng(7)
    stars = []
    for _ in range(20):
        nX2, nY2 = _unit_noise(X.shape, rng2), _unit_noise(Y.shape, rng2)
        Xn2 = X + math.sqrt(NOISE_VAR) * nX2
        Yn2 = Y + math.sqrt(NOISE_VAR) * nY2
        stars.append(_order_by_angle(tdmd_left_star(Xn2, Yn2, RANK, 1.0).eigenvalues))
    stars = np.asarray(stars)
    disp = float(np.sqrt(np.mean(np.abs(stars - stars.mean(axis=0)) ** 2)))
    bias = float(np.abs(stars.mean(axis=0) - lam).max())
    assert bias > 50 * disp, (bias, disp)


# --------------------------------------------------------------------------- 端到端
@pytest.fixture(scope="module")
def small_run():
    """小规模完整跑一次（模块级缓存，避免每个测试重跑）。"""
    cfg = Config()
    return run(cfg, ms=(100,), sigma2_grid=(0.05, 0.8), j_total=60, j_controls=30)


def test_noiseless_gate_passes(small_run) -> None:
    assert all(g["noiseless_max_err"] < 1e-10 for g in small_run.meta["gates"])


def test_controls_table_has_equivalence_rows(small_run) -> None:
    c = small_run.controls
    for tag in ("ref_vs_shipped[dmd]", "ref_vs_shipped[tdmd]"):
        row = c[c["method"] == tag]
        assert not row.empty, tag
        assert float(row["bias_complex_abs"].iloc[0]) < 1e-10, tag
    for tag in ("diff_vs_tdmd[star]", "diff_vs_tdmd[pinv]"):
        assert not c[c["method"] == tag].empty, tag
    pinv = float(c[c["method"] == "diff_vs_tdmd[pinv]"]["bias_complex_abs"].iloc[0])
    star = float(c[c["method"] == "diff_vs_tdmd[star]"]["bias_complex_abs"].iloc[0])
    assert pinv < 1e-10 and star > 0.1


def test_verdict_covers_every_cell(small_run) -> None:
    """★ 回归：裁决表必须覆盖**每一个** (m, sigma2, 噪声口径) 单元格。

    A 部分给出 `m in ms` 的行、B 部分给出 `m = M_SIGMA_SCAN` 的行，两者在
    `sigma2 == NOISE_VAR` 处重叠、去重时保留 A。若按"`s2 == NOISE_VAR` 就去
    table 里找"选来源，而该 `m` 又不在 `ms` 里，整行会被**静默丢掉**——已修过一次。
    """
    v = small_run.verdict
    assert not v.empty
    assert M_SIGMA_SCAN != 100, "本断言依赖 B 的 m 与 ms 不重合"
    # A：m=100 在原文噪声水平上必须有行
    assert ((v["m"] == 100) & np.isclose(v["sigma2"], NOISE_VAR)).any()
    # B：M_SIGMA_SCAN 必须在**全部**扫描到的 sigma2 上都有行（含与 A 重叠的那一点）
    need = {round(float(s), 6) for s in small_run.sigma_scan["sigma2"].unique()}
    have = {round(float(s), 6) for s in v[v["m"] == M_SIGMA_SCAN]["sigma2"].unique()}
    assert need <= have, (need - have)
    # 口径轴也不得丢失
    assert set(v["噪声口径"]) == {"trajectory", "independent"}


def test_crn_paired_effective(small_run) -> None:
    """公共随机数自检：`Var(delta)` 必须远小于 `Var(a)+Var(b)`。"""
    v = small_run.verdict.dropna(subset=["Var_delta", "Var_sum"])
    assert not v.empty
    assert (v["公共随机数生效"]).all()
    assert (v["Var_delta"] < 0.1 * v["Var_sum"]).all()


@pytest.mark.slow
def test_noise_mode_changes_debiasing_ratio() -> None:
    """★ 原文未规定的 `dX`/`dY` 相关性会显著改变降偏比。

    **必须在高噪声点上比较**：在原文噪声水平（`sigma^2 = 0.05`）处，
    两种口径下 TDMD 的偏差都**不可分辨**，此时只有比值的上界有意义，
    判不出口径差异。高噪声点才能给出严格结论。

    断言（`m = M_MODE_COMPARE`、`sigma^2 = max(SIGMA2_GRID)`）：
      - `trajectory` 的降偏比明显小于 1（且 TDMD 的偏差不可分辨）
      - `independent` 的降偏比显著更大
    """
    s2_hi = max(SIGMA2_GRID)
    cfg = Config()
    res = run(cfg, ms=(M_MODE_COMPARE,), sigma2_grid=(s2_hi,),
              j_total=600, j_controls=20,
              noise_modes=("trajectory", "independent"))
    v = res.verdict[(res.verdict["m"] == M_MODE_COMPARE)
                    & np.isclose(res.verdict["sigma2"], s2_hi)]
    r = dict(zip(v["噪声口径"], v["偏差比"]))
    assert set(r) == {"trajectory", "independent"}, r
    assert r["trajectory"] < 0.3, r
    assert r["independent"] > 0.4, r
    assert r["independent"] / r["trajectory"] > 2.0, r
    # 在原文噪声水平上，两口径都应表现为"TDMD 基本无偏"（都不可分辨）
    v0 = res.verdict[(res.verdict["m"] == M_MODE_COMPARE)
                     & np.isclose(res.verdict["sigma2"], NOISE_VAR)]
    assert not v0.empty
    assert not v0["TDMD可分辨"].astype(bool).any(), v0[["噪声口径", "TDMD可分辨"]]


@pytest.mark.slow
def test_bias_scaling_exponents_are_both_unit() -> None:
    """★ 机制验证：`|bias| ~ (sigma^2)^a`，实测**两法的 `a` 都接近 1**。

    ⚠️ 这条断言**修正了实现前的一个错误预期**。原预期是 DMD 的偏差为
    `O(sigma^2)`（`a=1`）、TDMD 去偏后只余 `O(sigma^4)`（`a=2`）。
    实测在原文算例上**两法的 `a` 都约为 1**，即 TDMD 把 `O(sigma^2)` 的**系数**
    削掉了约一个数量级（降偏比 0.09–0.21），但**没有改变标度**。

    故原文"TDMD 基本无偏"应读作**"偏差降到检测限之下"**，而不是"偏差恒为零"。
    本测试把这一实测事实固定下来，防止有人再按 `a=2` 的预期去"修正"代码。

    该判据**不依赖原文未给出的观测矩阵尺度**，故比在单一 SNR 上比较更可靠。
    """
    cfg = Config()
    res = run(cfg, ms=(M_MODE_COMPARE,), j_total=150, j_controls=20)
    sl = {r["method"]: r["slope_log_bias_vs_log_sigma2"] for r in res.meta["slopes"]}
    assert 0.6 < sl["dmd"] < 1.5, sl
    # TDMD 的斜率同样约 1（**不是 2**）；用较宽区间容忍低噪声点的噪声主导
    assert sl["tdmd"] < 1.6, sl
    # 且 TDMD 的偏差系数明显更小。
    # ⚠️ 只看**主噪声口径**与**两法偏差都可分辨**的高噪声点：`independent` 口径下
    # 降偏比本身就在 0.5 以上（正是该口径的问题所在），混进来会把断言搞错。
    s2_all = sorted(res.sigma_scan["sigma2"].unique())
    v = res.verdict[(res.verdict["噪声口径"] == NOISE_MODE_PRIMARY)
                    & np.isclose(res.verdict["sigma2"], max(s2_all))]
    assert not v.empty
    assert float(v["偏差比"].max()) < 0.25, v[["m", "sigma2", "偏差比"]]
