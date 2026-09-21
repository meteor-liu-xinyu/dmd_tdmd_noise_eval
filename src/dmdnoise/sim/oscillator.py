"""仿真层：双模态阻尼振荡系统的真值轨迹生成。

两个通道的构造方式**不可互换**（见 docs/algorithm-spec.md §2.1）：

  real    —— 4 维实 LTI 状态空间实现，使 Y = A_true @ X 精确成立，r_true = 4
  complex —— 复值解析信号 x_j = Σ c_k v_k μ_k^j，r_true = 2

若实值通道误用直接叠加式 `Σ v_k cos(ω_d t + φ_k) e^{-ζωt}`，
则 Y = A X 不精确成立、任何秩的 DMD 都无法恢复真值（实测无噪误差达 7.3）。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from dmdnoise.config import Config, ConstraintResult, check_c6

LOG = logging.getLogger(__name__)

CHANNELS = ("real", "complex", "hankel")


class SimulatorError(ValueError):
    """仿真器输入非法。"""


def _amp_scale(n_modes: int, amp_ratio: float) -> NDArray:
    """各模态的能量缩放因子，使 E_max/E_min = amp_ratio 且几何均值 = 1。

    amp_ratio = 1 时退化为全 1（平衡态）。
    """
    if amp_ratio <= 0.0:
        raise SimulatorError("amp_ratio 必须为正")
    if n_modes == 1 or amp_ratio == 1.0:
        return np.ones(n_modes)
    exps = np.array([(n_modes - 1 - 2 * k) / (2.0 * (n_modes - 1)) for k in range(n_modes)])
    return amp_ratio**exps


@dataclass(frozen=True)
class System:
    """一次系统构造的结果（固定系统实例 + 真值轨迹）。"""

    channel: str
    X: NDArray
    Y: NDArray
    f_true: NDArray          # (n_modes,) 阻尼频率，Hz，升序
    mu_true: NDArray         # (n_modes,) 离散特征值，与 f_true 同序
    rank_true: int
    mode_energy: NDArray     # (n_modes,) 逐模态可观测能量
    c6: ConstraintResult
    meta: dict[str, Any]

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def m(self) -> int:
        return self.X.shape[1]

    @property
    def energy_ratio(self) -> float:
        return float(self.mode_energy.max() / self.mode_energy.min())

    def local_snr_db(self, sigma: float) -> NDArray:
        """逐模态局部 SNR（dB），口径与聚合 SNR 相同（F-范数能量）。"""
        noise_e = self.n * self.m * sigma**2
        return 10.0 * np.log10(self.mode_energy**2 / noise_e)


# --------------------------------------------------------------------------- 系统参数
def _mu(cfg: Config) -> NDArray:
    """离散特征值 μ_k = exp(λ_k Δt)，λ_k = -ζ_k ω_k + j ω_k √(1-ζ_k²)。"""
    out = []
    for mode in cfg.oscillator.modes:
        w = 2.0 * math.pi * mode.f
        lam = -mode.zeta * w + 1j * w * math.sqrt(1.0 - mode.zeta**2)
        out.append(np.exp(lam * cfg.oscillator.dt))
    return np.asarray(out)


def _true_freqs(mu: NDArray, dt: float) -> NDArray:
    """由 μ 反解真值阻尼频率（Hz）。

    注意：真值是阻尼频率 f√(1-ζ²)，**不是**名义固有频率。
    二者相对差约 ζ²/2（当前参数下 8.0e-6），已大于 SNR=20 dB 档的实测偏差量级。
    """
    return np.sort(np.abs(np.angle(mu)) / (2.0 * math.pi * dt))


# --------------------------------------------------------------------------- 实值通道
def _real_blocks(mu: NDArray) -> list[NDArray]:
    blocks = []
    for m_ in mu:
        blocks.append(np.array([[m_.real, m_.imag], [-m_.imag, m_.real]], dtype=float))
    return blocks


def _krylov(block: NDArray, z0: NDArray, steps: int) -> NDArray:
    """[z0, B z0, ..., B^steps z0]，形状 (2, steps+1)。

    闭式：B = rho * R(phi)，故 B^j = rho^j R(j*phi)，可对 j 向量化。
    """
    rho = math.sqrt(block[0, 0] ** 2 + block[1, 0] ** 2)
    phi = math.atan2(block[0, 1], block[0, 0])
    j = np.arange(steps + 1)
    env = rho**j
    c = env * np.cos(phi * j)
    s = env * np.sin(phi * j)
    perp = np.array([z0[1], -z0[0]], dtype=float)
    return np.outer(z0, c) + np.outer(perp, s)


def _rot(theta: float) -> NDArray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, s], [-s, c]], dtype=float)


def build_real(cfg: Config, rng: np.random.Generator, m: int,
               *, amp_ratio: float = 1.0) -> System:
    """实值主通道：4 维状态空间 + n 通道观测，r_true = 4。"""
    oc = cfg.oscillator
    mu = _mu(cfg)
    blocks = _real_blocks(mu)
    n_modes = len(blocks)
    dim = 2 * n_modes

    if oc.n < dim:
        raise SimulatorError(f"n={oc.n} 不足以观测 {dim} 维状态")

    C = rng.standard_normal((oc.n, dim))
    scale = _amp_scale(n_modes, amp_ratio)

    # --- 相位随机化 + C6 平衡（按实现精确配平） ---
    # 注意：2x2 旋转与 B_k 可交换，故 R(theta) 作用在初状态上等价于相位平移。
    # 但可观测能量 E_k = ||C_blk @ kry(blk, z0_blk)|| 依赖 C_blk 与 z0_blk 的耦合，
    # **并非**旋转不变（C_blk 是固定的一般矩阵）。因此必须在每个实现内、
    # 按实际使用的 z0 方向重新标定缩放，才能保证平衡是精确的。
    z0 = np.empty(dim)
    for k, blk in enumerate(blocks):
        th = rng.uniform(0.0, 2.0 * math.pi)
        u = _rot(th) @ np.array([1.0, 0.0])
        g = np.linalg.norm(C[:, 2 * k : 2 * k + 2] @ _krylov(blk, u, m))
        if g <= 0.0 or not np.isfinite(g):
            raise SimulatorError(f"模态 {k} 不可观测（能量 {g}）")
        z0[2 * k : 2 * k + 2] = scale[k] * u / g

    # --- 轨迹（闭式向量化） ---
    Zs = np.empty((dim, m + 1))
    for k, blk in enumerate(blocks):
        Zs[2 * k : 2 * k + 2, :] = _krylov(
            blk, z0[2 * k : 2 * k + 2], m
        )
    X = C @ Zs[:, :m]
    Y = C @ Zs[:, 1 : m + 1]

    # --- 实测逐模态可观测能量 ---
    energy = np.array(
        [np.linalg.norm(C[:, 2 * k : 2 * k + 2] @ Zs[2 * k : 2 * k + 2, :]) for k in range(n_modes)]
    )
    c6 = check_c6(float(energy.max() / energy.min()), target=amp_ratio,
                  tol=oc.c_balance_tol)
    if not c6.passed:
        raise SimulatorError(f"C6 硬约束失败：可观测能量比 {c6.value} 超出 {c6.detail}")
    if c6.warning:
        LOG.warning("C6 进入警戒带：能量比 %s（%s）", c6.value, c6.detail)

    f_true = _true_freqs(mu, oc.dt)
    return System(
        channel="real",
        X=X,
        Y=Y,
        f_true=f_true,
        mu_true=mu[np.argsort(np.abs(np.angle(mu)))],
        rank_true=dim,
        mode_energy=energy,
        c6=c6,
        meta={
            "construction": "real_lti_state_space",
            "n_modes": n_modes,
            "dim": dim,
            "energy_ratio": float(energy.max() / energy.min()),
            "amp_ratio_target": amp_ratio,
            # 观测矩阵随实现随机抽取。**必须暴露出来**，否则
            # "固定 C、只改频率"这类对照实验无法被验证（见 exp8 与
            # tests/test_crossover_scatter.py::test_same_C_*）。
            "C": C.copy(),
        },
    )


# --------------------------------------------------------------------------- 复值通道
def build_complex(cfg: Config, rng: np.random.Generator, m: int,
                  *, amp_ratio: float = 1.0) -> System:
    """复值对照通道：解析信号，r_true = n_modes。"""
    oc = cfg.oscillator
    mu = _mu(cfg)
    n_modes = mu.size

    V = rng.standard_normal((oc.n, n_modes)) + 1j * rng.standard_normal((oc.n, n_modes))
    vnorm = np.linalg.norm(V, axis=0)

    # C6 平衡：|c_k| 使各模态时间累积能量等于目标
    j = np.arange(m + 1)
    decay = np.sqrt((np.abs(mu)[None, :] ** (2 * j[:, None])).sum(axis=0))
    amp = _amp_scale(n_modes, amp_ratio) / (vnorm * decay)
    phase = rng.uniform(0.0, 2.0 * math.pi, n_modes)
    c = amp * np.exp(1j * phase)

    Zs = (V * c[None, :]) @ (mu[None, :] ** j[:, None]).T
    X, Y = Zs[:, :m], Zs[:, 1 : m + 1]

    energy = np.abs(c) * vnorm * decay
    c6 = check_c6(float(energy.max() / energy.min()), target=amp_ratio,
                  tol=oc.c_balance_tol)
    if not c6.passed:
        raise SimulatorError(f"C6 硬约束失败：可观测能量比 {c6.value}")
    if c6.warning:
        LOG.warning("C6 进入警戒带：能量比 %s", c6.value)

    f_true = _true_freqs(mu, oc.dt)
    return System(
        channel="complex",
        X=X,
        Y=Y,
        f_true=f_true,
        mu_true=mu[np.argsort(np.abs(np.angle(mu)))],
        rank_true=n_modes,
        mode_energy=energy,
        c6=c6,
        meta={
            "construction": "complex_analytic",
            "n_modes": n_modes,
            "dim": 2 * n_modes,
            "energy_ratio": float(energy.max() / energy.min()),
            "amp_ratio_target": amp_ratio,
        },
    )


# --------------------------------------------------------------------------- 时延嵌入通道
def _hankel_rows(u: NDArray, length: int, m: int) -> NDArray:
    """由标量序列 u（长度 m+length）构造 X = [u[j:j+L]]_{j=0..m-1}，(L, m)。"""
    W = np.lib.stride_tricks.sliding_window_view(u, length)   # (m+1, length)
    return np.ascontiguousarray(W[:m].T)


def build_hankel(cfg: Config, rng: np.random.Generator, m: int, *,
                 embed: int, amp_ratio: float = 1.0) -> System:
    """时延嵌入通道：单测点标量序列 + L 维时延嵌入，r_true = n_modes。

    对应真实部署中"单/少测点 + 时延构造虚拟通道"的形态（ADR-013）。
    嵌入维数 L 即该通道的"空间维数 n"。

    X[i, j] = u[i+j],  Y[i, j] = u[i+j+1],  i = 0..L-1,  j = 0..m-1
    无噪时 rank(Z) = n_modes，且 Y = A_true X 精确成立。
    """
    oc = cfg.oscillator
    mu = _mu(cfg)
    n_modes = mu.size
    length = int(embed)
    if length < n_modes:
        raise SimulatorError(f"嵌入维数 L={length} 小于模态数 {n_modes}")

    j = np.arange(m + length)
    basis = np.stack([mu[k] ** j for k in range(n_modes)])      # (n_modes, m+L)
    gnorm = np.array([np.linalg.norm(_hankel_rows(g, length, m)) for g in basis])

    scale = _amp_scale(n_modes, amp_ratio)
    phase = rng.uniform(0.0, 2.0 * math.pi, n_modes)
    c = (scale / gnorm) * np.exp(1j * phase)

    u = (c[:, None] * basis).sum(axis=0)                       # 单测点标量序列
    W = np.lib.stride_tricks.sliding_window_view(u, length)    # (m+1, length)
    X = np.ascontiguousarray(W[:m].T)
    Y = np.ascontiguousarray(W[1 : m + 1].T)

    energy = np.abs(c) * gnorm
    c6 = check_c6(float(energy.max() / energy.min()), target=amp_ratio,
                  tol=oc.c_balance_tol)
    if not c6.passed:
        raise SimulatorError(f"C6 硬约束失败：可观测能量比 {c6.value}")
    if c6.warning:
        LOG.warning("C6 进入警戒带：能量比 %s", c6.value)

    f_true = _true_freqs(mu, oc.dt)
    return System(
        channel="hankel",
        X=X,
        Y=Y,
        f_true=f_true,
        mu_true=mu[np.argsort(np.abs(np.angle(mu)))],
        rank_true=n_modes,
        mode_energy=energy,
        c6=c6,
        meta={
            "construction": "hankel_delay_embedding",
            "n_modes": n_modes,
            "embed": length,
            "energy_ratio": float(energy.max() / energy.min()),
            "amp_ratio_target": amp_ratio,
        },
    )


def build(cfg: Config, channel: str, rng: np.random.Generator, m: int, *,
          embed: int | None = None, amp_ratio: float = 1.0) -> System:
    if channel == "real":
        return build_real(cfg, rng, m, amp_ratio=amp_ratio)
    if channel == "complex":
        return build_complex(cfg, rng, m, amp_ratio=amp_ratio)
    if channel == "hankel":
        if embed is None:
            raise SimulatorError("channel='hankel' 时必须给出 embed（嵌入维数 L）")
        return build_hankel(cfg, rng, m, embed=embed, amp_ratio=amp_ratio)
    raise SimulatorError(f"未知通道 {channel!r}，可选 {CHANNELS}")


def rank_for(cfg: Config, channel: str) -> int:
    if channel == "real":
        return cfg.oscillator.r_real
    if channel in ("complex", "hankel"):
        return cfg.oscillator.r_complex
    raise SimulatorError(f"未知通道 {channel!r}")
