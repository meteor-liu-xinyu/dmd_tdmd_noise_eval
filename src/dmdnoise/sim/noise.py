"""噪声模型与 SNR 口径。

规格依据：docs/algorithm-spec.md §3。

  - 聚合 SNR：SNR_dB = 10 log10( ||X_true||_F^2 / (n*m*sigma^2) )
    => sigma = ||X_true||_F / sqrt(n*m*10^(SNR/10))，与 m 无关
  - 单条噪声轨迹（trajectory）：X 与 Y 的噪声在列方向重叠，物理上对应
    一条连续含噪轨迹的相邻窗口。这是 DMD 偏差的真实机制。
  - 独立加噪（independent）：仅作对照，用于说明模型选择的影响。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

VALID_MODES = ("trajectory", "independent")


class NoiseError(ValueError):
    """噪声模型输入非法。"""


def _noise_like(shape: tuple[int, ...], sigma: float, rng: np.random.Generator,
                complex_: bool) -> NDArray:
    """逐元素标准差为 sigma 的零均值高斯噪声。

    实值：N(0, sigma^2)
    复值：CN(0, sigma^2)，即实虚部各 N(0, sigma^2/2)，使 E|z|^2 = sigma^2
    """
    if complex_:
        z = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / math.sqrt(2.0)
    else:
        z = rng.standard_normal(shape)
    return sigma * z


def sigma_from_snr(X_true: NDArray, snr_db: float, *, convention: str = "aggregate",
                   fixed_sigma: float | None = None) -> float:
    """由目标 SNR 反解逐元素噪声标准差。

    convention="aggregate"   —— 使用聚合 F-范数口径（主实验）
    convention="fixed_sigma" —— 直接使用给定的绝对 sigma（对照口径）
    """
    if convention == "fixed_sigma":
        if fixed_sigma is None:
            raise NoiseError("convention='fixed_sigma' 时必须给出 fixed_sigma")
        return float(fixed_sigma)
    if convention != "aggregate":
        raise NoiseError(f"未知 SNR 口径 {convention!r}")

    n, m = X_true.shape
    energy = float(np.linalg.norm(X_true) ** 2)
    denom = n * m * 10.0 ** (snr_db / 10.0)
    if denom <= 0.0:
        raise NoiseError("SNR 口径计算出非正噪声能量")
    return math.sqrt(energy / denom)


def eps_ratio(X_true: NDArray, sigma: float) -> float:
    """逐元素噪声比 eps = sigma / rms(X_true)。满足 SNR_dB = -20 log10(eps)。"""
    rms = float(np.linalg.norm(X_true) / math.sqrt(X_true.size))
    if rms <= 0.0:
        raise NoiseError("真值快照能量为零")
    return sigma / rms


def snr_from_eps(eps: float) -> float:
    return -20.0 * math.log10(eps)


def inject(X_true: NDArray, Y_true: NDArray, sigma: float, rng: np.random.Generator,
           *, mode: str = "trajectory") -> tuple[NDArray, NDArray, dict[str, Any]]:
    """注入噪声，返回 (X, Y, meta)。

    trajectory   —— 单条噪声轨迹：以完整序列 (m+1 列) 加噪后切片，
                    使 dX[:, 1:] 与 dY[:, :-1] 逐元素相等
    independent  —— X 与 Y 各自独立加噪（破坏两侧误差的相关结构，仅作对照）
    """
    if mode not in VALID_MODES:
        raise NoiseError(f"mode 必须为 {VALID_MODES} 之一，收到 {mode!r}")
    if sigma < 0.0:
        raise NoiseError("sigma 必须非负")

    complex_ = np.iscomplexobj(X_true) or np.iscomplexobj(Y_true)
    n, m = X_true.shape
    if Y_true.shape != (n, m):
        raise NoiseError(f"X 与 Y 形状必须一致，收到 {X_true.shape} 与 {Y_true.shape}")

    if mode == "trajectory":
        full_true = np.concatenate([X_true, Y_true[:, -1:]], axis=1)
        noisy = full_true + _noise_like(full_true.shape, sigma, rng, complex_)
        X, Y = noisy[:, :m], noisy[:, 1 : m + 1]
        overlap_ok = bool(np.array_equal(X[:, 1:], Y[:, :-1]))
        noise_energy = float(np.linalg.norm(noisy[:, :m] - X_true) ** 2)
    else:
        X = X_true + _noise_like(X_true.shape, sigma, rng, complex_)
        Y = Y_true + _noise_like(Y_true.shape, sigma, rng, complex_)
        overlap_ok = False
        noise_energy = (
            float(np.linalg.norm(X - X_true) ** 2) + float(np.linalg.norm(Y - Y_true) ** 2)
        )

    meta = {
        "noise_mode": mode,
        "sigma": sigma,
        "complex": bool(complex_),
        "n": n,
        "m": m,
        "overlap_ok": overlap_ok,
        "noise_energy_realized": noise_energy,
    }
    return X, Y, meta


def realized_snr_db(X_true: NDArray, X_noisy: NDArray) -> float:
    """实测聚合 SNR（dB），用于校验反解一致性（判据 T5）。"""
    num = float(np.linalg.norm(X_true) ** 2)
    den = float(np.linalg.norm(X_noisy - X_true) ** 2)
    if den <= 0.0:
        return math.inf
    return 10.0 * math.log10(num / den)
