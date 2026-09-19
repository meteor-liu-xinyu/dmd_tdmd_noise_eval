"""估计器统一接口与结果数据结构。

规格依据：docs/algorithm-spec.md §4、§5、§8.3。
所有估计器实现同一接口，使实验层对方法无感知。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

#: 奇异值置零的相对阈值：s_i / s_1 < SV_TOL 视为数值零
SV_TOL = 1e-12

#: 判定"接近 Nyquist"的频率比例阈值（相对 f_s/2）
NYQUIST_FRAC = 0.9


class EstimatorError(ValueError):
    """估计器输入非法或数值失败。"""


@dataclass
class Diagnostic:
    """数值诊断。低 SNR、大 m 下的病态风险必须可事后审计。"""

    cond: float = math.nan                 # 截断矩阵条件数 sigma_1 / sigma_r
    cond_full: float = math.nan            # 全谱条件数 sigma_1 / sigma_last
    sv_zeroed: int = 0                     # 被置零的奇异值计数
    sv_min_ratio: float = math.nan         # sigma_r / sigma_1
    rank_used: int = 0
    augmented: bool = False                # 是否使用增广矩阵（TDMD 断言）
    z_shape: tuple[int, int] | None = None
    near_nyquist: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "cond": self.cond,
            "cond_full": self.cond_full,
            "sv_zeroed": self.sv_zeroed,
            "sv_min_ratio": self.sv_min_ratio,
            "rank_used": self.rank_used,
            "augmented": self.augmented,
            "z_shape": list(self.z_shape) if self.z_shape else None,
            "near_nyquist": self.near_nyquist,
            **self.extra,
        }


@dataclass
class EstimateResult:
    """单个实现、单一方法的一次估计结果。"""

    freqs: NDArray              # (r,) 升序，Hz
    eigenvalues: NDArray        # (r,) 离散特征值 mu，与 freqs 同序
    modes: NDArray              # (n, r)
    diagnostics: Diagnostic

    def __post_init__(self) -> None:
        if self.freqs.shape != self.eigenvalues.shape:
            raise EstimatorError("freqs 与 eigenvalues 形状必须一致")

    @property
    def n_modes(self) -> int:
        return int(self.freqs.size)


@runtime_checkable
class Estimator(Protocol):
    name: str

    def fit(self, X: NDArray, Y: NDArray, r: int, dt: float) -> EstimateResult: ...


# --------------------------------------------------------------------------- 共用工具
def safe_pinv_diag(sv: NDArray, tol: float = SV_TOL) -> tuple[NDArray, int]:
    """对角奇异值求逆，小奇异值置零。返回 (逆对角, 置零计数)。

    规格 §9.2：不得对零奇异值直接求逆。
    """
    if sv.size == 0:
        raise EstimatorError("奇异值数组为空")
    inv = np.zeros_like(sv, dtype=float)
    big = sv > tol * sv[0]
    inv[big] = 1.0 / sv[big]
    return inv, int((~big).sum())


def order_by_frequency(evals: NDArray, dt: float) -> NDArray:
    """按 |arg(mu)|/(2*pi*dt) 升序返回重排索引。"""
    return np.argsort(np.abs(np.angle(evals)) / (2.0 * math.pi * dt))


def freqs_from_eigenvalues(evals: NDArray, dt: float) -> NDArray:
    """由离散特征值取频率（Hz）。使用 angle 主值，范围 (-fs/2, fs/2]。"""
    return np.abs(np.angle(evals)) / (2.0 * math.pi * dt)


def check_nyquist(freqs: NDArray, dt: float) -> bool:
    fs = 1.0 / dt
    return bool(np.any(freqs > NYQUIST_FRAC * fs / 2.0))
