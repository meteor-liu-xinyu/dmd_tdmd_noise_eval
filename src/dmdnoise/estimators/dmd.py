"""标准 DMD 估计器（算法 A1）。

规格依据：docs/algorithm-spec.md §4。

刻意保留"两阶段"写法，使 TDMD 的差异在代码中逐行可见：

    阶段 1 子空间投影      阶段 2 算子识别
    ------------------    -------------------
    P_{X*} = U_r U_r*     A = U_r* Y V_r Sigma_r^{-1}
    Yb = Y P_{X*}

偏差来源即在此：第一阶段只对 Y 做"纠错"，把 X 当作精确值（非对称处理）。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from dmdnoise.estimators.base import (
    EstimateResult,
    Diagnostic,
    EstimatorError,
    check_nyquist,
    freqs_from_eigenvalues,
    order_by_frequency,
    safe_pinv_diag,
)


class DMD:
    """标准 DMD。"""

    name = "dmd"

    def fit(self, X: NDArray, Y: NDArray, r: int, dt: float) -> EstimateResult:
        n, m = X.shape
        if Y.shape != (n, m):
            raise EstimatorError(f"X 与 Y 形状必须一致，收到 {X.shape} 与 {Y.shape}")
        if r < 1:
            raise EstimatorError("截断秩 r 必须 >= 1")
        if r > min(n, m):
            raise EstimatorError(f"r={r} 超过 min(n,m)={min(n, m)}")

        # --- 阶段 1：子空间投影（偏差来源） ---
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        Ur, Vr = U[:, :r], Vt[:r, :].conj().T
        Yb = Y @ Vr @ Vr.conj().T          # 仅 Y 被 "纠错"
        Xb = X @ Vr @ Vr.conj().T          # 由构造自动满足 X P_{X*} = X

        # --- 阶段 2：算子识别 ---
        Ub, Sb, Vtb = np.linalg.svd(Xb, full_matrices=False)
        Ubr, Vbr = Ub[:, :r], Vtb[:r, :].conj().T
        inv_s, zeroed = safe_pinv_diag(Sb[:r])
        A = Ubr.conj().T @ Yb @ Vbr @ np.diag(inv_s)

        evals = np.linalg.eigvals(A)
        idx = order_by_frequency(evals, dt)
        evals = evals[idx]
        freqs = freqs_from_eigenvalues(evals, dt)

        # 模态：phi_i = Ub w_i
        _, W = np.linalg.eig(A)
        modes = Ubr @ W[:, idx]

        diag = Diagnostic(
            cond=float(Sb[0] / Sb[r - 1]),
            cond_full=float(S[0] / S[-1]),
            sv_zeroed=zeroed,
            sv_min_ratio=float(S[r - 1] / S[0]),
            rank_used=r,
            augmented=False,
            z_shape=(n, m),
            near_nyquist=check_nyquist(freqs, dt),
            extra={"method": "dmd"},
        )
        return EstimateResult(freqs=freqs, eigenvalues=evals, modes=modes,
                              diagnostics=diag, operator=A)
