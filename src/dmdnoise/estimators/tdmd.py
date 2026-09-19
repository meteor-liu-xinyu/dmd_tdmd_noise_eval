"""TDMD（Total DMD）估计器（算法 A2）。

规格依据：docs/algorithm-spec.md §5。

对照 Hemati, Rowley, Deem & Cattafesta, *TCFD* 31(4):349-368, 2017（arXiv:1502.03854）
§2 的去偏算法，逐步编号如下 —— **实现与步骤一一对应，禁止"简化"**：

    Step 1  Z = [X ; Y]                        （增广快照矩阵，原文 Eq.9）
    Step 2  SVD(Z) -> 取 V 的前 r 列, 记 V_r
            投影算子 P_{Z_r*} = V_r V_r*
    Step 3  Xb = X V_r V_r* ,  Yb = Y V_r V_r*  （X 与 Y 被【对称】处理）
    Step 4  SVD(Xb) -> Ub, Sb, Vb*
    Step 5  A = Ub* Yb Vb Sb^{-1}

与标准 DMD 的唯一差别在 Step 2/Step 3：
  - DMD  用 **X 的左**奇异向量 U_r 构造投影 P_{X*} = U_r U_r*（只纠 Y）
  - TDMD 用 **Z 的右**奇异向量 V_r 构造投影 P_{Z_r*} = V_r V_r*（X、Y 同等对待）

!! 关键易错点（风险 R1）：必须是 **右**奇异向量 V_r，不是左奇异向量 U_r。
   标为 `augmented` 的诊断字段与 Z 形状断言用于拦截"遗漏增广"这一最常见错误形态。
   注意：无噪退化判据无法拦截该错误（详见 docs/algorithm-spec.md §8.2），
   必须依赖含噪去偏有效性判据 T11。
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


class TDMD:
    """Total DMD（去偏）。"""

    name = "tdmd"

    def fit(self, X: NDArray, Y: NDArray, r: int, dt: float) -> EstimateResult:
        n, m = X.shape
        if Y.shape != (n, m):
            raise EstimatorError(f"X 与 Y 形状必须一致，收到 {X.shape} 与 {Y.shape}")
        if r < 1:
            raise EstimatorError("截断秩 r 必须 >= 1")
        if r > min(n, m):
            raise EstimatorError(f"r={r} 超过 min(n,m)={min(n, m)}")

        # --- Step 1: 增广快照矩阵 Z = [X ; Y]（行堆叠） ---
        Z = np.vstack([X, Y])
        if Z.shape != (2 * n, m):                      # G3-c：拦截遗漏增广
            raise EstimatorError(f"增广矩阵形状异常：{Z.shape} != {(2 * n, m)}")

        # --- Step 2: SVD(Z)，取【右】奇异向量前 r 列构造投影 ---
        _, Sz, Vtz = np.linalg.svd(Z, full_matrices=False)
        Vr = Vtz[:r, :].conj().T                        # (m, r) —— 右奇异向量
        P_Z = Vr @ Vr.conj().T                          # (m, m) 投影算子

        # --- Step 3: 对称投影，X 与 Y 同等对待 ---
        Xb = X @ P_Z
        Yb = Y @ P_Z

        # --- Step 4: 对投影后的 Xb 做截断 SVD ---
        Ub, Sb, Vtb = np.linalg.svd(Xb, full_matrices=False)
        Ubr, Vbr = Ub[:, :r], Vtb[:r, :].conj().T
        inv_s, zeroed = safe_pinv_diag(Sb[:r])

        # --- Step 5: A~ = Ub* Yb Vb Sigma_b^{-1} ---
        A = Ubr.conj().T @ Yb @ Vbr @ np.diag(inv_s)

        evals = np.linalg.eigvals(A)
        idx = order_by_frequency(evals, dt)
        evals = evals[idx]
        freqs = freqs_from_eigenvalues(evals, dt)

        _, W = np.linalg.eig(A)
        modes = Ubr @ W[:, idx]

        diag = Diagnostic(
            cond=float(Sb[0] / Sb[r - 1]),
            cond_full=float(Sz[0] / Sz[-1]),
            sv_zeroed=zeroed,
            sv_min_ratio=float(Sb[r - 1] / Sb[0]),
            rank_used=r,
            augmented=True,
            z_shape=(2 * n, m),
            near_nyquist=check_nyquist(freqs, dt),
            extra={
                "method": "tdmd",
                "projection": "right_singular_vectors_of_Z",
                "z_sv_min_ratio": float(Sz[r - 1] / Sz[0]),
            },
        )
        return EstimateResult(freqs=freqs, eigenvalues=evals, modes=modes,
                              diagnostics=diag, operator=A)
