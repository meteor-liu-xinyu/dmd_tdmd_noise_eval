"""估计层：DMD / TDMD 及测试用负对照变体。"""

from dmdnoise.estimators.base import (
    NYQUIST_FRAC,
    SV_TOL,
    Diagnostic,
    EstimateResult,
    Estimator,
    EstimatorError,
    check_nyquist,
    freqs_from_eigenvalues,
    order_by_frequency,
    safe_pinv_diag,
)
from dmdnoise.estimators.dmd import DMD
from dmdnoise.estimators.pairing import (
    MISPAIR_FRAC,
    PairResult,
    pair_to_truth,
    relative_error,
    truth_separation,
    unwrap_flags,
)
from dmdnoise.estimators.tdmd import TDMD

#: 正式实验允许使用的估计器（负对照变体不在其中）
REGISTRY = {"dmd": DMD, "tdmd": TDMD}

__all__ = [
    "MISPAIR_FRAC",
    "NYQUIST_FRAC",
    "REGISTRY",
    "SV_TOL",
    "DMD",
    "TDMD",
    "Diagnostic",
    "EstimateResult",
    "Estimator",
    "EstimatorError",
    "PairResult",
    "check_nyquist",
    "freqs_from_eigenvalues",
    "order_by_frequency",
    "pair_to_truth",
    "relative_error",
    "safe_pinv_diag",
    "truth_separation",
    "unwrap_flags",
]
