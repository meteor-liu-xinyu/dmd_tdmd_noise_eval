"""实验层：三组实验驱动脚本。"""

from dmdnoise.experiments.exp1_bias_variance import (
    M_DEFAULT,
    METHODS,
    SNR_RESOLVABLE,
    SNR_UPPER_BOUND,
    Exp1Result,
    decision_notes,
    run as run_exp1,
)

__all__ = [
    "M_DEFAULT",
    "METHODS",
    "SNR_RESOLVABLE",
    "SNR_UPPER_BOUND",
    "Exp1Result",
    "decision_notes",
    "run_exp1",
]
