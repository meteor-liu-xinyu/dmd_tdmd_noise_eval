"""实验层：三组实验驱动脚本。"""

from dmdnoise.experiments.exp1_bias_variance import (
    PAIR_FAIL_GATE,
    Exp1Result,
    Slice,
    annotate_reportable,
    default_slices,
    headline,
    run as run_exp1,
)
from dmdnoise.experiments.exp2_sample_scan import (
    M_VALUES,
    SNR_DB as EXP2_SNR_DB,
    Exp2Result,
    judge as judge_exp2,
    run as run_exp2,
)
from dmdnoise.experiments.exp3_rank_sensitivity import (
    K_GRID,
    Exp3Result,
    run as run_exp3,
)

M_DEFAULT = 200
METHODS = ("dmd", "tdmd")

__all__ = [
    "EXP2_SNR_DB",
    "K_GRID",
    "M_DEFAULT",
    "METHODS",
    "M_VALUES",
    "PAIR_FAIL_GATE",
    "Exp1Result",
    "Exp2Result",
    "Exp3Result",
    "Slice",
    "annotate_reportable",
    "default_slices",
    "headline",
    "judge_exp2",
    "run_exp1",
    "run_exp2",
    "run_exp3",
]
