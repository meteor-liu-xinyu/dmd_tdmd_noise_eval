"""实验层：三组实验驱动脚本。"""

from dmdnoise.experiments.exp1_bias_variance import (
    M_DEFAULT,
    METHODS,
    Exp1Result,
    Slice,
    annotate_reportable,
    default_slices,
    headline,
    run as run_exp1,
)

__all__ = [
    "M_DEFAULT",
    "METHODS",
    "Exp1Result",
    "Slice",
    "annotate_reportable",
    "default_slices",
    "headline",
    "run_exp1",
]
