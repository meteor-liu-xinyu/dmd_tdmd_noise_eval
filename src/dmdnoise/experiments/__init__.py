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
from dmdnoise.experiments.exp10_prewhiten import (
    Exp10Result,
    run as run_exp10,
    verdict as verdict_exp10,
)
from dmdnoise.experiments.exp9_colored_noise import (
    NOISE_GRID,
    Exp9Result,
    NoiseSpec,
    run as run_exp9,
    verdict as verdict_exp9,
)
from dmdnoise.experiments.exp8_crossover_scatter import (
    Exp8Result,
    run as run_exp8,
    verdict as verdict_exp8,
)
from dmdnoise.experiments.exp7_rank_schemes import (
    CRITERIA as RANK_CRITERIA,
    Exp7Result,
    run as run_exp7,
    verdict as verdict_exp7,
)
from dmdnoise.experiments.exp6_end_to_end import (
    Exp6Result,
    STRATEGIES as RANK_STRATEGIES,
    run as run_exp6,
    verdict as verdict_exp6,
)
from dmdnoise.experiments.exp5_crossover import (
    COMBOS,
    Exp5Result,
    run as run_exp5,
    summary as summary_exp5,
)
from dmdnoise.experiments.exp4_robustness import (
    SEPARATIONS,
    WINDOW_DECAYS,
    Exp4Result,
    RobustPoint,
    default_points as robustness_points,
    run as run_exp4,
    verdict as verdict_exp4,
    zeta_from_decay,
)

M_DEFAULT = 200
METHODS = ("dmd", "tdmd")

__all__ = [
    "COMBOS",
    "Exp10Result",
    "run_exp10",
    "verdict_exp10",
    "NOISE_GRID",
    "NoiseSpec",
    "Exp9Result",
    "run_exp9",
    "verdict_exp9",
    "Exp8Result",
    "run_exp8",
    "verdict_exp8",
    "RANK_CRITERIA",
    "RANK_STRATEGIES",
    "Exp7Result",
    "run_exp7",
    "verdict_exp7",
    "Exp6Result",
    "run_exp6",
    "verdict_exp6",
    "EXP2_SNR_DB",
    "Exp5Result",
    "run_exp5",
    "summary_exp5",
    "SEPARATIONS",
    "WINDOW_DECAYS",
    "Exp4Result",
    "RobustPoint",
    "robustness_points",
    "run_exp4",
    "verdict_exp4",
    "zeta_from_decay",
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
