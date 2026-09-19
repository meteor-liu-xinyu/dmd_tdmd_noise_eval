"""仿真层：真值系统构造与噪声注入。"""

from dmdnoise.sim.noise import (
    NoiseError,
    eps_ratio,
    inject,
    realized_snr_db,
    sigma_from_snr,
    snr_from_eps,
)
from dmdnoise.sim.oscillator import (
    CHANNELS,
    SimulatorError,
    System,
    build,
    build_complex,
    build_real,
    rank_for,
)

__all__ = [
    "CHANNELS",
    "NoiseError",
    "SimulatorError",
    "System",
    "build",
    "build_complex",
    "build_real",
    "eps_ratio",
    "inject",
    "rank_for",
    "realized_snr_db",
    "sigma_from_snr",
    "snr_from_eps",
]
