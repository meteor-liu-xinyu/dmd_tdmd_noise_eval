"""蒙特卡洛驱动：种子派生、公共随机数配对、嵌套前缀设计。

规格依据：docs/algorithm-spec.md §7。

三条强制要求：
  1. 种子用 SeedSequence.spawn_key 由 (channel, snr, m, j, purpose) 派生，
     **禁止**全局 np.random.seed，也禁止依赖 Python 内建 hash（有随机盐）。
  2. 公共随机数：同一 (channel, snr, m) 单元内 DMD 与 TDMD 必须使用同一条噪声实现。
  3. 样本量扫描用嵌套前缀设计：每次实现只生成一条长序列，各 m 取其前缀；
     斜率必须**逐实现拟合后跨实现统计**，禁止对跨 m 的合并数据直接 OLS。
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from dmdnoise.config import Config, config_fingerprint
from dmdnoise.estimators import REGISTRY, pair_to_truth
from dmdnoise.sim import build, inject, rank_for, sigma_from_snr

LOG = logging.getLogger(__name__)

PURPOSE_PHASE = "phase"
PURPOSE_NOISE = "noise"
PURPOSE_BOOTSTRAP = "bootstrap"


def _stable_key(k: Any) -> int:
    """把任意键转为稳定的 64 位整数（blake2b，跨平台/跨进程可复现）。"""
    payload = repr(k).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def derive_rng(master_seed: int, *keys: Any) -> np.random.Generator:
    """由主种子与键序列派生独立随机流。"""
    ss = np.random.SeedSequence(master_seed, spawn_key=tuple(_stable_key(k) for k in keys))
    return np.random.default_rng(ss)


def derive_seed_sequence(master_seed: int, *keys: Any) -> np.random.SeedSequence:
    return np.random.SeedSequence(master_seed, spawn_key=tuple(_stable_key(k) for k in keys))


@dataclass
class CellResult:
    """单个 (channel, snr, m) 单元的估计结果。"""

    channel: str
    snr_db: float
    m: int
    j_total: int
    f_true: NDArray                       # (n_modes,)
    estimates: dict[str, NDArray]         # method -> (J, n_modes)，配对失败处为 NaN
    pair_ok: dict[str, NDArray]           # method -> (J,) bool
    sigma: float
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_modes(self) -> int:
        return int(self.f_true.size)

    def valid(self, method: str) -> NDArray:
        """返回配对成功的估计，形状 (J_valid, n_modes)。"""
        est = self.estimates[method]
        ok = self.pair_ok[method]
        return est[ok]

    def errors(self, method: str, mode: int = 0, *, relative: bool = True) -> NDArray:
        """误差序列 f_hat - f_true（默认相对误差），仅取**该方法**配对成功样本。"""
        v = self.valid(method)[:, mode]
        d = v - self.f_true[mode]
        return d / self.f_true[mode] if relative else d

    def paired_errors(self, mode: int = 0, *, relative: bool = True
                      ) -> tuple[NDArray, NDArray]:
        """返回 (dmd, tdmd) 在**两法均配对成功**的公共子集上的误差。

        ⚠️ 配对比较必须取公共子集。若两法各自过滤有效样本，失败集合不同，
           配对结构被破坏，`Var(Δ)` 的自检会失真、比值 CI 也不可信。
        """
        ok = self.pair_ok["dmd"] & self.pair_ok["tdmd"]
        if ok.sum() == 0:
            empty = np.empty(0)
            return empty, empty.copy()
        d = self.estimates["dmd"][ok, mode] - self.f_true[mode]
        t = self.estimates["tdmd"][ok, mode] - self.f_true[mode]
        if relative:
            d = d / self.f_true[mode]
            t = t / self.f_true[mode]
        return d, t

    @property
    def n_paired(self) -> int:
        """两法均配对成功的实现数。"""
        return int((self.pair_ok["dmd"] & self.pair_ok["tdmd"]).sum())

    def summary_row(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "snr_db": self.snr_db,
            "m": self.m,
            "j_total": self.j_total,
            "sigma": self.sigma,
            **{f"pair_fail_{k}": int((~v).sum()) for k, v in self.pair_ok.items()},
        }


def _estimate_cell(cfg: Config, channel: str, snr_db: float, m: int, j_total: int,
                   methods: Sequence[str], *, master_seed: int,
                   progress: Callable[[int, int], None] | None = None,
                   embed: int | None = None, amp_ratio: float = 1.0,
                   axis_tag: str = "") -> CellResult:
    """公共随机数配对比较：两法共享同一条噪声实现。"""
    r = rank_for(cfg, channel)
    est: dict[str, list[NDArray]] = {k: [] for k in methods}
    ok: dict[str, list[bool]] = {k: [] for k in methods}
    f_true: NDArray | None = None
    sigma_used = float("nan")
    pair_fail_detail: dict[str, int] = {k: 0 for k in methods}
    key_extra = (axis_tag, embed, amp_ratio)

    for j in range(j_total):
        if progress is not None and j % max(1, j_total // 10) == 0:
            progress(j, j_total)

        rng_phase = derive_rng(master_seed, channel, snr_db, m, PURPOSE_PHASE, *key_extra, j)
        system = build(cfg, channel, rng_phase, m, embed=embed, amp_ratio=amp_ratio)
        if f_true is None:
            f_true = system.f_true
            sigma_used = sigma_from_snr(system.X, snr_db, convention=cfg.noise.convention)

        rng_noise = derive_rng(master_seed, channel, snr_db, m, PURPOSE_NOISE, *key_extra, j)
        Xn, Yn, _ = inject(system.X, system.Y, sigma_used, rng_noise, mode=cfg.noise.mode)

        for name in methods:
            res = REGISTRY[name]().fit(Xn, Yn, r, cfg.oscillator.dt)
            pr = pair_to_truth(res.freqs, system.f_true)
            if pr.ok:
                est[name].append(pr.freqs)
                ok[name].append(True)
            else:
                est[name].append(np.full(system.f_true.size, np.nan))
                ok[name].append(False)
                pair_fail_detail[name] += 1

    assert f_true is not None
    return CellResult(
        channel=channel,
        snr_db=snr_db,
        m=m,
        j_total=j_total,
        f_true=f_true,
        estimates={k: np.asarray(v, dtype=float) for k, v in est.items()},
        pair_ok={k: np.asarray(v, dtype=bool) for k, v in ok.items()},
        sigma=sigma_used,
        meta={
            "fingerprint": config_fingerprint(cfg),
            "rank": r,
            "noise_mode": cfg.noise.mode,
            "snr_convention": cfg.noise.convention,
            "master_seed": master_seed,
            "pair_fail": pair_fail_detail,
            "n": cfg.oscillator.n if channel != "hankel" else embed,
            "embed": embed,
            "amp_ratio": amp_ratio,
            "axis_tag": axis_tag,
        },
    )


def run_cell(cfg: Config, channel: str, snr_db: float, m: int, j_total: int,
             *, methods: Sequence[str] = ("dmd", "tdmd"), master_seed: int | None = None,
             progress: Callable[[int, int], None] | None = None,
             embed: int | None = None, amp_ratio: float = 1.0,
             axis_tag: str = "") -> CellResult:
    return _estimate_cell(cfg, channel, snr_db, m, j_total, methods,
                          master_seed=master_seed if master_seed is not None else cfg.run.seed,
                          progress=progress, embed=embed, amp_ratio=amp_ratio,
                          axis_tag=axis_tag)


def run_m_scan(cfg: Config, channel: str, snr_db: float, m_values: Sequence[int],
               j_total: int, *, methods: Sequence[str] = ("dmd", "tdmd"),
               master_seed: int | None = None,
               snr_normalization: str = "m_max",
               progress: Callable[[int, int], None] | None = None
               ) -> dict[int, CellResult]:
    """嵌套前缀设计：每次实现生成一条长序列（m_max+1 个快照），各 m 取其前缀。

    **SNR 归一化口径**（`snr_normalization`）：

    `"m_max"`（默认）
        噪声标准差由 `m_max` 前缀一次性反解、跨全部 `m` 共用。
        同一实现的各 `m` 面对**同一条噪声序列、同一个 σ**。
        代价：`m` 越大，窗内信号衰减越多而 σ 不变 → **大 `m` 的有效 SNR 更低**。
        因此"偏差随 `m` 变化"中混入了"有效 SNR 变化"的成分。

    `"per_m"`
        每个 `m` 用**该 `m` 的前缀**反解各自的 `σ_m`，使聚合 SNR 在每个 `m` 上都等于目标值。
        噪声序列仍完全共用（先生成单位方差序列，再逐 `m` 乘 `σ_m`），
        故嵌套结构不受破坏、公共随机数依然成立。
        **这是分离"m 效应"与"有效 SNR 效应"的对照口径。**

    两种口径的斜率差异即"有效 SNR 变化"对结论的贡献。
    """
    if snr_normalization not in ("m_max", "per_m"):
        raise ValueError(f"未知 SNR 归一化口径 {snr_normalization!r}")

    ms = sorted(int(v) for v in m_values)
    m_max = ms[-1]
    r = rank_for(cfg, channel)
    seed = master_seed if master_seed is not None else cfg.run.seed

    est: dict[int, dict[str, list[NDArray]]] = {mm: {k: [] for k in methods} for mm in ms}
    ok: dict[int, dict[str, list[bool]]] = {mm: {k: [] for k in methods} for mm in ms}
    f_true: NDArray | None = None
    sigma_used = float("nan")
    pair_fail: dict[int, dict[str, int]] = {mm: {k: 0 for k in methods} for mm in ms}

    for j in range(j_total):
        if progress is not None and j % max(1, j_total // 10) == 0:
            progress(j, j_total)

        rng_phase = derive_rng(seed, channel, snr_db, "scan", PURPOSE_PHASE, j)
        system = build(cfg, channel, rng_phase, m_max)
        if f_true is None:
            f_true = system.f_true
            sigma_used = sigma_from_snr(system.X, snr_db, convention=cfg.noise.convention)

        rng_noise = derive_rng(seed, channel, snr_db, "scan", PURPOSE_NOISE, j)
        # 以 sigma=1 生成单位方差噪声轨迹；逐 m 再缩放，保证嵌套结构
        Xn1, Yn1, _ = inject(system.X, system.Y, 1.0, rng_noise, mode=cfg.noise.mode)

        for mm in ms:
            if snr_normalization == "per_m":
                sig = sigma_from_snr(system.X[:, :mm], snr_db,
                                     convention=cfg.noise.convention)
            else:
                sig = sigma_used
            Xs = system.X[:, :mm] + sig * (Xn1[:, :mm] - system.X[:, :mm])
            Ys = system.Y[:, :mm] + sig * (Yn1[:, :mm] - system.Y[:, :mm])
            for name in methods:
                res = REGISTRY[name]().fit(Xs, Ys, r, cfg.oscillator.dt)
                pr = pair_to_truth(res.freqs, system.f_true)
                if pr.ok:
                    est[mm][name].append(pr.freqs)
                    ok[mm][name].append(True)
                else:
                    est[mm][name].append(np.full(system.f_true.size, np.nan))
                    ok[mm][name].append(False)
                    pair_fail[mm][name] += 1

    assert f_true is not None
    out: dict[int, CellResult] = {}
    for mm in ms:
        out[mm] = CellResult(
            channel=channel,
            snr_db=snr_db,
            m=mm,
            j_total=j_total,
            f_true=f_true,
            estimates={k: np.asarray(v, dtype=float) for k, v in est[mm].items()},
            pair_ok={k: np.asarray(v, dtype=bool) for k, v in ok[mm].items()},
            sigma=sigma_used,
            meta={
                "fingerprint": config_fingerprint(cfg),
                "rank": r,
                "noise_mode": cfg.noise.mode,
                "snr_convention": cfg.noise.convention,
                "snr_normalization": snr_normalization,
                "master_seed": seed,
                "nested": True,
                "m_max": m_max,
                "sigma_this_m": (
                    sigma_from_snr(system.X[:, :mm], snr_db,
                                   convention=cfg.noise.convention)
                    if snr_normalization == "per_m" else sigma_used
                ),
                "pair_fail": pair_fail[mm],
            },
        )
    return out
