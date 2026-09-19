"""pytest 共用夹具。

注意：仓库根 conftest.py 会在本文件之前执行并锁定 BLAS 线程数。
"""

from __future__ import annotations

import os

# 防御性重复设置（若根 conftest 被跳过，例如从其它目录调用 pytest）
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from dmdnoise.config import Config  # noqa: E402


@pytest.fixture(scope="session")
def cfg() -> Config:
    return Config()


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)
