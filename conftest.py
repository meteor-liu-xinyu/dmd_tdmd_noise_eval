"""仓库根 conftest：在任何 numpy 导入之前锁定 BLAS 线程数。

背景（实测）：本项目矩阵规模很小（2n x m <= 128 x 500），OpenBLAS 的多线程调度
开销远大于计算本身。同一台机器上对 128x200 复数矩阵做 SVD：

    OPENBLAS_NUM_THREADS=1   5.27 ms
    OPENBLAS_NUM_THREADS=2  11.14 ms
    OPENBLAS_NUM_THREADS=4  14.79 ms
    OPENBLAS_NUM_THREADS=8  28.06 ms
    默认（未设置）          70.50 ms

即单线程比默认快约 13 倍。OpenBLAS 在库加载时读取该环境变量，
因此必须在 import numpy 之前设置。
"""

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")
