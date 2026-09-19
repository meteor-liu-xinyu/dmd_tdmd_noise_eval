"""dmdnoise —— 噪声数据下 DMD / TDMD 频率估计的偏差-方差评估。

分层依赖约束（详见 docs/implementation-plan.md §3）：
    sim / estimators -> metrics -> experiments -> report
L(n) 只能依赖 L(≤n)。
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
