# 噪声数据下 DMD 与 TDMD 频率估计的偏差-方差评估及 TDMD 强收敛性验证

> 有限样本条件下，标准 DMD 的噪声偏差与 TDMD 去偏收益的定量刻画。

---

## 1. 题目

**噪声数据下 DMD 与 TDMD 频率估计的偏差-方差评估及 TDMD 强收敛性验证。**

## 2. 背景

DMD（动态模式分解，Dynamic Mode Decomposition）是从时间序列数据中提取动态模式的方法，比如从传感器数据里算出振动频率。

标准 DMD 假设一组快照是干净的、另一组含噪声，但实际数据两边都有噪声，所以它算出的频率会有系统性偏差，而且这种偏差**不会因为数据变多就消失**。

TDMD（Total DMD）是标准 DMD 的抗噪改进版，理论上能纠正这种偏差，并且已有强收敛性证明。但理论只说明"数据足够多时管用"，**有限样本下到底好多少、方差代价多大、噪声估计不准时还能不能用，目前缺乏系统验证**。

## 3. 我们想做的

用两个已知频率的模拟振荡系统，加不同强度的高斯噪声，做三组实验：

### 实验一 · 偏差-方差分解

在 SNR = 20、10、5 dB 下，比较标准 DMD 和 TDMD 的偏差、标准差和 MSE，看 TDMD 纠偏的收益和方差代价。

### 实验二 · 样本量扫描

固定 SNR = 10 dB，让快照数从 50 增加到 500，验证标准 DMD 的偏差是否不随样本量下降、TDMD 的偏差是否按理论预测的 O(1/√n) 速率下降。

### 实验三 · 噪声估计敏感性

用偏离真实值的噪声估计运行 TDMD，检验它的鲁棒边界。

## 4. 预期结论

给出"什么噪声和样本条件下，TDMD 值得用、标准 DMD 够用、或者两者都不行"的实用判断。

---

## 5. 立项后核对结论（重要）

对照原始文献（Hemati, Rowley, Deem & Cattafesta, *TCFD* 2017, arXiv:1502.03854）逐条核对后，有三点需要修订记录，**其中第 3 点直接影响实验三的可行性**：

| # | 立项表述 | 核对结果 | 处理 |
|---|---|---|---|
| 1 | "TDMD 是标准 DMD 的抗噪改进版，理论上能纠正这种偏差，已有强收敛性证明" | 成立。TDMD 的偏差来源被定位为 DMD **第一阶段子空间投影的非对称处理**（只修正 `Y`、把 `X` 当精确值）；去偏依据是误差变量模型下 TLS 的强收敛性（TLS 解收敛到真解，LS 解不收敛）。 | 直接采用 |
| 2 | "标准 DMD 的偏差不随样本量下降" | 成立。原文明确：*DMD possesses bias in an expected value sense, so while such methods will reduce the variance, they will not remove the bias error.* | 直接采用 |
| 3 | "用**偏离真实值的噪声估计**运行 TDMD" | **不成立。原文版 TDMD 不含任何噪声水平输入参数**，其"noise-aware"是相对 TLS 框架而言，并非需要外部给定 σ。原文将加权/正则化 TLS 明确列为 *outside the scope of this study... the focus of ongoing work*。 | **实验三需重新定义**，见 `docs/experiment-protocol.md` §4 |
| 4 | 实值振荡信号的建模方式 | 立项未涉及，但**直接叠加式 `Σ v_k·cos(ω_d t+φ_k)·e^{-ζ_kω_k t}` 是错误的**：该形式使 `Y = A·X` 不精确成立，任何秩的 DMD 都无法恢复真值（实测无噪频率误差达 7.3，完全失败）。 | 实值通道改用 **4 维实 LTI 状态空间实现**（`Y = A·X` 精确成立，实测误差降至 8.1e-16）。见 `docs/algorithm-spec.md` §2.1.3 |
| 5 | 立项写"TDMD 的偏差是否按 O(1/√n) 下降" | 术语笔误。`n` 在 DMD 文献中固定指**状态维数**，样本量应为**快照数 `m`**。 | 全文统一为 `O(1/√m)`（标准差/RMSE）与 `O(1/m)`（MSE） |
| 6 | 隐含假设：无噪退化判据可验证 TDMD 实现正确 | **不成立**。实测误用 `Z` 的**左**奇异向量时，无噪条件下频率误差仍为 `8.9e-16`，判据全部通过。错误只在含噪时显现。 | 验收门 G3 增补**含噪去偏有效性判据**（T11），并以独立参考实现标定阈值。见 `docs/implementation-plan.md` §6 与 `docs/algorithm-spec.md` §8.2 |

### 5.1 已锁定的关键技术决策

以下决策已生效，完整依据与可推翻条件见 [`docs/implementation-plan.md`](docs/implementation-plan.md) §9 决策记录（ADR-001 ~ ADR-009）：

| 项 | 决策 |
|---|---|
| TDMD 定义 | Total DMD（Hemati et al. 2017） |
| 实验三方案 | **3A 秩估计对噪声水平的敏感性**（3B/3C 为可选扩展） |
| 实值通道建模 | 4 维实 LTI 状态空间 + 观测矩阵（**非**直接叠加式） |
| 仿真参数 | `f_s=200Hz`，`(f_1,f_2)=(12.0,17.5)Hz`，`(ζ_1,ζ_2)=(0.005,0.004)`，`n=64`，`r=4`/`2` |
| 噪声模型 | 单条噪声轨迹（`X`/`Y` 噪声列方向重叠） |
| SNR 口径 | 聚合 F-范数：`10log₁₀(‖X̄‖_F²/(n·m·σ²))` |
| 真值频率基准 | 阻尼频率 `f_i√(1-ζ_i²)`（**非**名义频率） |
| 实现次数 | `J=5000`（主）／`20000`（实验二关键点） |
| 双通道判据 | X1–X3（降偏方向一致、方差代价一致、分歧须解释） |
| 项目 venv | `.venv/`，Python 3.12.10 |

### 5.2 校验阶段已实测确认的事实

| 事实 | 数值 |
|---|---|
| 无噪下 DMD / TDMD 恢复真频率 | 误差 8.1e-16 / 4.4e-16 |
| 无噪下 TDMD 退化为 DMD（判据 D1） | `‖Ã_TDMD-Ã_DMD‖_F/‖Ã_DMD‖_F = 1.30e-15` ✅ |
| 直接叠加式建模的失败程度 | 无噪 DMD 频率误差 **7.3**（完全失败） |
| 无噪判据对投影错误的判别力 | **零**——误用左奇异向量时误差仍 8.9e-16 |
| 复现原文 Figure 1 算例 | TDMD 降偏 45%（`m=100`）到 55–77%（`m=500`）✅ |
| 负对照 `U1*U2` 在原文算例上 | 偏差 0.912 vs 正确版 0.073（**12 倍**） |
| `U1⁺U2` 是否等价于正确实现 | 在复值 `r=2` 的 10 个样本上**特征值完全相同**（差 ~1e-15）→ 非有效反例 |
| T11 判别性 | ✅ 通过；负对照（遗漏增广）被正确拒绝 |
| TDMD 的方差代价 | **未观察到**——配对 bootstrap 给出 std 比 ≈ 0.91，CI 上界远低于 1 |
| BLAS 线程对速度的影响 | 128×200 复数 SVD：单线程 5.27 ms，默认 70.5 ms（**13 倍**） |
| 单次完整实现（含数据生成） | 复值 16.8 ms；实值 7.8 ms |
| 全部实验推算总工时 | ≈ 1.2 小时（单核单线程） |

> 检索中也存在另一个同名缩写 TDMD = *Two-stage Difference Mode Decomposition*（Guo et al., *Measurement* 2024，面向轴承/齿轮故障的噪声频带剔除）。该工作与偏差-方差理论无关，**不是本项目所指对象**，特此排除以免混引。

## 6. 文档索引

| 文档 | 内容 |
|---|---|
| [`docs/implementation-plan.md`](docs/implementation-plan.md) | 项目实现规划：可检验假设、系统架构、目录结构、模块划分、里程碑与验收门、风险登记册、**决策记录（ADR）、工作量估算** |
| [`docs/algorithm-spec.md`](docs/algorithm-spec.md) | 算法与统计规格：符号约定、仿真器、SNR 定义、DMD/TDMD 精确算法、指标与置信区间定义、蒙特卡洛协议、秩估计器规格、单元测试判据、基线预验证 |
| [`docs/experiment-protocol.md`](docs/experiment-protocol.md) | 三组实验的参数网格、输出物、判定准则、图表清单与运行矩阵 |
| [`docs/exp1-results.md`](docs/exp1-results.md) | **实验一结果**：通道数/SNR/幅度比/时延嵌入四轴扫描，四类区域决策表 |
| [`docs/exp2-exp3-results.md`](docs/exp2-exp3-results.md) | **实验二、三结果**：样本量扫描的斜率与假设判定；秩判据对噪声估计的敏感性 |
| [`docs/final-report.md`](docs/final-report.md) | **最终报告**：11 条主要结论、假设检验汇总、适用边界决策表、实践决策流程、认知边界与被推翻的中间结论 |

### 6.1 工作量速览

| 项 | 量 |
|---|---|
| 基准总量 | **≈ 20 人日**（含 30% 缓冲） |
| **已完成** | ≈ **12 人日**（文档与预验证 3 + M0–M4 实现 6 + 可行性标定与门验证 3） |
| **剩余** | ≈ **8 人日**（M5–M8：三组实验、图表、决策表、技术报告） |
| 代码规模 | ≈ 3 600 行（`src/` + `tests/`），实测 |
| 算力 | 全部实验 ≈ 1.2 小时单核（实测），**不构成约束** |
| 可选扩展 | 方案 3B +4 人日；方案 3C +6 人日 |

风险分布已明显收窄：原"两个高不确定项"（M3 分歧调查、M4 机制解释）均已闭环，
剩余为常规执行工作。详见 [`docs/implementation-plan.md`](docs/implementation-plan.md) §6、§7、§10。

## 7. 目录结构（规划）

```
dmd_tdmd_noise_eval/
├── README.md
├── .venv/                      # 项目虚拟环境（Python 3.12.10，不入版本库）
├── docs/                       # 设计与协议文档
├── configs/                    # 实验配置（YAML）
├── src/dmdnoise/
│   ├── sim/                    # 仿真器：双模态 LTI 系统 + 噪声注入
│   ├── estimators/             # dmd.py / tdmd.py / rank.py
│   ├── metrics.py              # 偏差 / 标准差 / MSE / RMSE
│   ├── experiments/            # exp1 / exp2 / exp3 驱动脚本
│   └── report/                 # 表格与图表生成
├── scripts/                    # 命令行入口
├── results/                    # raw / tables / figures（不入版本库）
└── tests/                      # 正确性判据单元测试
```

### 7.1 环境

```bash
# 已就绪，无需重建
.venv/Scripts/python.exe -c "import numpy; print(numpy.__version__)"

# 依赖：numpy / scipy / matplotlib / pandas / pyyaml / pytest
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

## 8. 当前状态

- [x] 立项描述归档
- [x] 文献核对与关键澄清（6 项修订）
- [x] 实现规划、算法规格、实验协议
- [x] 决策记录（ADR-001 ~ **ADR-018**）
- [x] 仿真参数设计约束数值校验（C3/C4/C6 处于警戒带）
- [x] 项目虚拟环境（Python 3.12.10）
- [x] **M0 配置层** — schema / 指纹 / C1–C6 校验
- [x] **M1 仿真层** — 双通道系统 + 单轨迹噪声 + SNR 口径
- [x] **M2/M3 估计层** — DMD / TDMD / 配对 / 负对照变体
- [x] **M4 统计层** — 指标、置信区间、配对 bootstrap、蒙特卡洛驱动
- [x] **测试** — 210 项全部通过（含 3 项 T11 判别性测试）
- [x] **G3 验收门闭合** — 投影子空间判别问题已解决，T11 已标定
- [x] **可行性标定** — 已实测确定各 SNR 档位的分辨所需 `J`，并据此修订实验设计
- [x] **M5 实验一执行完成** — 29 个切片 / 116 行，见 [`docs/exp1-results.md`](docs/exp1-results.md)
- [x] **M6 实验二执行完成** — `m ∈ {50…500}` 斜率与 H5/H6 判定
- [x] **M7 实验三执行完成** — 三个秩判据对 σ̂ 的敏感性，鲁棒区间
- [x] **实验二方法学修正** — 加入逐 m 归一化 SNR 对照，推翻"偏差随 m 上升"的假象
- [x] **图表生成** — fig1–fig16（`report/figures.py`，只读 CSV）
- [x] **M8 汇总与最终报告** — [`docs/final-report.md`](docs/final-report.md)
- [x] **实验四 · 参数稳健性** — 模态间隔是强杠杆（降 3.7 倍），窗内衰减几乎无影响；发现交叉点
- [x] **实验五 · 交叉点定位** — 交叉点**不是普遍现象**：复值通道 `f2/f1 ≤ 2.0` 内不交叉；`n` 的影响在两个模态上方向相反
- [x] **实验六 · 端到端评估** — 补上 oracle 秩与判据比较之间的断层；查明"虚假不动点"机制
- [x] **σ̂ 解析偏差校正** — `√(M/(M−r))`，把 `σ̂/σ` 由 0.931 恢复到 0.9955；改写了秩判据的可用性结论
- [x] **实验七 · 秩估计方案** — `single` 有效（断开反馈回路）、`damped` 无效、`multi` 与 `iter` 等价
- [x] **实验八 · 交叉点的实现间散布** — 推翻"随 `n` 非单调"：那是观测矩阵 `C` 的实现噪声（10–300 倍）
- [x] **实验九 · 有色噪声** — **秩判据全线失效**；TDMD 收益只对通道相关稳健
- [x] **实验十 · 预白化** — 盲估计预白化在 10 个配置中的 7 个上完全达到 oracle
- [x] **谱标定秩判据**（失败记录，未编号）— 两条死路已记录，开放问题如实标注
- [ ] 建议的后续工作（见最终报告 §7/§8.5）

### 8.1 本轮（初步验证）的关键修订

| # | 修订 | 依据 |
|---|---|---|
| 1 | **BLAS 必须锁单线程** | 小矩阵 SVD 单线程比默认快 **13 倍**；曾据此误判项目需数十小时 |
| 2 | **偏差可分辨组下移到 SNR 0/−5/−10 dB** | 实测所需 `J` 约按 `10^(SNR/10)` 变化；10 dB 处需 `J≈1.5e5` |
| 3 | **高 SNR 档保留但偏差只报上界** | 偏差在检测下限之下，这是测试床的固有性质而非实现缺陷 |
| 4 | **实验二 SNR 由 10 dB 改为 −5 dB** | 10 dB 处斜率拟合会退化为对噪声拟合 |
| 5 | **C6 构造目标放宽，幅度比列为实验因子** | 非平衡幅度使所需 `J` 降低 3.6–8 倍 |
| 6 | **自造"错误变体"不得直接充当反例** | `U1⁺U2` 实测与正确实现特征值等价，曾导致一次错误结论 |

## 9. 主要参考文献

1. M. S. Hemati, C. W. Rowley, E. A. Deem, L. N. Cattafesta. *De-biasing the dynamic mode decomposition for applied Koopman spectral analysis of noisy datasets.* Theoretical and Computational Fluid Dynamics, 31(4): 349–368, 2017. doi:10.1007/s00162-017-0432-2（预印本 arXiv:1502.03854）
2. P. J. Schmid. *Dynamic mode decomposition of numerical and experimental data.* Journal of Fluid Mechanics, 656: 5–28, 2010.
3. J. N. Tu et al. *On dynamic mode decomposition: Theory and applications.* Journal of Computational Dynamics, 1(2): 391–421, 2014.（含噪声对 DMD 影响的讨论）
4. I. Markovsky, S. Van Huffel. *Overview of total least-squares methods.* Signal Processing, 87(10): 2283–2302, 2007.（TLS 的收敛性与统计性质）
