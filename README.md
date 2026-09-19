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
| 无噪下 DMD 恢复真频率 | 误差 8.1e-16 |
| 无噪下 TDMD 退化为 DMD（判据 D1） | `‖Ã_TDMD-Ã_DMD‖_F/‖Ã_DMD‖_F = 1.30e-15` ✅ |
| 直接叠加式建模的失败程度 | 无噪 DMD 频率误差 **7.3**（完全失败） |
| 误用左奇异向量：无噪 | 误差 8.9e-16（**无法区分，判据失效**） |
| 误用左奇异向量：含噪 | 特征值差达 0.498、频率差 7.2e-3 Hz（约为待测效应的 9 倍） |
| 复值通道 TDMD 降偏（SNR 5–10 dB） | 48–92% |
| 实值通道 TDMD 降偏 | 10–29%（显著更弱，已登记为风险 R11） |
| TDMD 的方差代价 | **未观察到**：`std` 比稳定在 0.99–1.01 |
| 单次运算耗时（DMD+TDMD） | 13.7 ms @ `m=200`；26.5 ms @ `m=500` |
| 全部实验推算总工时 | ≈ 1.5 小时（单核） |

> ⚠️ **含噪数值存在一项未决问题**：校验中发现两套测试台给出互相矛盾的结果（详见 `docs/algorithm-spec.md` §8.2.1），根因待查。上表含噪行仅作量级参考，**不可作为实现的正确性依据**。该问题已列为 M3 阶段的前置阻塞任务。

> 检索中也存在另一个同名缩写 TDMD = *Two-stage Difference Mode Decomposition*（Guo et al., *Measurement* 2024，面向轴承/齿轮故障的噪声频带剔除）。该工作与偏差-方差理论无关，**不是本项目所指对象**，特此排除以免混引。

## 6. 文档索引

| 文档 | 内容 |
|---|---|
| [`docs/implementation-plan.md`](docs/implementation-plan.md) | 项目实现规划：可检验假设、系统架构、目录结构、模块划分、里程碑与验收门、风险登记册、**决策记录（ADR）、工作量估算** |
| [`docs/algorithm-spec.md`](docs/algorithm-spec.md) | 算法与统计规格：符号约定、仿真器、SNR 定义、DMD/TDMD 精确算法、指标与置信区间定义、蒙特卡洛协议、秩估计器规格、单元测试判据、基线预验证 |
| [`docs/experiment-protocol.md`](docs/experiment-protocol.md) | 三组实验的参数网格、输出物、判定准则、图表清单与运行矩阵 |

### 6.1 工作量速览

| 项 | 量 |
|---|---|
| 基准总量 | **≈ 20 人日**（含 30% 缓冲） |
| 其中已完成 | ≈ 3 人日（立项归档 + 文档 + 预验证） |
| **剩余** | **≈ 17 人日**（约 3.5 周单人） |
| 代码规模 | ≈ 3 150–3 500 行（含单测） |
| 区间（乐观／基准／悲观） | 13 / 17 / 25 人日 |
| 算力 | 全部实验 ≈ 1.5 小时单核（实测），**不构成约束** |
| 可选扩展 | 方案 3B +4 人日；方案 3C +6 人日 |

两个高不确定项（占总工作量约 1/3，均无法事先估准）：M3 的测试台分歧调查、M4 的 R11 机制解释。详见 [`docs/implementation-plan.md`](docs/implementation-plan.md) §10。

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
- [x] 决策记录（ADR-001 ~ ADR-009）
- [x] 仿真参数设计约束数值校验（C3/C4/C6 处于警戒带）
- [x] 项目虚拟环境
- [x] 关键规格预验证（无噪判据全部通过；含噪部分存在未决问题）
- [ ] **查明测试台分歧、标定 T11 阈值**（阻塞 M3，见 `docs/algorithm-spec.md` §8.2.1）
- [ ] 代码实现（见 `docs/implementation-plan.md` §6 里程碑）

## 9. 主要参考文献

1. M. S. Hemati, C. W. Rowley, E. A. Deem, L. N. Cattafesta. *De-biasing the dynamic mode decomposition for applied Koopman spectral analysis of noisy datasets.* Theoretical and Computational Fluid Dynamics, 31(4): 349–368, 2017. doi:10.1007/s00162-017-0432-2（预印本 arXiv:1502.03854）
2. P. J. Schmid. *Dynamic mode decomposition of numerical and experimental data.* Journal of Fluid Mechanics, 656: 5–28, 2010.
3. J. N. Tu et al. *On dynamic mode decomposition: Theory and applications.* Journal of Computational Dynamics, 1(2): 391–421, 2014.（含噪声对 DMD 影响的讨论）
4. I. Markovsky, S. Van Huffel. *Overview of total least-squares methods.* Signal Processing, 87(10): 2283–2302, 2007.（TLS 的收敛性与统计性质）
