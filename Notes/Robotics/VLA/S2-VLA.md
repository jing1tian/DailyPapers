---
title: "S²-VLA: State-Space Guided Vision-Language-Action Models for Long-Horizon Manipulation"
method_name: "S2-VLA"
authors: [Zhipeng Xie, Zongyi Han, Xiangyi Wei, Shiliang Sun, Yang Li, Jing Zhao]
year: 2026
venue: IJCAI 2026
tags: [vla, long-horizon-manipulation, belief-state, adaptive-attention, action-chunking]
zotero_collection: Robotics/VLA
image_source: local
arxiv_html: https://arxiv.org/html/2606.27872
created: 2026-06-30
---

# 论文笔记：S²-VLA: State-Space Guided Vision-Language-Action Models for Long-Horizon Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | East China Normal University; Shanghai Jiao Tong University |
| 日期 | June 2026 |
| 项目主页 | 未提供 |
| 对比基线 | [[MemoryVLA]], [[OpenVLA-OFT]], [[CronusVLA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.27872) / Code 未公开 |

---

## 一句话总结

> S²-VLA 通过信念状态引导的自适应注意力机制，让 2B 参数的 VLA 在长时序操作任务中超越 7B 级别模型，核心在于动态感知任务阶段并按需融合视觉、语义和动作信息。

---

## 核心贡献

1. **信念驱动的 VLA 范式（Belief-Driven VLA Paradigm）**: 引入动态信念状态 $\mathbf{b}_t$ 追踪任务进度与执行保真度，通过 [[GRU]] 递归压缩历史动作与感知信息
2. **状态空间引导的自适应注意力（SSGAA）**: 三条并行注意力分支（视觉、意图、动作序列）由信念状态生成软门控权重，实现阶段感知的信息融合
3. **高效小模型超越大模型**: 仅 2B 参数的 S²-VLA 在 LIBERO 和 SimplerEnv 上超越 7B+ 规模的 MemoryVLA、CronusVLA、UnifiedVLA，推理速度达 80.8 Hz

---

## 问题背景

### 要解决的问题

[[VLA（视觉-语言-动作模型）]] 在长时序操作任务（long-horizon manipulation）中性能显著下降：任务越长，累积误差传播（cumulative error propagation）越严重，导致后期动作预测失准。

### 现有方法的局限

- **静态压缩方法**（如 [[OpenVLA-OFT]]）：仅关注最终压缩嵌入，丢失空间细节
- **静态多级融合方法**（如 SmolVLA、VLA-Adapter）：用固定权重聚合多层特征，无法根据任务阶段动态调整
- 两类方法均无法感知任务进度，在精确定位阶段与语义规划阶段之间无法灵活切换

### 本文的动机

不同任务阶段需要不同信息权重：接近目标时需要高精度视觉定位；切换任务阶段时需要语义意图理解；稳态运动时需要动作序列的时序一致性。作者认为，显式建模任务阶段信息（信念状态）并用其动态控制注意力权重，是解决长时序累积误差的关键。

---

## 方法详解

### 模型架构

S²-VLA 采用 **自回归 Transformer + 递归状态空间** 混合架构：

- **输入**: 语言指令 $\mathbf{L}_t$ + 多视角观测 $\mathbf{V}_t$（ViT 编码）+ 本体感知反馈 $\mathbf{P}_t$
- **Backbone**: [[Qwen3-VL]]（2B 参数）
- **核心模块**: [[SSGAA|状态空间引导自适应注意力]] 在中间层进行相位感知融合，[[GRU]] 压缩历史信息为信念状态
- **输出**: [[Action Chunking|动作块]] $\hat{\mathbf{A}}_{t:t+K-1}$，预测未来 K 步动作
- **总参数**: 2B

### 统一输入构造

$$
\mathbf{X} = [\mathbf{F}_t;\, l_1,\ldots,l_{N_l};\, \tilde{a}_1,\ldots,\tilde{a}_{N_a}] \in \mathbb{R}^{(N_v + N_l + N_a) \times d}
$$

**符号说明**:
- $\mathbf{F}_t$：ViT 提取的视觉特征，维度 $N_v \times d$
- $l_1,\ldots,l_{N_l}$：语言指令 token
- $\tilde{a}_1,\ldots,\tilde{a}_{N_a}$：可学习意图 token（learnable intent tokens），捕获任务上下文语义

### 核心模块

#### 模块 1：信念状态更新（Belief State Update）

**设计动机**: 利用 [[GRU|门控循环单元]] 将历史动作序列与本体感知反馈压缩为紧凑的任务进度表示，无需显式监督，通过端到端优化自发涌现出阶段信息与执行保真度编码。

**具体实现**:

$$
\mathbf{b}_t = f_\phi(\mathbf{b}_{t-1},\, \mathbf{a}_{t-K:t-1},\, \mathbf{P}_t)
$$

- $\mathbf{b}_t \in \mathbb{R}^{d_b}$：当前时刻信念状态向量
- $f_\phi$：[[GRU]] 参数化的更新函数
- $\mathbf{a}_{t-K:t-1}$：近期执行的动作序列
- $\mathbf{P}_t$：本体感知反馈（关节角、末端执行器位姿等）

#### 模块 2：状态空间引导自适应注意力（SSGAA）

**设计动机**: 在 Transformer 的特定层（语义最丰富的中间层，默认第 12 层）插入三支并行注意力分支，由信念状态生成软门控权重，实现阶段感知的动态信息融合。

**三条并行分支**:

1. **动作序列自注意力（Action Sequence Self-Attention）**: 对 K 步预测动作序列建模时序依赖，保证运动轨迹的时序连贯性
2. **低层视觉交叉注意力（Low-Level Visual Cross-Attention）**: 对 ViT 编码的细粒度视觉特征做 [[Cross-Attention|交叉注意力]]，提供精确的空间定位能力
3. **高层意图交叉注意力（High-Level Intent Cross-Attention）**: 对可学习意图 token 做交叉注意力，捕获任务语义，辅助长时序规划

**信念引导门控融合（Belief-Guided Gating）**:

$$
[g_\text{vis}^{(l)},\, g_\text{ite}^{(l)},\, g_\text{act}^{(l)}]^\top = \text{Softmax}\!\left(\text{MLP}_g^{(l)}(\mathbf{b}_t^{(l)})\right)
$$

$$
\mathbf{H}^{(l)} = g_\text{vis}^{(l)} \cdot \mathbf{O}_\text{vis}^{(l)} + g_\text{ite}^{(l)} \cdot \mathbf{O}_\text{ite}^{(l)} + g_\text{act}^{(l)} \cdot \mathbf{O}_\text{act}^{(l)}
$$

**符号说明**:
- $g_\text{vis}^{(l)}, g_\text{ite}^{(l)}, g_\text{act}^{(l)}$：视觉、意图、动作分支的软门控权重（Softmax 归一化，三者之和为 1）
- $\mathbf{O}_\text{vis}^{(l)}, \mathbf{O}_\text{ite}^{(l)}, \mathbf{O}_\text{act}^{(l)}$：三分支的输出特征
- $\mathbf{b}_t^{(l)}$：第 $l$ 层的信念状态（从全局 $\mathbf{b}_t$ 投影得到）
- $\text{MLP}_g^{(l)}$：门控权重生成的小型 MLP

---

## 关键公式

### 公式 1：[[VLA（视觉-语言-动作模型）|标准 VLA 公式]]

$$
\mathbf{A}_{t:t+K-1} = \pi_\text{VLA}(\mathbf{V}_t,\, \mathbf{L}_t,\, \mathbf{P}_t)
$$

**含义**: 标准 VLA 模型根据当前视觉观测、语言指令和本体感知预测 K 步动作序列

**符号说明**:
- $\mathbf{V}_t$：多视角视觉观测
- $\mathbf{L}_t$：语言任务指令
- $\mathbf{P}_t$：本体感知反馈
- $K$：预测动作步数（Action Chunk 长度）

### 公式 2：[[信念状态|S²-VLA 信念条件化公式]]

$$
\mathbf{A}_{t:t+K-1} = \pi'_\text{VLA}(\mathbf{V}_t,\, \mathbf{L}_t,\, \mathbf{P}_t \mid \mathbf{b}_t)
$$

**含义**: S²-VLA 在标准 VLA 基础上额外条件化信念状态 $\mathbf{b}_t$，使预测具备时序连贯性和任务阶段感知

### 公式 3：[[GRU|信念状态递归更新]]

$$
\mathbf{b}_t = f_\phi(\mathbf{b}_{t-1},\, \mathbf{a}_{t-K:t-1},\, \mathbf{P}_t)
$$

**含义**: 利用 GRU 将前一时刻信念、近期动作历史和当前本体感知更新为新的信念状态

**符号说明**:
- $f_\phi$：GRU 参数化的更新函数
- $\mathbf{b}_{t-1}$：上一时刻信念状态
- $\mathbf{a}_{t-K:t-1}$：近期执行的 K 步动作

### 公式 4：[[SSGAA|门控权重生成]]

$$
[g_\text{vis}^{(l)},\, g_\text{ite}^{(l)},\, g_\text{act}^{(l)}]^\top = \text{Softmax}\!\left(\text{MLP}_g^{(l)}(\mathbf{b}_t^{(l)})\right)
$$

**含义**: 由信念状态通过 MLP 映射后 Softmax 归一化，生成三分支的软门控权重

**符号说明**:
- $g_\text{vis}^{(l)}$：视觉分支权重
- $g_\text{ite}^{(l)}$：意图分支权重
- $g_\text{act}^{(l)}$：动作序列分支权重

### 公式 5：[[SSGAA|多分支信息融合]]

$$
\mathbf{H}^{(l)} = g_\text{vis}^{(l)} \cdot \mathbf{O}_\text{vis}^{(l)} + g_\text{ite}^{(l)} \cdot \mathbf{O}_\text{ite}^{(l)} + g_\text{act}^{(l)} \cdot \mathbf{O}_\text{act}^{(l)}
$$

**含义**: 三分支输出经软门控加权融合，得到当前层的综合特征表示

### 公式 6：[[Action Chunking|并行动作解码]]

$$
\hat{\mathbf{A}}_{t:t+K-1} = \text{LN}(\mathbf{H}^{(L_\text{out})}) \mathbf{W}_\text{out}^\top + \boldsymbol{\beta}_\text{out}
$$

**含义**: 将最终层输出经 LayerNorm 和线性投影后得到 K 步预测动作

**符号说明**:
- $\text{LN}$：[[Layer Normalization|层归一化]]
- $\mathbf{W}_\text{out}$：输出投影矩阵
- $\boldsymbol{\beta}_\text{out}$：偏置项

### 公式 7：[[均方误差|训练损失函数]]

$$
\mathcal{L}_\text{total} = \mathcal{L}_\text{action} = \frac{1}{K} \sum_{k=0}^{K-1} \|\mathbf{a}_{t+k} - \hat{\mathbf{a}}_{t+k}\|_2^2
$$

**含义**: 对 K 步预测动作与真值动作的均方误差取平均，端到端联合优化 backbone、信念更新模块和 SSGAA

**符号说明**:
- $\mathbf{a}_{t+k}$：第 $t+k$ 步真值动作
- $\hat{\mathbf{a}}_{t+k}$：模型预测的第 $t+k$ 步动作
- $K$：Action Chunk 长度

---

## 关键图表

### Figure 1：S²-VLA vs. 静态融合方法对比

![[Abs_v2.png]]

**说明**: 对比 S²-VLA 与静态融合方法（如 OpenVLA-OFT）的差异。静态方法在早期阶段过度关注特定信息来源导致"早期偏差（early-stage bias）"，而 S²-VLA 的 [[SSGAA]] 能根据任务阶段动态调整注意力权重，有效缓解累积误差。

### Figure 2：S²-VLA 整体架构与工作流

![[SSVLA.png]]

**说明**: 展示 S²-VLA 的完整架构。多模态输入（视觉 + 语言 + 本体感知）经 [[Qwen3-VL]] backbone 处理，核心 [[SSGAA]] 模块在第 12 层插入，信念状态 $\mathbf{b}_t$ 由 [[GRU]] 递归更新并通过门控 MLP 生成三分支的自适应权重。三条并行注意力分支（低层视觉交叉注意力、高层意图交叉注意力、动作序列自注意力）的输出经软门控融合，最终通过线性头解码为 [[Action Chunking|动作块]]。

### Figure 3：真实世界 ALOHA 平台评估

![[realworldperf.jpg]]

**说明**: 在真实 ALOHA 双臂机器人平台（30 Hz 控制，多视角相机）上的定性对比结果。S²-VLA 在拾取放置、叠放、桌面整理、餐具分拣等任务中均优于 [[ACT]] 和 [[Pi0-FAST|π₀-FAST]] 基线，展示了在真实物理系统上的泛化能力。

### Figure 4：门控权重轨迹与任务阶段对齐

![[x1.png]]

**说明**: 可视化"将奶油芝士盒和黄油都放入篮子"任务的 LIBERO-Long 完整 rollout 中门控权重动态变化：
- **精确定位阶段**（接近目标物体时）：$g_\text{vis}$ 达到峰值，视觉注意力主导
- **阶段切换时**（抓取/放置时）：$g_\text{ite}$ 上升，语义意图理解主导
- **稳态运动阶段**：$g_\text{act}$ 偏高，动作序列自注意力主导时序一致性

这一分析证明模型自发学到了阶段感知的自适应焦点，而非人工设计的规则。

### Table 1：LIBERO 基准测试结果

| 方法 | 参数量 | Spatial | Object | Goal | Long | **Avg** |
|------|--------|---------|--------|------|------|---------|
| ACT | - | 84.7% | 85.7% | 84.7% | 72.6% | 81.9% |
| OpenVLA-OFT | 7B | 95.6% | 96.7% | 93.6% | 88.3% | 93.6% |
| TraceVLA | 7B | 96.4% | 98.0% | 95.6% | 90.3% | 95.1% |
| MemoryVLA | 7B | 97.2% | 98.4% | 96.8% | 94.4% | 96.7% |
| CronusVLA | 7B | 97.6% | 98.4% | 97.6% | 94.4% | 97.0% |
| UnifiedVLA | 8.5B | 97.6% | 98.0% | 96.8% | 89.6% | 95.5% |
| **S²-VLA（Ours）** | **2B** | **98.4%** | **99.6%** | **98.4%** | **96.4%** | **98.2%** |

**关键发现**: S²-VLA 以 2B 参数在全部四个 LIBERO 子集（尤其是最难的 Long 子集 96.4%）上超越所有 7B+ 基线，平均成功率 98.2%，验证了架构设计而非参数规模是长时序任务的关键。

### Table 2：SimplerEnv-Bridge（WidowX）评测结果

| 方法 | Spoon on Towel | Carrot on Plate | Stack Cube | Eggplant in Basket | **Avg** |
|------|---------------|-----------------|------------|-------------------|---------|
| OpenVLA | - | - | - | - | 32.4% |
| TraceVLA | 75.0% | 75.0% | 16.7% | 91.7% | 64.6% |
| MemoryVLA | 83.3% | 75.0% | 33.3% | 95.8% | 71.9% |
| **S²-VLA（Ours）** | **83.3%** | **87.5%** | **41.7%** | **100%** | **78.1%** |

**关键发现**: 在 sim-to-real 迁移场景下 S²-VLA 仍保持优势，Eggplant in Basket 达到 100% 成功率，平均 78.1% 超越 MemoryVLA（71.9%）6.2 个百分点。

### Table 3：消融实验（LIBERO-Long，成功率 %）

| 配置 | 成功率 | 说明 |
|------|--------|------|
| 所有层使用静态门控（Static, all layers） | 94.4% | 固定权重无法适应阶段变化 |
| 仅第 6 层单层门控 | 95.2% | 过早插入，语义不够丰富 |
| **仅第 12 层单层门控（Full Model）** | **96.4%** | 最优——语义最丰富的中间层 |
| 仅第 18 层单层门控 | 95.8% | 过深，特征过于抽象 |
| 多层门控（第 6、12、18 层） | 94.4% | 多层门控引入不稳定性 |
| 去除信念状态引导（w/o belief state） | 95.8% | 信念状态贡献 0.6 个百分点 |

**关键发现**: 门控插入位置极为关键，第 12 层（语义最丰富的中间层）单层门控最优；多层门控反而因不稳定性导致性能下降；信念状态引导是自适应机制的必要组成部分。

### Table 4：QwenVL 系列 Backbone 对比（LIBERO-Long）

| 模型 | 参数量 | 成功率 |
|------|--------|--------|
| Qwen2.5-VL + FAST head | 3B | 90.2% |
| Qwen3-VL + OFT head | 4B | 93.8% |
| **S²-VLA（Qwen3-VL + SSGAA）** | **2B** | **96.4%** |

**关键发现**: 在相同 Backbone 家族下，SSGAA 头相比 OFT 头提升 2.6%，且参数量更少（2B vs 4B），证明架构创新优于简单的 Head 替换或参数扩增。

### Table 5：推理效率对比

| 方法 | 吞吐量（Hz） | VRAM |
|------|------------|------|
| TraceVLA | 4.3 Hz | - |
| CronusVLA | 8.7 Hz | - |
| OpenVLA-OFT | 71.4 Hz | - |
| **S²-VLA** | **80.8 Hz** | **7 GB** |

**关键发现**: S²-VLA 推理速度 80.8 Hz，远超同类方法，且仅需 7 GB VRAM，具备实际部署潜力。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[LIBERO]] | 4 子集（Spatial/Object/Goal/Long） | 固定轨迹仿真，脚本化初始场景采样 | 主要基准测试 |
| [[SimplerEnv]] | Bridge 子集，WidowX 机器人 | 高保真仿真，弥合 real-to-sim 控制差距 | 迁移评估 |
| ALOHA | 真实双臂平台 | 30 Hz 控制，多视角相机 | 真实世界验证 |

### 实现细节

- **Backbone**: [[Qwen3-VL]]（2B 参数），端到端联合微调
- **信念状态维度**: $d_b$（未公开具体值）
- **Action Chunk 长度**: K 步（未公开具体值）
- **门控插入位置**: 第 12 层（共 ~24 层 Transformer）
- **训练损失**: 动作预测 MSE，无额外正则项
- **硬件**: 4× NVIDIA H100 GPU
- **推理**: 时序递归循环，持续状态估计
- **代码**: 未开源

### 可视化结果

Figure 4 的门控轨迹分析揭示了模型的内部机制：在 LIBERO-Long 的"将奶油芝士盒和黄油放入篮子"任务中，模型自发学会在精确定位阶段提升视觉权重、在语义过渡阶段提升意图权重，与人类直觉一致，表明 SSGAA 的可解释性良好。

---

## 批判性思考

### 优点

1. **高效架构**: 2B 参数超越 7B+ 模型，证明架构创新比暴力扩参更有效
2. **可解释性强**: 门控权重轨迹直接可视化，与任务阶段直觉对齐，不是黑箱
3. **工程友好**: 80.8 Hz + 7 GB VRAM，具备真实部署可行性
4. **消融充分**: 消融实验精细覆盖门控位置、层数、信念状态等关键设计选择

### 局限性

1. **单一动作头**: 仅使用 MSE 回归，未兼容扩散模型或 Flow-Matching 等更强的生成框架，论文作者自认是未来方向
2. **门控位置人工选择**: 第 12 层的选择基于消融搜索而非理论推导，不同任务/不同 backbone 是否通用存疑
3. **代码未开源**: 难以复现和二次研究
4. **LIBERO 局限**: LIBERO 是固定轨迹仿真 benchmark，对泛化性（open-vocabulary / novel scenes）的考察有限

### 潜在改进方向

1. 将 SSGAA 与扩散策略或 Flow-Matching 结合，提升复杂多模态动作分布建模能力
2. 自动化门控层位置选择（如基于注意力熵的自适应插入）
3. 扩展到移动操作（mobile manipulation）等更长时序场景

### 可复现性评估

- [ ] 代码开源
- [ ] 预训练模型
- [x] 训练细节完整（硬件、损失函数、backbone）
- [x] 数据集可获取（LIBERO/SimplerEnv 均公开）

---

## 关联笔记

### 基于

- [[Qwen3-VL]]: 使用 Qwen3-VL 作为 VLM backbone
- [[GRU]]: 信念状态更新的具体实现
- [[Action Chunking]]: 并行预测 K 步动作的输出形式

### 对比

- [[MemoryVLA]]: 7B 参数，LIBERO 长时序任务的主要对比方法（96.7% vs 98.2%）
- [[OpenVLA-OFT]]: 静态压缩注意力方法，7B 参数
- [[CronusVLA]]: 7B 参数长时序 VLA，SimplerEnv 对比基线
- [[ACT]]: 经典模仿学习方法，ALOHA 真实世界对比基线
- [[Pi0-FAST]]: π₀-FAST，ALOHA 真实世界对比基线

### 方法相关

- [[SSGAA]]: 核心创新——状态空间引导自适应注意力
- [[信念状态]]: 任务进度追踪的核心表示
- [[Cross-Attention|交叉注意力]]: SSGAA 视觉分支和意图分支的底层机制
- [[VLA（视觉-语言-动作模型）]]: 所属方法类别

### 硬件/数据相关

- [[LIBERO]]: 主要评估 benchmark
- [[SimplerEnv]]: sim-to-real 迁移评估
- [[ALOHA]]: 真实双臂机器人平台

---

## 速查卡片

> [!summary] S²-VLA (IJCAI 2026)
> - **核心**: 信念状态引导的自适应注意力，使 VLA 感知任务阶段动态调整信息融合权重
> - **方法**: SSGAA（三分支并行注意力 + 软门控 + GRU 信念状态）插入 Transformer 第 12 层
> - **结果**: 2B 模型 LIBERO 平均 98.2%（超越 7B MemoryVLA 的 96.7%），推理 80.8 Hz / 7GB VRAM
> - **代码**: 未公开

---

*笔记创建时间: 2026-06-30*
