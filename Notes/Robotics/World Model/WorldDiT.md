---
title: "WorldDiT: A Unified Diffusion Architecture for World and Action Modeling"
method_name: "WorldDiT"
authors: [Sen Wang, R. Gnana Praveen, Bidhan Roy, Marcos Villagra]
year: 2026
venue: arXiv
tags: [world-action-model, diffusion-policy, flow-matching, robot-manipulation, visual-prediction, parameter-efficient]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2607.23909v2/
created: 2026-08-04
---

# 论文笔记：WorldDiT: A Unified Diffusion Architecture for World and Action Modeling

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 未公开（arXiv 投稿） |
| 日期 | July 2026 |
| 项目主页 | 未公开 |
| 对比基线 | [[DreamVLA]], [[FlowVLA]], [[Diffusion Policy]] |
| 链接 | [arXiv](https://arxiv.org/abs/2607.23909) / Code: 未公开 |

---

## 一句话总结

> WorldDiT 用单个[[DiT|扩散变换器]]同时生成连续动作序列和预测未来 RGB 图像块，无需大型预训练 VLM 主干，以 4 亿参数量在 LIBERO 基准上达到 Pareto 前沿性能。

---

## 核心贡献

1. **统一 World-Action 建模架构**: 单一[[DiT]]主干在训练时同时优化连续动作预测和未来 RGB 图像块预测两个目标，推理时只执行动作路径，零额外计算开销。
2. **无 VLM 主干的高效策略**: 使用冻结 [[Masked Autoencoding|MAE]] 视觉编码器 + 冻结 [[CLIP]] 文本编码器作为特征提取器，可训练参数仅 1.35 亿，总参数 3.99 亿，成为 sub-billion 参数量基线。
3. **LIBERO Pareto 前沿**: 在 24 个对比方法中，WorldDiT 位于参数量-性能 Pareto 前沿——所有比它性能高的方法都使用了更多参数，所有参数量相当或更少的方法都性能更低。

---

## 问题背景

### 要解决的问题

当前机器人操作策略的主流范式是将动作生成模块附着在数十亿参数的预训练视觉语言模型（VLM）上，参数量巨大，部署成本高，不适合计算资源受限的机器人平台。

### 现有方法的局限

- **VLM 主干依赖**: 如 OpenVLA、π0、GR00T 等方法依赖 7B+ 参数的预训练 LLM/VLM，参数量过大。
- **世界模型与动作模型分离**: 现有方法通常将视觉世界预测与动作生成作为独立模块，未能充分利用两者的协同监督信号。
- **效率 vs. 性能权衡**: 轻量级方法在性能上往往不及大模型方法。

### 本文的动机

单个[[Flow Matching|流匹配]] [[DiT|扩散变换器]]是否能在不依赖大型预训练 VLM 的情况下，同时完成连续动作生成和视觉世界预测，并保持强大的控制性能？若辅助的 RGB 图像块预测目标能在训练时提供额外监督而无需在推理时使用，则能实现"训练时双目标、推理时零开销"。

---

## 方法详解

### 模型架构

WorldDiT 采用**统一扩散变换器**架构：

- **输入**: 语言指令（通过冻结 [[CLIP]] 文本编码器）+ 多视角 RGB 观测（通过冻结 [[Masked Autoencoding|MAE]] 图像编码器）+ 机器人状态（可训练状态编码器）
- **Backbone**: 共享 [[DiT|扩散变换器]] 主干（44 层，隐层维度 1024，16 个注意力头，4 个[[Register Token|注册 token]]）
- **核心模块**: [[Flow Matching|流匹配]] 用于联合建模动作和 RGB 图像块的速度场
- **输出（训练）**: 连续[[Action Chunking|动作块]] $\hat{A}_t \in \mathbb{R}^{H \times d_a}$ + 未来归一化 RGB 图像块目标
- **输出（推理）**: 仅连续动作块（RGB 预测头从推理计算图中剔除）
- **总参数**: 399.084M（可训练：135.107M）

### 核心模块

#### 模块1: 多模态输入处理

**设计动机**: 将异构输入（图像、语言、状态）统一为 token 序列送入共享 DiT 主干

**具体实现**:
- **视觉 token**: 每张 224×224 RGB 帧（经[[CLIP]]预处理）切成 16×16 图像块，每个相机保留均匀采样的 64 个块（128 个 token/帧），使用个体均值和方差统计归一化每个图像块
- **语言 token**: 冻结 [[CLIP]] 文本编码器提取特征，经共享[[Perceiver-IO|Perceiver Resampler]]压缩为固定长度 token
- **状态 token**: 可训练机器人状态编码器将机器人本体感知状态编码为 token
- **观测窗口**: $C=3$ 步历史观测 + $H=7$ 步动作预测 + 1 个未来 RGB 图像块目标（时间偏移 $H$），总窗口长度 $N=10$

#### 模块2: 统一 DiT 主干与双目标监督

**设计动机**: 利用[[Flow Matching|流匹配]]的灵活性，对动作和视觉目标使用相同的速度场预测框架

**具体实现**:
- 同一 DiT 主干并行预测动作速度场 $v_\theta^{act}$ 和 RGB 图像块速度场 $v_\theta^{rgb}$
- 训练时双路损失联合优化；推理时 RGB 预测头完全剥离
- 使用 4 个[[Register Token|注册 token]]稳定注意力计算

#### 模块3: 推理时滚动预测控制

**设计动机**: [[Receding Horizon Control|渐退视界控制]]策略在机器人实时控制中平衡计划质量与响应速度

**具体实现**:
- 每次采样 $H=7$ 步动作块，执行前 3 步，观测新状态后重新规划
- [[Temporal Ensembling|时序集成]]：对重叠预测的动作进行集成，平滑控制输出
- 推理时使用 20 步 Euler 积分离散化流匹配速度场

---

## 关键公式

### 公式1: [[Flow Matching|流匹配]]插值轨迹

$$
x_\tau = (1 - \tau)\varepsilon + \tau y
$$

**含义**: 从标准高斯噪声 $\varepsilon$ 到干净目标 $y$ 的线性插值，定义了训练时的噪声-数据连接路径。

**符号说明**:
- $x_\tau$: 时间步 $\tau$ 处的插值样本
- $\varepsilon \sim \mathcal{N}(0, I)$: 标准高斯噪声
- $y$: 干净目标（动作序列或归一化 RGB 图像块）
- $\tau \sim \mathcal{U}[0, 1]$: 均匀采样的时间步

### 公式2: [[Flow Matching|流匹配]]回归损失

$$
\mathcal{L}_\text{flow} = \left\| v_\theta(x_\tau, \tau, c) - (y - \varepsilon) \right\|_2^2
$$

**含义**: 模型预测的速度场与真实速度 $(y - \varepsilon)$ 之间的 MSE 损失，真实速度即噪声到干净目标方向的向量。

**符号说明**:
- $v_\theta(x_\tau, \tau, c)$: 模型在条件 $c$（多模态上下文）下预测的速度场
- $y - \varepsilon$: 目标速度（从噪声指向干净数据的方向）
- $c$: 条件信息（观测、语言、机器人状态）

### 公式3: 联合多目标训练损失

$$
\mathcal{L}_\text{total} = w_\text{action} \cdot \mathcal{L}_\text{flow}^\text{action} + w_\text{rgb} \cdot \mathcal{L}_\text{flow}^\text{rgb}
$$

**含义**: 将动作流匹配损失和 RGB 图像块流匹配损失加权求和，共同优化统一 DiT 主干。

**符号说明**:
- $w_\text{action} = 0.1$: 动作损失权重
- $w_\text{rgb} = 0.001$: RGB 图像块损失权重（权重较小，作为辅助监督）
- $\mathcal{L}_\text{flow}^\text{action}$: 动作序列的流匹配回归损失
- $\mathcal{L}_\text{flow}^\text{rgb}$: 未来归一化 RGB 图像块的流匹配回归损失

### 公式4: 推理时 Euler 积分（[[Receding Horizon Control|动作采样]]）

$$
\frac{dx_{t,\tau}^\text{act}}{d\tau} = v_\theta^\text{act}\!\left(x_{t,\tau}^\text{act},\, \tau,\, G_t\right), \quad \tau \in [0, 1]
$$

**含义**: 用学习到的速度场对动作 token 进行 Euler 积分，从初始噪声 $x_{t,0}^\text{act} \sim \mathcal{N}(0, I)$ 出发，经 $T_\text{samp}=20$ 步积分得到预测动作块 $\hat{A}_t = x_{t,1}^\text{act} \in \mathbb{R}^{H \times d_a}$。

**符号说明**:
- $x_{t,\tau}^\text{act}$: 时刻 $t$、流时间步 $\tau$ 处的动作 token
- $G_t$: 当前时刻的多模态上下文（观测历史 + 语言 + 状态）
- $T_\text{samp} = 20$: Euler 积分步数
- $H = 7$: 动作块长度；$d_a = 7$: 动作维度

---

## 关键图表

### Figure 1: 参数量 vs. 性能 Pareto 前沿

![Figure 1](https://arxiv.org/html/2607.23909v2/x1.png)

**说明**: 24 个方法在 LIBERO 基准上的总参数量（x 轴）与平均成功率（y 轴）对比。WorldDiT 位于 Pareto 前沿连线上——所有报告更高成功率的方法都使用了更多参数，所有参数量相当或更少的方法成功率都更低。

### Figure 2: 各 LIBERO 任务套件成功示例帧

| 行 | 任务套件 | 帧 1 | 帧 2 | 帧 3 | 帧 4 |
|----|---------|------|------|------|------|
| 1 | LIBERO Spatial | ![](https://arxiv.org/html/2607.23909v2/figures/worlddit_rollouts/worlddit_libero_spatial_frontview_task05_episode01_frame_01.png) | ![](https://arxiv.org/html/2607.23909v2/figures/worlddit_rollouts/worlddit_libero_spatial_frontview_task05_episode01_frame_02.png) | ![](https://arxiv.org/html/2607.23909v2/figures/worlddit_rollouts/worlddit_libero_spatial_frontview_task05_episode01_frame_03.png) | ![](https://arxiv.org/html/2607.23909v2/figures/worlddit_rollouts/worlddit_libero_spatial_frontview_task05_episode01_frame_04.png) |
| 2 | LIBERO Object | ![](https://arxiv.org/html/2607.23909v2/figures/worlddit_rollouts/worlddit_libero_object_agentview_task08_episode01_frame_01.png) | ![](https://arxiv.org/html/2607.23909v2/figures/worlddit_rollouts/worlddit_libero_object_agentview_task08_episode01_frame_02.png) | ![](https://arxiv.org/html/2607.23909v2/figures/worlddit_rollouts/worlddit_libero_object_agentview_task08_episode01_frame_03.png) | ![](https://arxiv.org/html/2607.23909v2/figures/worlddit_rollouts/worlddit_libero_object_agentview_task08_episode01_frame_04.png) |
| 3 | LIBERO Goal | ![](https://arxiv.org/html/2607.23909v2/figures/worlddit_rollouts/worlddit_libero_goal_sideview_task10_episode01_frame_01.png) | ![](https://arxiv.org/html/2607.23909v2/figures/worlddit_rollouts/worlddit_libero_goal_sideview_task10_episode01_frame_02.png) | ![](https://arxiv.org/html/2607.23909v2/figures/worlddit_rollouts/worlddit_libero_goal_sideview_task10_episode01_frame_03.png) | ![](https://arxiv.org/html/2607.23909v2/figures/worlddit_rollouts/worlddit_libero_goal_sideview_task10_episode01_frame_04.png) |
| 4 | LIBERO Long | ![](https://arxiv.org/html/2607.23909v2/figures/worlddit_rollouts/worlddit_libero_10_frontview_task06_episode01_frame_01.png) | ![](https://arxiv.org/html/2607.23909v2/figures/worlddit_rollouts/worlddit_libero_10_frontview_task06_episode01_frame_02.png) | ![](https://arxiv.org/html/2607.23909v2/figures/worlddit_rollouts/worlddit_libero_10_frontview_task06_episode01_frame_03.png) | ![](https://arxiv.org/html/2607.23909v2/figures/worlddit_rollouts/worlddit_libero_10_frontview_task06_episode01_frame_04.png) |

**说明**: 每行展示一个 LIBERO 任务套件中 WorldDiT 成功执行的 4 个时序帧，展示了从初始状态到任务完成的过程。

### Figure 3: 模型架构

![Figure 3](https://arxiv.org/html/2607.23909v2/x2.png)

**说明**: 十步时间窗口包含 3 步观测上下文、7 步目标动作块和 1 个未来归一化 RGB 图像块目标。冻结编码器与可训练投影层构建多模态 token，统一 WorldDiT 主干同时预测动作流速度和 RGB 图像块流速度（即 Slot 监督）。

### Figure 4: 推理流程

![Figure 4](https://arxiv.org/html/2607.23909v2/x3.png)

**说明**: 统一主干编码 3 步观测上下文，通过 20 步流积分采样 7 步动作块。执行前 3 步动作后更新历史窗口并重新规划（[[Receding Horizon Control|渐退视界控制]]）。

### Table 1: LIBERO 各套件成功率对比（与 24 种方法比较）

**A. 无大型预训练 VLM 主干的方法**

| 方法 | Spatial | Object | Goal | Long | Mean | 总参数 |
|------|---------|--------|------|------|------|--------|
| FlowVLA | — | — | — | — | 88.1% | — |
| DreamVLA | — | — | — | — | 92.6% | — |
| **WorldDiT（本文）** | **98.0%** | **97.0%** | **92.8%** | **91.8%** | **94.9%** | **399M** |

**B. 使用大型预训练 VLM 主干的方法（代表性）**

| 方法 | Mean | 参数规模 |
|------|------|---------|
| ACoT VLA | 98.5% | 7B+ |
| （其余 18 种方法，均 > WorldDiT 参数量） | — | 7B+ |

**表格说明**: WorldDiT 以 399M 总参数、135M 可训练参数在"无 VLM 主干"类方法中达到最高性能（94.9%），且位于全体 24 种方法的 Pareto 前沿。

> **重要说明**: 评估报告的 94.9% 包含了用于检查点选择的 300 个 episode（非完全保留的测试集），因此不应作为无偏测试估计。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[LIBERO]]-90 | 90 任务多任务数据 | 多样化操作任务 | 预训练 |
| LIBERO Spatial | 10 任务，500 评估 episodes | 空间关系推理 | 微调 + 测试 |
| LIBERO Object | 10 任务，500 评估 episodes | 物体识别与操作 | 微调 + 测试 |
| LIBERO Goal | 10 任务，500 评估 episodes | 目标条件操作 | 微调 + 测试 |
| LIBERO Long (LIBERO-10) | 10 任务，500 评估 episodes | 长时域复杂操作 | 微调 + 测试 |

**数据处理**: 原始演示转换为固定长度 10 步时间窗口，包含多视角 RGB 观测（主相机 + 手腕相机）、机器人状态、语言指令和动作序列。

### 实现细节

**预训练阶段**:
- **骨干网络**: 共享 DiT（44 层，隐层维度 1024，16 头，4 个注册 token）
- **优化器**: 学习率 $1 \times 10^{-4}$，余弦调度，1 个 warmup epoch
- **Batch Size**: 每卡 40，梯度累积 2 步
- **训练轮数**: 30 epochs（LIBERO-90）
- **硬件**: 8× RTX Pro 6000 GPU（单节点）

**微调阶段**:
- 每个下游任务套件独立微调
- **有效 Batch Size**: 512（每卡 32，梯度累积 2 步，8 卡）
- **混合精度**: bf16

### 可视化结果

Figure 2 展示了 WorldDiT 在四个 LIBERO 任务套件上的成功执行帧。各套件覆盖不同难度：Spatial 和 Object 套件成功率更高（98%+），而 Goal 和 Long 套件（包含更复杂时序推理）相对较低但仍超 91%。

---

## 批判性思考

### 优点
1. **零推理开销的辅助监督**: RGB 图像块预测在训练时提供视觉世界模型监督，推理时完全剥离，无额外计算成本——这是一个优雅的设计选择。
2. **参数效率极强**: 仅 135M 可训练参数，在 LIBERO 上达到 Pareto 最优，为未来的 scaling 研究提供了坚实基线。
3. **清晰的架构设计**: 统一主干 + 冻结编码器的组合使得训练稳定，且方法描述透明，易于复现。

### 局限性
1. **评估偏差**: 报告的 94.9% 包含了用于检查点选择的数据，不是完全无偏估计，结果可能偏乐观。
2. **仅在仿真基准上验证**: 所有实验均在 LIBERO 仿真环境中完成，未见真实机器人部署验证。
3. **RGB 预测目标影响尚不明确**: 作者也承认"归一化未来 RGB 目标如何影响控制行为"仍是开放问题，论文未做消融实验验证辅助目标的实际贡献。
4. **缺少消融实验**: 论文未对关键设计选择（如 RGB 权重 $w_\text{rgb}$、patch 数量、注册 token 数量等）进行系统消融，限制了对方法各组件贡献的理解。

### 潜在改进方向
1. **消融辅助目标**: 系统验证 RGB 图像块预测对控制性能的具体贡献，优化 $w_\text{rgb}$ 权重。
2. **真实机器人部署**: 在真实硬件上验证方法的迁移能力，探索 Sim2Real 效果。
3. **Scaling 研究**: 作者提出的 Pareto 前沿是否在更大参数量下仍成立，值得系统探索。
4. **独立专家分解**: 作者提到的将模型分解为独立训练专家以支持异构硬件的思路值得深入研究。

### 可复现性评估
- [ ] 代码开源（未发布）
- [ ] 预训练模型（未发布）
- [x] 训练细节完整（论文描述详细）
- [x] 数据集可获取（LIBERO 公开可用）

---

## 关联笔记

### 基于
- [[Diffusion Policy]]: WorldDiT 在扩散策略框架基础上引入视觉世界模型辅助目标
- [[Flow Matching]]: 训练和推理的核心数学框架，用于速度场预测
- [[DiT]]: 扩散变换器架构，WorldDiT 的主干网络类型
- [[CLIP]]: 冻结视觉和语言编码器来源
- [[Masked Autoencoding]]: 冻结 MAE 图像编码器提供视觉特征

### 对比
- [[DreamVLA]]: 同为"无 VLM 主干"类方法，WorldDiT 超越其 92.6% mean 达到 94.9%
- [[FlowVLA]]: 同类方法，WorldDiT 超越其 88.1% mean

### 方法相关
- [[Flow Matching]]: 核心训练目标
- [[Action Chunking]]: 7 步动作块预测
- [[Receding Horizon Control]]: 推理时执行 3 步后重规划的控制策略
- [[Temporal Ensembling]]: 推理时对重叠预测的集成平滑
- [[Register Token]]: DiT 主干中用于稳定注意力的特殊 token
- [[World Model]]: WorldDiT 的辅助目标——预测未来视觉世界状态

### 硬件/数据相关
- [[LIBERO]]: 评估基准，四个操作任务套件

---

## 速查卡片

> [!summary] WorldDiT: A Unified Diffusion Architecture for World and Action Modeling
> - **核心**: 单个 DiT 同时做动作生成和未来 RGB 预测，推理时剥离视觉头
> - **方法**: Flow Matching + 双目标训练（$w_{action}=0.1$, $w_{rgb}=0.001$），20 步 Euler 积分，渐退视界控制
> - **结果**: 94.9% LIBERO 平均成功率，399M 总参数，Pareto 最优
> - **代码**: 未公开

---

*笔记创建时间: 2026-08-04*
