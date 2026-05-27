---
title: "EXPO-FT: Sample-Efficient Reinforcement Learning Finetuning for Vision-Language-Action Models"
method_name: "EXPO-FT"
authors: [Perry Dong, Kuo-Han Hung, Tian Gao, Dorsa Sadigh, Chelsea Finn]
year: 2026
venue: arXiv
tags: [vla-finetuning, reinforcement-learning, robot-manipulation, sample-efficient-rl, human-in-the-loop, action-chunking]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2605.25477
created: 2026-05-27
---

# 论文笔记：EXPO-FT: Sample-Efficient Reinforcement Learning Finetuning for Vision-Language-Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Stanford University |
| 日期 | May 2026 |
| 项目主页 | [pd-perry.github.io/expo-ft](https://pd-perry.github.io/expo-ft/) |
| 对比基线 | [[HIL-SERL]], [[HG-DAgger]], [[DSRL]], [[SFT]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.25477) / [Code](https://pd-perry.github.io/expo-ft/) |

---

## 一句话总结

> EXPO-FT 通过将 [[EXPO]] 算法扩展至 [[Action Chunking|动作块]] + 人类干预，实现对预训练 [[VLA]] 模型的高效 RL 微调，平均仅需 19.1 分钟在线数据即可在所有评估任务上达到 30/30 完美成功率。

---

## 核心贡献

1. **VLA 的 RL 微调系统**: 将 EXPO 算法扩展支持 [[Action Chunking|动作块]]，直接对完整预训练 [[VLA]] 进行端到端 RL 微调，无需辅助策略。
2. **人类在环干预集成**: 在 RL 训练过程中融入人类干预机制，对动作块中的单步动作进行纠正，显著降低探索负担并加速收敛。
3. **解耦学习器-执行器架构**: 服务器-学习器分离的高效训练架构，支持同步与异步通信，大幅减少大型 VLA 模型的计算开销。

---

## 问题背景

### 要解决的问题

预训练的 [[VLA]] 模型（如 π0、OpenVLA）虽然在大量数据上训练具备泛化能力，但在实际部署时仍难以达到可靠的任务成功率。如何利用有限的在线机器人数据，通过 [[强化学习]] 微调 VLA 模型以达到部署级可靠性是核心问题。

### 现有方法的局限

- **从零开始 RL**（HIL-SERL、SERL）：高成功率但使用小型高斯策略，无法利用预训练的大模型先验知识，数据收集量要求高
- **监督微调（SFT）**：行为克隆无法超越示范数据分布的上限，难以通过奖励信号持续优化
- **潜空间优化**（DSRL）：在潜空间中引导扩散策略，受限于先验分布，无法探索超出初始策略的行为
- **HG-DAgger**：仅靠模仿学习，没有奖励优化，泛化能力有限
- **现有 VLA RL 微调**：要么冻结 VLA 主干（丢失表达能力），要么训练效率极低（需要大量交互）

### 本文的动机

[[EXPO]] 算法能对扩散/流策略进行有原则的离策略微调，但尚未支持 [[Action Chunking|动作块]] 和人类干预。EXPO-FT 将其扩展，使其能够直接微调真实 VLA 模型，结合人类监督降低探索负担，同时通过高效的系统架构克服大模型的计算瓶颈。

---

## 方法详解

### 模型架构

EXPO-FT 采用**编辑策略 + Q 函数**的 [[Actor-Critic]] 架构：

- **输入**: 相机观测 $o_t$ + 语言指令 $l$（通过 VLA 视觉编码器处理）
- **Backbone (Actor)**: 预训练 VLA 的视觉编码器（参数可微调）
- **Backbone (Critic)**: 轻量级 [[ResNet-50]] 以减少计算开销
- **核心模块**: [[EXPO]] 编辑策略用于 [[Action Chunking|动作块]] 优化
- **输出**: 动作块 $a_{t:t+H}$（H 个未来动作），每步执行 $C \leq H$ 个动作

### 核心模块

#### 模块 1: 时间扩展的编辑策略（Temporally Extended Edit Policy）

**设计动机**: 标准 EXPO 仅处理单步动作，VLA 预测动作块需要扩展至时序维度

**具体实现**:
- 编辑策略 $\pi_{\text{edit}}$ 在给定 VLA 基础策略输出 $a_{t:t+C}$ 的条件下，预测残差编辑量 $\hat{a}_{t:t+C}$
- 最终执行动作 $\tilde{a}_{t:t+C} = a_{t:t+C} + \hat{a}_{t:t+C}$
- 编辑量受 [[SAC|软演员评论家]] 框架中熵正则化约束，防止过度偏离预训练先验

#### 模块 2: 动作块 Q 函数（Chunked Q-Function）

**设计动机**: 稀疏奖励下需要对多步动作序列进行价值估计

**具体实现**:
- 使用 [[REDQ]] 风格的 Q 网络集成（10 个 Q 网络），降低过估计偏差
- Q 函数接收状态和完整动作块作为输入
- 折扣因子 $\gamma$ 应用于块级转换 $s_{t+C}$，而非单步

#### 模块 3: 在线策略选择（On-the-fly Policy Selection）

**设计动机**: 在探索和利用之间找到平衡，利用 Q 值选择最优候选动作

**具体实现**:
- 同时采样 N 个基础动作 $\{a_i\}$ 和 N 个编辑动作 $\{\tilde{a}_i\}$
- 通过 Q 网络评分选择最优动作块执行
- 这使得策略在训练初期能利用预训练先验，后期转向 RL 优化的编辑策略

#### 模块 4: 人类在环干预（Human-in-the-Loop Interventions）

**设计动机**: 降低复杂任务中的随机探索代价，提供人类先验知识

**具体实现**:
- 操作员在训练时可对动作块中的任意单步动作进行干预纠正
- 干预数据直接加入 [[经验回放]] 缓冲区，同时用于更新 VLA 和 Q 函数
- 随着训练进行，干预率自然下降，机器人逐渐实现自主学习

#### 模块 5: 解耦训练架构

**设计动机**: 数十亿参数的 VLA 推理计算量大，需要高效的训练-推理分离

**具体实现**:
- **服务器组件**: 负责 VLA 训练和推理（GPU 密集型）
- **学习器进程**: 负责环境步进和 RL 更新
- 支持同步和异步通信模式，适配不同硬件配置

---

## 关键公式

### 公式 1: [[EXPO|编辑策略损失函数]]

$$
\mathcal{L}(\pi_{\text{edit}}) = -\mathbb{E}\left[Q_\phi(s_t, a_{t:t+C} + \hat{a}_{t:t+C}) - \alpha \log \pi_{\text{edit}}(\hat{a}_{t:t+C} | s_t, a_{t:t+C})\right]
$$

**含义**: 最大化 Q 值的同时保持编辑动作的熵（防止过度确定性），$\alpha$ 平衡探索与利用

**符号说明**:
- $\pi_{\text{edit}}$: 编辑策略，预测对基础动作的残差修正量
- $\hat{a}_{t:t+C}$: 编辑量（残差动作块）
- $a_{t:t+C}$: VLA 基础策略预测的动作块，包含 $C$ 步动作
- $Q_\phi$: 参数为 $\phi$ 的 Q 函数，评估状态-动作对的价值
- $\alpha$: 熵正则化温度系数

### 公式 2: [[Actor-Critic|Q 函数时序差分损失]]

$$
\mathcal{L}(\phi) = \mathbb{E}\left[\left(r_t + \gamma Q_{\phi'}(s_{t+C}, \tilde{a}^*_{t+C:t+2C}) - Q_\phi(s_t, a_{t:t+C})\right)^2\right]
$$

**含义**: 通过 TD 误差最小化训练 Q 函数，目标值使用目标网络 $\phi'$ 计算，折扣步长为动作块长度 $C$

**符号说明**:
- $r_t$: 时刻 $t$ 的奖励（稀疏二值奖励，完成任务得 1）
- $\gamma$: 折扣因子（实验中取 0.99）
- $Q_{\phi'}$: 目标 Q 网络（慢速更新副本）
- $s_{t+C}$: 执行 $C$ 步动作块后的下一状态
- $\tilde{a}^*_{t+C:t+2C}$: 下一时刻通过策略选择的最优动作块

### 公式 3: [[Actor-Critic|在线策略选择]]

$$
\tilde{a}^* = \arg\max_{a \in \bigcup_{i=1}^N \{a_i, \tilde{a}_i\}} Q_\phi(s, a)
$$

**含义**: 从 N 个基础动作和 N 个编辑动作的并集中，选择 Q 值最高的动作块执行

**符号说明**:
- $N$: 采样候选动作数量
- $a_i$: VLA 基础策略采样的第 $i$ 个动作块
- $\tilde{a}_i$: 编辑策略修正后的第 $i$ 个动作块
- $Q_\phi(s, a)$: Q 函数对状态-动作对的评分

---

## 关键图表

### Figure 1: 训练成功率对比曲线

![Figure 1 - Training Success Rates](https://arxiv.org/html/2605.25477/figures/figure_1.png)

**说明**: EXPO-FT 在所有评估任务上收敛至 30/30 完美成功率，而 HIL-SERL、DSRL、HG-DAgger 等基线方法收敛不稳定或达不到完美性能。

### Figure 2: EXPO-FT 系统架构图

![Figure 2 - System Overview](https://arxiv.org/html/2605.25477v1/x1.png)

**说明**: 展示服务器-学习器解耦架构。VLA 服务器处理高计算量的推理和训练，学习器进程负责环境交互与 RL 更新，两者通过异步通信协同工作。

### Figure 3: 8 项真实世界操作任务

![Figure 3 - Eight Manipulation Tasks](https://arxiv.org/html/2605.25477v1/x2.png)

**说明**: 评估套件包含 8 个具有挑战性的操作任务，涵盖接触丰富型（Egg Flip）、精密对齐型（String Light Routing、Flower Insertion）、动态力控型（Pool Shot）等不同难度类型。

### Figure 4: 各任务训练曲线（成功率 + 干预率）

![Figure 4 - Training Curves](https://arxiv.org/html/2605.25477v1/x3.png)

**说明**: 各任务的训练过程中，成功率单调上升，人类干预率随训练进行自然下降，验证了机器人逐步实现自主学习的效果。

### Figure 5: 各任务的 Episode 时长变化

![Figure 5 - Episode Time](https://arxiv.org/html/2605.25477v1/x12.png)

**说明**: Episode 完成时间随训练缩短，表明策略效率不断提升，机器人操作更流畅高效。

### Figure 6: 成功任务执行序列图

![Figure 6 - Task Execution Strips](https://arxiv.org/html/2605.25477v1/x21.png)

**说明**: 展示 EXPO-FT 训练后策略在各任务上的完整执行序列，包括 String Lights、Pool Shot、Flower Insert 等高难度任务的成功案例。

### Figure 7: 初始状态随机化可视化

![Figure 7 - Initial State Randomization](https://arxiv.org/html/2605.25477/figures/initial.png)

**说明**: 橙色区域标示各任务中物体初始位置的随机化范围，EXPO-FT 在这些大范围随机化条件下仍保持完美成功率。

### Table 1: 成功率对比（各基线方法，30 次试验）

| 任务 | SFT | HG-DAgger | DSRL | HIL-SERL | **EXPO-FT** |
|------|-----|-----------|------|----------|-------------|
| Egg Flip | 16/30 | 18/30 | 15/30 | 13/30 | **30/30** |
| Cube Pick | 22/30 | 26/30 | 24/30 | 0/30 | **30/30** |
| Pool Shot | 23/30 | 14/30 | 25/30 | 1/30 | **30/30** |
| Flower Insertion | 14/30 | 24/30 | 12/30 | 8/30 | **30/30** |
| **平均** | 18.8/30 | 20.5/30 | 19/30 | 5.5/30 | **30/30** |

**关键发现**: EXPO-FT 是唯一在所有任务上达到完美成功率的方法。HIL-SERL 虽从零 RL 训练效果良好但在 VLA 场景失败，SFT/HG-DAgger 受限于示范分布上限。

### Table 2: 样本效率（全部 8 个任务）

| 方法 | 平均在线数据时长 | 相对 SFT 提升 |
|------|----------------|---------------|
| SFT | 0（离线） | — |
| HG-DAgger | 较多在线交互 | +9% |
| **EXPO-FT** | **19.1 分钟** | **+44%** |

**关键发现**: EXPO-FT 平均仅需 19.1 分钟在线机器人数据（约等于 100-200 次 episode），比其他 RL 方法更高效。

---

## 实验

### 评估任务

| 任务 | 复杂度类型 | 关键挑战 |
|------|-----------|---------|
| Egg Flip | 接触丰富 | 空间感接触 + 动态铲翻控制 |
| String Light Routing (3 variants) | 精密对齐 | 多步线材穿绕，毫米级精度 |
| Candy Scoop | 视觉复杂 | 杂乱环境中稳定舀取 |
| Cube Pick | 大范围随机化 | 宽分布抓取泛化 |
| Flower Insert | 精密配合 | 花茎插入窄口，公差极小 |
| Pool Shot | 力控 | 精准击球速度和方向 |

### 实现细节

- **Backbone (Actor)**: 预训练 VLA 视觉编码器（完整微调）
- **Backbone (Critic)**: ResNet-50（轻量级，减少计算负担）
- **优化器**: Adam，学习率 $3 \times 10^{-4}$
- **Batch Size**: 64
- **折扣因子**: $\gamma = 0.99$
- **更新数据比**: 20（每个环境步做 20 次梯度更新）
- **Q 网络集成**: REDQ 风格 10 个 Q 网络
- **末端执行器控制**: 笛卡尔速度 + 夹爪指令，10 Hz
- **编辑尺度**: 0.05~0.2（精密任务用小值）
- **重规划间隔**: 4~8 步（任务相关）
- **训练步数**: 8,000~20,000 步（任务相关）

### 奖励设计

稀疏二值奖励，任务完成得 1，否则为 0：
- 基于规则的分类器（准确率 >95%）
- 基于像素的物体位置检测
- 高度阈值和空间约束综合判断

---

## 批判性思考

### 优点

1. **零妥协的任务成功率**: 所有 8 个任务均达到 30/30，是首个在如此多样化复杂任务上实现完美性能的 VLA RL 微调方法
2. **极致样本效率**: 平均 19.1 分钟在线数据，对工业部署极具吸引力
3. **保留预训练先验**: 直接微调完整 VLA，不冻结参数，充分利用大模型的泛化能力
4. **人类干预自然融合**: 干预率随训练自动衰减，人机协作有机结合

### 局限性

1. **需要人工环境复位**: 每个 episode 结束后需要人工重置环境，限制了大规模自动化训练的可行性
2. **计算资源密集**: 数十亿参数 VLA 的微调对硬件要求极高，限制了高频训练和推理
3. **无消融实验**: 论文未提供关键组件（如动作块 Q 函数、REDQ 集成、在线策略选择）的独立贡献分析
4. **任务复杂性局限**: 目前仅评估单步或短序列操作任务，长程推理任务的适用性未知

### 潜在改进方向

1. **自动环境复位**: 结合机器人自动复位策略或多机器人轮换训练
2. **参数高效微调**: 结合 LoRA 等方法降低微调计算成本
3. **多任务 RL 微调**: 探索单次训练同时优化多个任务的能力

### 可复现性评估

- [x] 代码开源
- [ ] 预训练模型（未明确说明）
- [x] 训练细节完整（附录 C 提供完整超参数）
- [x] 任务规格完整（附录中详细说明）

---

## 关联笔记

### 基于

- [[EXPO]]: 核心 RL 微调算法，EXPO-FT 将其扩展至动作块和 VLA 设置
- [[SERL]]: 样本高效 RL 框架先驱，HIL-SERL 的基础

### 对比

- [[HIL-SERL]]: 同为人类在环 RL，但从头训练无法利用 VLA 先验
- [[HG-DAgger]]: 人机协作但仅靠模仿学习，无奖励优化
- [[DSRL]]: 扩散策略 RL，在潜空间引导但受限先验分布

### 方法相关

- [[VLA]]: 被微调的预训练视觉-语言-动作模型
- [[Action Chunking]]: EXPO-FT 处理的核心动作表示形式
- [[强化学习]]: 核心优化框架
- [[SAC]]: 熵正则化 RL 算法基础，影响编辑策略损失设计
- [[REDQ]]: Q 网络集成策略，降低价值估计偏差
- [[模仿学习]]: SFT 和 HG-DAgger 基线使用的方法
- [[Actor-Critic]]: EXPO-FT 编辑策略 + Q 函数的整体框架
- [[经验回放]]: 人类干预数据和 RL 采样数据的存储与复用

### 硬件/数据相关

- [[ResNet-50]]: Critic 网络的轻量级视觉骨干

---

## 速查卡片

> [!summary] EXPO-FT (Stanford, 2026)
> - **核心**: 将 EXPO 扩展至 VLA RL 微调，支持动作块和人类干预
> - **方法**: 编辑策略 + 动作块 Q 函数 + 在线策略选择 + 人类在环干预
> - **结果**: 8 任务全部 30/30，平均仅需 19.1 分钟在线数据
> - **代码**: [pd-perry.github.io/expo-ft](https://pd-perry.github.io/expo-ft/)

---

*笔记创建时间: 2026-05-27*
