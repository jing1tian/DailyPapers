---
title: "Fine-Tuning VLAs with Self-Demonstrated Generative Control for Multi-Task Manipulation"
method_name: "SDGC"
authors: [Prachi Garg, Steve Xing, Prahit Yaug, Saurabh Gupta, Derek Hoiem]
year: 2026
venue: arXiv
tags: [vla-fine-tuning, catastrophic-forgetting, self-supervised-learning, continual-learning, imitation-learning, multi-task-manipulation, generative-replay]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2608.19490v1
created: 2026-08-22
---

# 论文笔记：Fine-Tuning VLAs with Self-Demonstrated Generative Control

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | University of Illinois Urbana-Champaign (UIUC) |
| 日期 | August 2026 |
| 项目主页 | [self-supervised-control.pages.dev](https://self-supervised-control.pages.dev/) |
| 对比基线 | [[π0.5]] (base model) |
| 链接 | [arXiv](https://arxiv.org/abs/2608.19490) |

---

## 一句话总结

> 通过将冻结的预训练 [[π0.5]] 在目标机器人上做自监督 rollout，生成「自我示教」数据并联合训练，解决 [[VLA]] 迁移到新机器人时的[[灾难性遗忘]]问题，同时无需访问原始预训练数据。

---

## 核心贡献

1. **自监督数据生成（Self-Demonstration）**: 将冻结的基础 [[π0.5]] 策略在目标机器人上执行 rollout，收集观测-动作对作为自监督训练数据，消除了 embodiment gap——因为数据就是在目标硬件上生成的。
2. **联合目标函数（Joint Training Objective）**: 将专家监督损失 $\mathcal{L}_{es}$ 与自监督蒸馏损失 $\mathcal{L}_{ss}$ 以1:1比例联合优化，保留预训练先验同时学习新技能，避免[[灾难性遗忘]]。
3. **综合评估基准（Multi-Platform Benchmark）**: 提出覆盖 ALOHA 真实机器人（抓取、双臂）和 RoboTwin 仿真的三个 benchmark，包含四类测试分布（$T_{es}$、$T_{ss}$、$T_{no}$、$T_{nc}$）。

---

## 问题背景

### 要解决的问题

[[VLA]] 模型（如 [[π0.5]]）在大规模预训练后具备强大的零样本指令跟随能力，但部署到新机器人时存在**硬件配置不匹配（embodiment gap）**——微小的硬件差异导致抓取失败率极高（零样本成功率约0%）。直接在专家数据上微调可以修复抓取，但会导致**[[灾难性遗忘]]**：模型不再能跟随指令（如颜色区分），也丢失了预训练时学到的其他任务能力。

### 现有方法的局限

- **参数高效微调（LoRA 等）**：保留能力有限（$T_{ss}$ 仅28%），且新任务性能也不如全量微调（87%）。
- **Experience Replay**：公认最有效，但需要访问原始预训练数据集，现实中往往无法获得。
- **权重合并/Adapter路由**：需要额外的架构设计，且缺乏数据生成能力。

### 本文的动机

即使 [[π0.5]] 的零样本策略无法成功完成任务（由于 embodiment gap），其动作预测仍然与语言指令**语义相关**。例如，基础策略会尝试抓取正确颜色的方块，只是因为硬件差异而抓取失败。这一观察表明：可以将冻结的基础策略本身作为**轨迹生成器**，在目标硬件上执行 rollout，产出「自我示教」的训练数据来做[[生成式回放]]，无需访问原始预训练数据。

---

## 方法详解

### 模型架构

SDGC 的基础模型是 [[π0.5]]，这是一个连续[[Action Chunking|动作块]] VLA：
- **输入**: 语言指令 $p$ + 多模态观测 $\mathbf{o}_t = (\mathbf{I}_t, p, \mathbf{q}_t)$，其中 $\mathbf{I}_t$ 为相机图像，$\mathbf{q}_t$ 为本体感知状态
- **视觉骨干**: [[SigLIP]] 视觉编码器
- **语言编码**: [[PaliGemma]] 文本编码
- **Action Expert**: 专门的动作专家网络，预测 [[Flow Matching|流速度]] 向量
- **输出**: 连续[[Action Chunking|动作块]] $\mathbf{a}_t = a_{t:t+H}$，其中 $H=50$，执行前25步后重新规划

SDGC **不改变** [[π0.5]] 的架构，而是改变**训练数据构成**：将专家数据 $\mathcal{D}_{es}$ 与冻结基础策略生成的自监督数据 $\mathcal{D}_{ss}$ 混合。

### 核心模块

#### 模块1：噪声化动作表示

[[π0.5]] 使用 [[Flow Matching]] 的连续动作和 [[FAST]] 的离散动作 token 双轨输出。噪声化动作：

$$
\mathbf{a}_t^{\tau,\omega} = \tau \mathbf{a}_t + (1 - \tau)\omega
$$

其中 $\tau \in [0,1]$ 是 [[Flow Matching]] 时间索引，$\omega$ 是高斯噪声，$\mathbf{a}_t \in \mathbb{R}^{H \times D}$（$D=32$ 维度，关节绝对角度）。

#### 模块2：自监督 Rollout 数据收集

在目标机器人上执行冻结基础策略 $\pi_0^{base}$：

$$
\hat{\mathbf{a}}_t \sim \pi_0^{base}(\cdot | \mathbf{I}_t, p_{ss}, \mathbf{q}_t)
$$

收集所有时间步的 $(\mathbf{o}_t, \hat{\mathbf{a}}_t)$ 对，形成自监督数据集 $\mathcal{D}_{ss}$。任务提示 $p_{ss}$ 从预训练任务族中选取（如14条抓放指令）。手动过滤夹爪卡顿、相机遮挡等不良行为后使用。

### 图示概览

![Figure 3: 方法示意图](https://arxiv.org/html/2608.19490v1/Method_Diagram.png)

**说明**: SDGC 的两路数据来源。底行：专家遥操作演示（新任务）；顶行：冻结 [[π0.5]] 在目标机器人上的 rollout 数据（预训练任务族），两者联合训练形成多任务策略。

---

## 关键公式

### 公式1：[[FAST|FAST 离散 Token]] + [[Flow Matching|流匹配]] 联合损失

$$
\ell(\mathbf{o}_t, \mathbf{a}; \theta) = \mathbb{E}_{t,\omega}\left[L_{ce}\!\left(\mathbf{a}_{1:m}^{FAST},\, f_\theta^l(\mathbf{o}_t, \mathbf{a}_{<m}^{FAST})\right) + \alpha\left\|(\omega - \mathbf{a}) - f_\theta^a(\mathbf{a}^{\tau,\omega}, \mathbf{o}_t)\right\|^2\right]
$$

**含义**: 单样本损失，同时优化离散 [[FAST]] token 的交叉熵和连续 [[Flow Matching|流匹配]] 速度预测误差。

**符号说明**:
- $L_{ce}$: FAST token 的交叉熵损失
- $f_\theta^l$: VLM backbone 预测离散 token
- $f_\theta^a$: Action Expert 预测流速度向量
- $m$: FAST token 数量
- $\alpha$: 离散/连续目标的平衡权重（抓取任务 $\alpha=8.0$，齿轮任务 $\alpha=10.0$）

### 公式2：[[灾难性遗忘|专家监督目标]]

$$
\mathcal{L}_{es}(\theta) = \mathbb{E}_{(\mathbf{o}_t, \mathbf{a}_t) \sim \mathcal{D}_{es}}\left[\ell(\mathbf{o}_t, \mathbf{a}_t, \theta)\right]
$$

**含义**: 在专家示教数据集 $\mathcal{D}_{es}$（遥操作收集）上的期望损失，用于学习新任务。

**符号说明**:
- $\mathcal{D}_{es}$: 专家监督数据集，$N$ 条轨迹，任务提示 $p_{es} \in \mathcal{P}_{es}$

### 公式3：[[生成式回放|自监督蒸馏目标]]

$$
\mathcal{L}_{ss}(\theta) = \mathbb{E}_{(\mathbf{o}_t, \hat{\mathbf{a}}_t) \sim \mathcal{D}_{ss}}\left[\ell(\mathbf{o}_t, \hat{\mathbf{a}}_t, \theta)\right]
$$

**含义**: 在自生成 rollout 数据集 $\mathcal{D}_{ss}$ 上的期望蒸馏损失，用于保留预训练先验。

**符号说明**:
- $\mathcal{D}_{ss}$: 冻结 $\pi_0^{base}$ 在目标机器人上生成的数据集
- $\hat{\mathbf{a}}_t$: 基础策略的动作样本（非专家动作）

### 公式4：[[持续学习|联合训练目标]]

$$
\mathcal{L}(\theta) = \mathcal{L}_{es}(\theta) + \lambda \mathcal{L}_{ss}(\theta)
$$

**含义**: 核心训练目标，以权重 $\lambda$ 平衡新技能学习（$\mathcal{L}_{es}$）与先验保留（$\mathcal{L}_{ss}$）。

**符号说明**:
- $\lambda$: 自监督权重（最佳为1:1混合比，即 $\lambda=1$）

---

## 关键图表

### Figure 1: 方法对比演示

![Figure 1: Multi-task manipulation demo](https://arxiv.org/html/2608.19490v1/DEMO_Figure1_MT1_updated.png)

**说明**: 三列对比。左：零样本 [[π0.5]] 能跟随指令（如"pick up purple cube"）但因 embodiment gap 抓取失败。中：专家数据微调后丢失指令跟随（抓错颜色）并忘记「放置」动作。右（SDGC）：14分钟专家遥操作 + 自监督 rollout 联合训练，单策略成功完成所有任务族。

### Figure 2: 零样本指令跟随能力

![Figure 2: Text steerability](https://arxiv.org/html/2608.19490v1/text_steerability_figure.png)

**说明**: 展示 [[π0.5]] 零样本状态下在 ALOHA 平台上能语义定位并跟随指令到达绿色八边形和标记物——这是 SDGC 自监督数据生成的基础能力。

### Figure 3: 方法整体示意图

![Figure 3: Method diagram](https://arxiv.org/html/2608.19490v1/Method_Diagram.png)

**说明**: 两路数据构成。底行：专家遥操作演示（"Pick up the red cube"）；顶行：冻结 [[π0.5]] 在目标机器人上的 rollout（"Pick up the spoon and place it in the blue bowl"）。两者等比混合训练，无需访问原始预训练数据。

### Figure 4: RoboTwin 仿真 Benchmark

![Figure 4: RoboTwin benchmark overview](https://arxiv.org/html/2608.19490v1/task_overview_plain.png)

**说明**: RoboTwin 2.0 仿真 benchmark 设计。$T_{es}$ 为专家监督任务（叠方块），$T_{ss}$ 为自监督任务（抓放、举起、开合等10个任务），测试在不同对象布局下的泛化能力。

### Figure 5: 齿轮插入演示

![Figure 5: Caterpillar gear insertion demo](https://arxiv.org/html/2608.19490v1/DEMO_Figure3_MT2.png)

**说明**: ALOHA 双臂齿轮插入任务的成功 rollout，展示橙色齿轮在非固定桩板上的精准对准，验证 SDGC 在高接触力任务的有效性（成功率从30%提升至90%）。

### Figure 6: OOD 预训练任务对比 Rollout 1

![Figure 6: OOD montage 1](https://arxiv.org/html/2608.19490v1/montage_ood_pretrain_1.png)

**说明**: 对比三种策略在分布外任务（预训练任务族）的 rollout。零样本策略有正确的指令跟随和语义理解，但因硬件标定差异无法抓取对象。

### Figure 7: OOD 预训练任务对比 Rollout 2

![Figure 7: OOD montage 2](https://arxiv.org/html/2608.19490v1/montage2_ood_pretrain_2.png)

**说明**: SDGC 能更好地定位并抓取目标（叉子），展示了自监督训练后对 $T_{ss}$ 任务的改善；同时注意到放置绿色方块后策略又去拿紫色方块的 loop 行为（限制之一）。

### Figure 8: 洗衣任务 Rollout

![Figure 8: Laundry task rollouts](https://arxiv.org/html/2608.19490v1/laundry_rollouts.png)

**说明**: 最后一列为测试 episode 末帧。因为基础策略的 rollout 本身很嘈杂，自监督数据质量较低，策略有时会将衣物放到篮子边缘导致掉落（部分成功率40%）。Multi-Task ES 基线则因过拟合到抓取动作而直接停下。

### Figure 9: 齿轮 Benchmark $T_{es}$ Rollout

![Figure 9: Caterpillar montage](https://arxiv.org/html/2608.19490v1/montage_caterpillar.png)

**说明**: ALOHA Caterpillar Benchmark 的 $T_{es}$ 测试集 rollout，展示紫色/蓝色颜色区分、稳定抓握和正确轴对准插入能力。

### Figure 10: ALOHA 机器人平台

![Figure 10: ALOHA system](https://arxiv.org/html/2608.19490v1/Figures/aloha_system.JPG)

**说明**: 实验使用的固定式 [[ALOHA]]-1 平台——两条 ViperX 300 从手臂、两条 WidowX 250 主手臂、并联夹爪、Intel RealSense D435 相机（640×480，30-50 Hz 控制频率）。

---

### Table 1: 三个 Benchmark 的训练数据与测试集

| Benchmark | 专家数据 | 自监督数据 | $T_{es}$ | $T_{ss}$ | $T_{no}$ | $T_{nc}$ |
|-----------|---------|-----------|--------|--------|--------|--------|
| ALOHA 抓取 | 30 demos | 14 episodes | ✓ | ✓ | 新颜色对象 | 放置组合 |
| ALOHA 齿轮 | 60 demos | 30 episodes | ✓ | — | 新颜色齿轮 | — |
| RoboTwin 叠块 | 100 episodes | 100 episodes | 叠方块 | 10个旧任务 | 圆柱/球 | 三块叠/碗 |

**说明**: 四类测试分布设计，系统评估任务保留、指令跟随和泛化能力。

### Table 2: ALOHA 抓取 Benchmark 结果

| 方法 | 红绿 SR↑ | 新颜色 SR↑ | 抓放 SR↑ | 洗衣 Partial↑ | 指令跟随 IF↑ |
|------|---------|-----------|--------|------------|-----------|
| 零样本 π₀.₅ | 0% | 0% | 0% | 0% | 80% |
| 多任务 ES (专家only) | 90% | 50% | **0%** | 10% | 50%* |
| **SDGC (ES+SS)** | **90%** | **55%** | **55%** | **40%** | **100%** |

**关键发现**: 仅用专家「抓取」数据微调完全遗忘了「放置」行为（0% SR）。SDGC 用自监督放置数据恢复到55%，同时无需任何专家放置演示。*IF 指 pre-grasp 指令跟随。

### Table 3: ALOHA 齿轮插入结果

| 方法 | Pre-grasp IF↑ | Post-grasp IF↑ | SR↑ | 新颜色 SR↑ |
|------|-------------|--------------|-----|---------|
| 零样本 π₀.₅ | 0% | 0% | 0% | — |
| 多任务 ES (专家only) | **100%** | — | 30% | — |
| **SDGC (ES+SS)** | **100%** | **90%** | **90%** | **30%** |

**关键发现**: 接触力丰富的插轴任务从 30% 提升到 90%，证明自监督训练在预训练分布外的高精度任务同样有效。

### Table 4: RoboTwin 仿真结果（Stage 2 后训练）

| 方法 | $T_{ss}$（10个旧任务）↑ | $T_{es}$（新叠块）↑ |
|------|---------------------|-----------------|
| 多任务 ES (专家only) | 16.6% | 93.0% |
| **SDGC (ES+SS)** | **70.6%** | **98.0%** |
| Oracle（Experience Replay） | 85.6% | 97.0% |

**关键发现**: 专家only微调导致旧任务性能下降74.2%（90.8%→16.6%）。SDGC 在不访问原始数据的情况下恢复了54%的遗忘性能，达到 Oracle 的82.5%。

### Table 5: 参数高效微调消融（RoboTwin $T_{ss}$）

| 方法 | $T_{ss}$↑ | $T_{es}$↑ |
|------|---------|---------|
| [[LoRA]] | 28.0% | 87.0% |
| 冻结 SigLIP，调 VLM+AE | 23.8% | 93.0% |
| 仅 Flow 损失 | 11.2% | 94.0% |
| 全量微调（FAST+Flow） | 16.6% | 93.0% |
| **全量 + 自监督（SDGC）** | **70.6%** | **98.0%** |

**关键发现**: 全量微调配合双 FAST+Flow 目标效果最佳；FAST 损失相比 Flow-only 显著缓解遗忘。[[Self-Supervised Learning|自监督]]数据是最关键的提升来源，而非架构选择。

### Table 6: 与全专家数据对比（ALOHA）

| 方法 | 专家任务 SR↑ | 新颜色 SR↑ | 抓放 SR↑ | 洗衣 Partial↑ |
|------|-----------|---------|---------|------------|
| 全专家（ES+ES） | 95% | 60% | 65% | 90% |
| **SDGC（ES+SS）** | 90% | 55% | 55% | 40% |

**分析**: SDGC 接近全专家性能，最大差距在洗衣任务（40% vs 90%），因为该任务的自监督 rollout 噪声较大（物品时常落到篮子边缘）。

### Table 7: 保留 Hold-out 技能泛化（推送任务）

| 方法 | 正确对象 IF↑ | 推送动作↑ | 误抓率↓ | SR↑ |
|------|-----------|---------|-------|-----|
| 零样本 π₀.₅ | 45% | 35% | 0% | 30% |
| 全专家（ES+ES） | 85% | **10%** | 35% | 5% |
| **SDGC（ES+SS）** | **85%** | **70%** | **10%** | **60%** |

**关键洞察**: 全专家微调覆盖了基础先验，策略默认执行「抓-放」链而非推送（5% SR）。SDGC 保留了与专家任务无关的推送先验，SR 达60%，甚至超越零样本的30%。

### Table 8: ALOHA 自监督任务提示列表（14条）

| # | 提示 |
|---|------|
| 1 | "Pick up the laundry and place it in the basket" |
| 2 | "Pick up the spoon and place it in the red bowl" |
| 3 | "Pick up the marker and place it in the blue bowl" |
| 4-14 | 其余10条抓放变体（不同物品+容器组合） |

**说明**: 从预训练任务族中手工选取，覆盖不同物品和放置目标，用于生成自监督数据。

### Table 9: 自监督数据混合消融（RoboTwin）

| 配置 | $T_{ss}$↑ |
|------|---------|
| 自然比例（~1:2） | 56.4% |
| **1:1 均匀混合** | **69.2%** |
| 任务平衡采样 | 66.2% |
| 5 episodes/task | 56.8% |
| **10 episodes/task** | **69.2%** |

**关键发现**: 1:1 等比混合 + 每任务10个 episode + 均匀采样为最优配置。

---

## 实验

### 数据集与平台

| 平台 | 任务 | 专家数据 | 自监督数据 | 特点 |
|------|------|---------|-----------|------|
| ALOHA-1 | 抓取/放置 | 30 demos | 14 episodes | 真实机器人，RGB 640×480 |
| ALOHA-1 | 齿轮插入 | 60 demos | 30 episodes | 双臂协调，高接触力 |
| RoboTwin 2.0 | 叠块 (Stage 2) | 100 episodes | 100 episodes | 仿真，ALOHA-AgileX |

### 实现细节

**真实机器人（ALOHA）**:
- 相机: Intel RealSense D435，RGB 640×480，30-50 Hz 控制
- 训练迭代: 10,000（抓取），16,000-18,000（齿轮）
- Batch Size: 64
- 硬件: 1× NVIDIA H200 (140GB) 或 2× H100 (96GB)，部署用 RTX 4090
- Action Chunk: $H=50$，执行前25步

**仿真（RoboTwin 2.0）**:
- Stage 1: 10任务 × 500 episodes (130,815 帧)，训练至90.8%平均性能
- Stage 2: 6,000 次迭代（稳定性-可塑性最优权衡）
- Batch Size: 64，学习率 $2.5 \times 10^{-5}$，FAST 损失用逆频率任务权重

### 可视化结果

在 ALOHA 推送任务实验中最为显著：全专家微调策略看到物品后固执地执行「抓-放」动作链；SDGC 策略正确理解推送指令并完成60%的推送动作，展示了[[Self-Supervised Learning|自监督]]数据对保留预训练行为先验的关键作用。

---

## 批判性思考

### 优点

1. **无需原始预训练数据**: 仅用冻结基础模型在目标机器人上的 rollout 即可防遗忘，实用性强。
2. **无 embodiment gap**: 自监督数据就是在目标机器人、目标场景、目标提示下生成的，不存在跨域问题。
3. **强大的 held-out 技能保留**: 推送任务（两路训练数据均无 push 演示）SR 达60%，证明预训练先验被有效保留。

### 局限性

1. **手动选择 $\mathcal{P}_{ss}$**: 需要人工指定自监督任务提示集，缺乏自动化。
2. **需要手动数据过滤**: 夹爪卡顿、相机遮挡等不良行为需人工筛除。
3. **噪声数据传播**: 基础策略本身不完美，噪声自监督数据会影响如洗衣等高难度任务性能（40% vs 90%全专家）。
4. **无终止判断**: 因为自监督和专家数据均不包含任务完成后的场景，策略不知何时停止，导致完成放置后重复抓取的 loop 行为。

### 潜在改进方向

1. **自动 prompt 选择和数据过滤**，减少人工干预。
2. **与强化学习结合**，自我改进而非仅自我蒸馏。
3. **数据飞轮**：多轮迭代持续产生高质量自监督数据。

### 可复现性评估

- [ ] 代码开源（参考 openpi 代码库，自定义部分待发布）
- [ ] 预训练模型（依赖 π0.5，需要 Physical Intelligence 模型）
- [x] 训练细节完整（batch size、learning rate、超参数等均已列出）
- [ ] 数据集可获取（真实机器人数据未公开）

---

## 关联笔记

### 基于

- [[π0.5]]: 基础预训练 VLA 模型，SDGC 的冻结生成器和初始化权重
- [[FAST]]: 离散动作 token 编码方式，SDGC 联合训练的关键——仅 flow 损失时遗忘更严重
- [[Action Chunking]]: π0.5 的动作输出形式，SDGC 沿用 $H=50$ chunk 设计

### 对比

- [[π0.5]]: 零样本部署，指令跟随好但抓取失败；SDGC 在保留其先验的同时修复抓取
- EXPO-FT（见同目录）: 另一个 VLA 参数高效微调方法，但 LoRA 在 $T_{ss}$ 保留上仅达28%

### 方法相关

- [[灾难性遗忘]]: SDGC 要解决的核心问题
- [[生成式回放]]: SDGC 的核心机制——用生成模型自身产生回放数据
- [[持续学习]]: SDGC 所属的技术类别
- [[Self-Supervised Learning]]: 自监督目标 $\mathcal{L}_{ss}$ 的学习范式
- [[Flow Matching]]: π0.5 的连续动作预测机制

### 硬件/数据相关

- [[ALOHA]]: 实验平台，双臂遥操作系统
- [[SigLIP]]: π0.5 的视觉骨干编码器
- [[PaliGemma]]: π0.5 的语言编码器

---

## 速查卡片

> [!summary] SDGC (Self-Demonstrated Generative Control)
> - **核心**: 用冻结 [[π0.5]] 在目标机器人上 rollout 生成自监督数据，联合专家数据训练
> - **方法**: $\mathcal{L} = \mathcal{L}_{es} + \lambda \mathcal{L}_{ss}$，1:1 数据混合，全量微调
> - **结果**: 避免[[灾难性遗忘]]，恢复 Oracle Replay 82.5% 性能；齿轮插入 30%→90%；推送保留技能 60%
> - **代码**: [Project Page](https://self-supervised-control.pages.dev/)

---

*笔记创建时间: 2026-08-22*
