---
title: "Aligning Video World Models with Executable Robot Actions via Inverse Dynamics Rewards"
method_name: "EVA"
authors: [Ruixiang Wang, Qingming Liu, Yueci Deng, Guiliang Liu, Zhen Liu, Kui Jia]
year: 2026
venue: arXiv
tags: [video-world-model, rl-alignment, inverse-dynamics-model, robot-manipulation, flow-matching, bimanual-robot]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2603.17808v3/
created: 2026-10-21
---

# 论文笔记：Aligning Video World Models with Executable Robot Actions via Inverse Dynamics Rewards

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | The Chinese University of Hong Kong, Shenzhen; DexForce Technology Co., Ltd. |
| 日期 | March 2026 |
| 项目主页 | [eva-project-page.github.io](http://eva-project-page.github.io/) |
| 对比基线 | [[PhiZero\|π0]], [[Vidar]] |
| 链接 | [arXiv](https://arxiv.org/abs/2603.17808) |

---

## 一句话总结

> EVA 通过基于[[逆动力学模型]]（IDM）的可执行性奖励对视频世界模型进行 RL 后训练，消除生成视频帧在运动学上不可执行的"可执行性差距"，显著提升机器人任务成功率。

---

## 核心贡献

1. **可执行性差距定义**: 首次系统刻画视频世界模型生成帧在视觉连贯但运动学不可执行之间的矛盾，命名为"executability gap"。
2. **IDM 可执行性奖励**: 以真实机器人轨迹训练的 [[逆动力学模型]] 为奖励源，度量关节速度/加速度/加加速度平滑性及关节极限违反，引导 [[GRPO]] 后训练。
3. **端到端验证**: 在 RoboTwin 2.0（21 任务）和真实双臂机器人（5 seen + 5 OOD 任务）上验证，EVA 超越所有对比方法。

---

## 问题背景

### 要解决的问题

视频生成模型作为机器人[[世界模型]]时，生成帧虽视觉真实，但经[[逆动力学模型]]解码后的关节轨迹常出现：关节角突变（抖动）、手臂形变（artifact）、关节超限等问题，导致机器人无法实际执行。

### 现有方法的局限

- 纯视频生成模型（如 Wan2.1/Vidar）优化图像质量，与机器人物理约束完全解耦
- [[Diffusion Policy]] / [[ACT]] / [[RDT]] 等端到端方法不生成中间视频帧，可解释性差
- [[RLHF]]/[[GRPO]] 虽已用于语言/图像对齐，但如何定义机器人动作可执行性奖励尚无先例

### 本文的动机

"可执行性"可由 IDM 定量衡量：若生成视频帧对应的关节轨迹满足运动学约束（平滑、不超限），则视频即为"可执行的世界模型预测"。这为 RL 对齐提供了自动可微的标量奖励。

---

## 方法详解

### 模型架构

EVA 采用 **解耦视频规划器 + IDM** 架构：

- **输入**: 语言指令 $c$ + 初始帧 $x_0$
- **Backbone**: [[Wan2.1]]（14B 参数的 [[扩散 Transformer]]，[[Flow Matching]] 框架）
- **核心模块**: [[GRPO]] 驱动的 RL 后训练 + [[逆动力学模型]] 奖励
- **输出**: 生成视频帧序列 $V = \{I_1, \ldots, I_T\}$，经 IDM 解码为关节动作 $A = \{a_1, \ldots, a_T\}$
- **总参数**: 14B（视频模型）+ 轻量级 IDM

### 核心模块

#### 模块1: 逆动力学模型（IDM）

**设计动机**: 利用真实机器人轨迹监督学习"图像对 → 关节动作"的映射，作为免标注的可执行性评估器。

**具体实现**:
- [[卷积神经网络]] 骨干提取帧特征
- [[Spatial Softmax]] 将特征图转为关键点坐标（端到端可微）
- [[MLP]] 头预测关节动作 $a_t \in \mathbb{R}^{12}$（双臂 6+6 DoF）
- 训练方式：有监督回归，输入 $I_{t-k:t+k}$，监督信号为真实关节角

#### 模块2: 可执行性奖励

**设计动机**: 用 IDM 解码得到的关节轨迹，计算运动学平滑性惩罚，替代人工标注。

**具体实现**:
- 有限差分计算关节速度 $\dot{q}$、加速度 $\ddot{q}$、加加速度 $\dddot{q}$
- [[Huber 损失]] 惩罚过大的速度/加速度偏差（平滑性惩罚 $\mathcal{P}_j$、$\mathcal{P}_\alpha$）
- Hinge 惩罚超出速度/加速度极限（物理约束惩罚 $\mathcal{P}_{vel}$、$\mathcal{P}_{acc}$）
- 最终奖励将总惩罚映射为 $[0,1]$ 区间内的标量

#### 模块3: GRPO 后训练

**设计动机**: [[GRPO]]（Group Relative Policy Optimization）相比 [[PPO]] 无需价值网络，适合大规模视频生成模型的在线 RL 训练。

**具体实现**:
- 每个 prompt 生成 $G=8$ 个独立视频 rollout
- 对每个 rollout 计算可执行性奖励 $R_i$，组内归一化得优势 $\hat{A}_i$
- 梯度通过 [[SDE]] 采样路径的 ODE 求解器回传（Flow-GRPO 扩展）
- [[LoRA]] rank 32 微调，8×A800 GPU 训练

---

## 关键公式

### 公式1: [[Flow Matching|Flow Matching 训练目标]]

$$
\mathcal{L}_{FM} = \mathbb{E}\left[\left\|(x_1 - x_0) - v_\theta(x_t, t, c)\right\|_2^2\right]
$$

其中 $x_t = (1 - t)x_0 + t \cdot x_1$

**含义**: 训练速度场网络 $v_\theta$ 预测从噪声 $x_0$ 到真实帧 $x_1$ 的 ODE 流。

**符号说明**:
- $x_0$: 标准高斯噪声
- $x_1$: 真实视频帧（VAE 潜空间）
- $x_t$: 时刻 $t$ 的插值状态
- $v_\theta$: 速度场网络（Wan2.1 DiT）
- $c$: 条件（语言指令 + 初始帧）

### 公式2: [[逆动力学模型|IDM 监督损失]]

$$
\mathcal{L}_{IDM} = \mathbb{E}\left[\sum_t \left\|f_\phi(I_{t-k:t+k}) - a_t^{gt}\right\|_2^2\right]
$$

**含义**: 用真实关节角 $a_t^{gt}$ 监督 IDM 网络 $f_\phi$ 从帧序列预测关节动作。

**符号说明**:
- $f_\phi$: IDM 网络（CNN + Spatial Softmax + MLP）
- $I_{t-k:t+k}$: 以时刻 $t$ 为中心的 $2k+1$ 帧窗口
- $a_t^{gt}$: 真实关节角（从机器人数据集获取）

### 公式3: [[Spatial Softmax|空间 Softmax 关键点提取]]

$$
p_{ij}^c = \frac{\exp(F_{ij}^c)}{\sum_{i'j'} \exp(F_{i'j'}^c)}, \quad (x_c, y_c) = \sum_{ij} p_{ij}^c (i, j)
$$

**含义**: 将 CNN 特征图 $F^c$ 转化为通道 $c$ 的"软"关键点坐标 $(x_c, y_c)$，保持梯度流动。

**符号说明**:
- $F_{ij}^c$: 通道 $c$ 在位置 $(i,j)$ 的特征响应
- $p_{ij}^c$: 位置 $(i,j)$ 在通道 $c$ 的注意力权重
- $(x_c, y_c)$: 第 $c$ 个关键点的期望坐标

### 公式4: [[Huber 损失|平滑性惩罚（Huber）]]

$$
\text{Huber}(x; \delta) = \begin{cases} \frac{1}{2}x^2 & \text{if } |x| \leq \delta \\ \delta\left(|x| - \frac{1}{2}\delta\right) & \text{otherwise} \end{cases}
$$

$$
\mathcal{P}_j = \sum_t \text{Huber}(\dddot{q}_t; \delta_j), \quad \mathcal{P}_\alpha = \sum_t \text{Huber}(\ddot{q}_t; \delta_\alpha)
$$

**含义**: 用 Huber 函数惩罚关节加加速度 $\dddot{q}$（jerk）和加速度 $\ddot{q}$（acceleration）过大，鼓励平滑轨迹。

**符号说明**:
- $\dddot{q}_t$: $t$ 时刻关节角的三阶有限差分（jerk）
- $\ddot{q}_t$: $t$ 时刻关节加速度
- $\delta_j, \delta_\alpha$: 各自的 Huber 阈值

### 公式5: [[可执行性奖励|总惩罚与奖励映射]]

$$
\mathcal{P}(A) = \lambda_j \mathcal{P}_j + \lambda_\alpha \mathcal{P}_\alpha + \lambda_{v\text{-lim}} \mathcal{P}_{vel} + \lambda_{a\text{-lim}} \mathcal{P}_{acc}
$$

$$
R(V) = \left(1 + \frac{\mathcal{P}(A)}{P_0}\right)^{-\gamma}
$$

**含义**: 将四类惩罚加权求和后，通过幂函数映射为 $[0,1]$ 区间的奖励 $R(V)$，惩罚越大奖励越低。

**符号说明**:
- $\mathcal{P}_{vel}, \mathcal{P}_{acc}$: 超关节速度/加速度极限的 hinge 惩罚
- $P_0$: 惩罚归一化尺度（超参数）
- $\gamma$: 衰减率（超参数），控制惩罚的敏感度

### 公式6: [[GRPO|GRPO 优势估计]]

$$
\hat{A}_i = \frac{R_i - \mu_R}{\sigma_R + \varepsilon}
$$

**含义**: 对同一 prompt 的 $G$ 个 rollout 奖励做组内 z-score 归一化，得到相对优势信号。

**符号说明**:
- $R_i$: 第 $i$ 个 rollout 的可执行性奖励
- $\mu_R, \sigma_R$: 同组奖励的均值和标准差
- $\varepsilon$: 数值稳定小量

---

## 关键图表

### Figure 1: EVA 系统概览

![Figure 1 - EVA Overview](https://arxiv.org/html/2603.17808v3/main_fig.png)

**说明**: 上方展示"可执行性差距"（视觉合理但关节轨迹抖动），下方展示 EVA 通过 [[GRPO]] 后训练使生成视频满足运动学约束，并通过 [[逆动力学模型]] 解码为可执行动作。

### Figure 2: IDM 解码的关节轨迹对比

![Figure 2 - IDM Action Comparison](https://arxiv.org/html/2603.17808v3/IDM.png)

**说明**: 高奖励 rollout（奖励 7.94）的关节角轨迹平滑连续；低奖励 rollout（奖励 3.04）出现明显抖动和突变，对应视频中的视觉 artifact。

### Figure 3: 定性对比

![Figure 3 - Qualitative Comparison](https://arxiv.org/html/2603.17808v3/comparison.png)

**说明**: 未对齐的 Base 模型和 Vidar 出现手臂形变、穿插等不合理现象；EVA 生成帧保持运动学完整性，外观自然。

### Figure 4: 真实机器人部署

![Figure 4 - Real Robot Deployment](https://arxiv.org/html/2603.17808v3/exp_fig.png)

**说明**: EVA 生成的视频计划与 AgileX PiPER 双臂机器人实际执行对比，展示在"seen"和"OOD"任务中的视频规划质量与动作执行一致性。

### Figure S1: 失败案例分析

![Figure S1 - Failure Modes](https://arxiv.org/html/2603.17808v3/failure_mode.png)

**说明**: 三类主要失败模式：①运动学不合理（手臂形变）、②接触错误（抓错物体）、③目标错误（任务理解偏差）。

### Figure S2: LIBERO 任务对比

![Figure S2 - LIBERO Comparison](https://arxiv.org/html/2603.17808v3/libero_base_eva_comparison.png)

**说明**: 在 LIBERO 数据集上 Base（未对齐）与 EVA（RL 对齐后）的生成视频对比，验证 EVA 的泛化能力。

### Figure S3: 奖励有效性验证

![Figure S3 - Reward Validity](https://arxiv.org/html/2603.17808v3/reward_vs.png)

**说明**: 验证可执行性奖励的有效性：高奖励视频与视觉 artifact 出现率负相关，与仿真任务成功率正相关。

### Figure S4: 训练奖励曲线

![Figure S4 - Training Reward Curves](https://arxiv.org/html/2603.17808v3/reward_curve.png)

**说明**: 仿真和真实机器人设置下，GRPO 训练过程中奖励均稳步提升，表明 RL 后训练收敛稳定。

### Figure S5: 奖励 Hacking 案例

![Figure S5 - Reward Hacking](https://arxiv.org/html/2603.17808v3/hacking.png)

**说明**: 训练中发现的奖励 hacking 行为：错误交互（奖励高但任务错误）、连杆长度 artifact、机器人静止不动。说明奖励设计中接触力/扭矩信息的缺失仍是局限。

### Figure S6: 机器人平台

![Figure S6 - Robot Platform](https://arxiv.org/html/2603.17808v3/figures/robot.png)

**说明**: 实验使用的 AgileX PiPER 双臂平台，每臂 6 DoF，负载 1.5 kg，夹爪开合 0–70 mm。

### Figure S7: OOD 任务零样本生成

![Figure S7 - OOD Zero-shot Generation](https://arxiv.org/html/2603.17808v3/final_fig.png)

**说明**: EVA 在从未见过的 OOD 任务上，仅凭单张条件帧即可生成合理的视频规划，体现世界模型的泛化能力。

### Figure S8: 合成场景条件生成

![Figure S8 - Synthetic Scene Generation](https://arxiv.org/html/2603.17808v3/Nano.png)

**说明**: EVA 以合成场景图像为条件生成视频规划，探索零样本数据扩展的可能性。

---

### Table 1: 人工评估视觉规划质量（210 个 prompt）

| 方法 | Kinematic ↑ | Interaction ↑ | Instruction ↑ | Perfect ↑ |
|------|------------|--------------|--------------|----------|
| Vidar (Wan2.2) | 67.6% | 66.7% | 87.6% | 62.9% |
| EVA (w/o RL) | 70.5% | 83.3% | 90.5% | 68.1% |
| **EVA (with RL)** | **91.4%** | **86.2%** | **89.5%** | **83.8%** |

**说明**: RL 后训练对"Kinematic"（运动学合理性）提升最显著（+20.9 pp），证明可执行性奖励针对性强；"Perfect"（全部通过）提升 +15.7 pp。

### Table 2: RoboTwin 2.0 仿真成功率（21 任务平均）

| 方法 | 平均成功率 |
|------|-----------|
| ACT | 29.0% |
| Diffusion Policy | 29.5% |
| RDT | 37.1% |
| π0 | 45.7% |
| EVA (w/o RL) | 46.2% |
| **EVA (with RL)** | **52.6%** |

**说明**: EVA 超越 [[PhiZero|π0]] 6.9 pp，说明视频世界模型 + IDM 解码路径在标准 benchmark 上已具竞争力。

### Table 3: 真实机器人成功率（每任务 20 次）

| 方法 | Seen 平均 | OOD 平均 |
|------|----------|---------|
| ACT | 42.0% | — |
| π0 | 51.0% | 11.0% |
| Vidar | 44.0% | 34.0% |
| GE-Act | 43.0% | 3.0% |
| EVA (w/o RL) | 52.0% | 42.0% |
| **EVA (with RL)** | **64.0%** | **60.0%** |

**说明**: EVA 在 OOD 任务上优势最为突出（+26 pp vs π0），体现视频世界模型泛化能力在分布外场景中的优势。

### Table 4: IDM 在 RoboTwin 上的上限实验

| 设置 | 平均成功率 |
|------|-----------|
| IDM + Ground-truth Video | 89.52% |

**说明**: 使用真实视频作为输入时 IDM 可达 89.52% 成功率，说明视频生成质量（而非 IDM 能力）是当前系统的主要瓶颈。

### Table S1: IDM 消融实验

| 变体 | 预测精度 | 任务成功率 |
|------|---------|-----------|
| **Ours (Spatial Softmax)** | **0.9864** | **89.52%** |
| w/o Spatial Softmax | 0.7738 | 84.29% |

**说明**: [[Spatial Softmax]] 带来 +0.21 精度提升和 +5.23 pp 成功率提升，是 IDM 的关键设计。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| 真实机器人轨迹 | 未明确 | 双臂操作，含关节角标注 | IDM 训练 |
| RoboTwin 2.0 | 21 任务 | 仿真，含多样物体操作 | 仿真评估 |
| 真实机器人任务 | 10 任务（5 seen + 5 OOD） | 双臂 AgileX PiPER | 真实评估 |
| LIBERO | 标准 benchmark | 桌面操作 | OOD 泛化验证 |

### 实现细节

- **视频生成 Backbone**: Wan2.1-14B [[扩散 Transformer]]（[[Flow Matching]] 框架）
- **微调方式**: [[LoRA]] rank 32
- **Rollout 数**: $G=8$（每个 prompt 并行生成 8 个候选）
- **硬件**: 8 × A800 GPU
- **IDM 架构**: CNN + [[Spatial Softmax]] + MLP，输入 $2k+1$ 帧窗口
- **奖励超参**: $P_0$（惩罚归一化尺度）、$\gamma$（衰减率）、$\lambda_{j/\alpha/v/a}$（各项权重）

### 可视化结果

- RL 后训练后，生成视频中手臂形变、抖动等 artifact 明显减少
- OOD 任务中 EVA 生成的视频在外观和运动学上均比 baseline 更自然
- 仿真和真实场景的训练奖励曲线均单调上升，训练稳定

---

## 批判性思考

### 优点

1. **奖励设计简洁有效**: IDM 作为奖励源不需要额外人工标注，且与任务目标（可执行动作）直接对齐
2. **OOD 泛化突出**: 视频世界模型的先验知识使其在 OOD 场景下（60% vs 11% π0）远超端到端方法
3. **上限分析清晰**: IDM + GT video 达 89.52% 成功率，明确指出生成质量是当前瓶颈

### 局限性

1. **奖励不完整**: 忽略接触力、摩擦力、关节扭矩，导致奖励 hacking（机器人静止不动以避免惩罚）
2. **推理速度慢**: 扩散/Flow Matching 的迭代采样使实时控制受限
3. **无 force/torque 反馈**: 纯视觉 IDM 无法感知物体质量、接触状态等力学信息

### 潜在改进方向

1. 引入接触力/扭矩作为额外奖励信号，减少奖励 hacking
2. 结合一致性蒸馏（consistency distillation）加速视频推理
3. 扩展到移动操作（mobile manipulation）场景

### 可复现性评估

- [ ] 代码开源（项目页面未见代码链接）
- [ ] 预训练模型（未公开）
- [x] 训练细节完整（附录详述超参数和 robot spec）
- [ ] 数据集可获取（真实机器人数据未公开）

---

## 关联笔记

### 基于

- [[Wan2.1]]: 使用其 14B DiT 作为视频生成 backbone
- [[GRPO]]: RL 后训练算法，扩展至 Flow Matching（Flow-GRPO）
- [[Flow Matching]]: 视频生成的训练和推理框架

### 对比

- [[PhiZero|π0]]: 代表性端到端扩散策略，RoboTwin 上被 EVA 超越
- [[Vidar]]: 视频世界模型基线，无可执行性对齐
- [[ACT]]: 经典行为克隆方法，作为低线 baseline
- [[RDT]]: 扩散策略，对比方法之一

### 方法相关

- [[逆动力学模型]]: EVA 的核心奖励组件
- [[Spatial Softmax]]: IDM 中的关键点提取模块
- [[世界模型]]: EVA 的核心范式
- [[可执行性奖励]]: 本文提出的新概念

### 硬件/数据相关

- [[AgileX PiPER]]: 实验使用的双臂机器人平台（6 DoF × 2，1.5 kg 负载）
- [[RoboTwin 2.0]]: 仿真评估 benchmark（21 任务）

---

## 速查卡片

> [!summary] EVA (2026, arXiv:2603.17808)
> - **核心**: 用 IDM 可执行性奖励对视频世界模型做 GRPO 后训练，消除"可执行性差距"
> - **方法**: Flow Matching 视频生成 + IDM 奖励（运动学平滑 + 关节限制） + GRPO
> - **结果**: RoboTwin 52.6%（+6.9 pp vs π0），真实机器人 64%/60%（seen/OOD）
> - **代码**: [eva-project-page.github.io](http://eva-project-page.github.io/)

---

*笔记创建时间: 2026-10-21*
