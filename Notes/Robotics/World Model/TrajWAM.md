---
title: "TrajWAM: An Efficient Trajectory-Guided World Action Model for Dynamic Robotic Manipulation"
method_name: "TrajWAM"
authors: [Haohan Geng, Sha Zhang, Weishuo Sun, Guanting Ye, Wenhao Yu, Yuang Xie, Qicheng Zeng, Peng Ye, Jiajun Deng, Jianmin Ji]
year: 2026
venue: arXiv
tags: [world-action-model, dynamic-manipulation, trajectory-prediction, diffusion-distillation, video-generation, robot-learning]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2610.25763v1/
created: 2026-10-21
---

# 论文笔记：TrajWAM: An Efficient Trajectory-Guided World Action Model for Dynamic Robotic Manipulation

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | USTC、CUHK、University of Macau、Huawei |
| 日期 | October 2026 |
| 项目主页 | [kevingeng3.github.io/TrajWAM](https://kevingeng3.github.io/TrajWAM) |
| 对比基线 | [[Fast-WAM]]、[[Flash-WAM]]、[[PUMA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2610.25763) / Code N/A |

---

## 一句话总结

> TrajWAM 通过预测目标物体的未来遮罩轨迹来引导视频生成和动作预测，并结合解耦蒸馏实现单步推理，在动态机械臂操作任务上达到 SOTA。

---

## 核心贡献

1. **轨迹引导的 WAM 框架**: 提出以物体为中心的未来遮罩预测器，将预测轨迹通过轻量级 [[Trajectory ControlNet]] 注入预训练视频模型，使 WAM 的视频想象显式受目标运动约束。
2. **解耦单步蒸馏策略**: 分三阶段独立蒸馏——视频骨干（[[Distribution Matching Distillation|DMD]]）、ControlNet 适配器对齐（对抗 + 重建损失）、动作展开蒸馏——将多步推理压缩到单步（108.5 ms）。
3. **动态操作 SOTA**: 在 DOMINO 基准（35 个动态任务）上以 43.54% 成功率超越最强 baseline 14.24 个百分点，真实机器人两项任务综合成功率达 74.2%。

---

## 问题背景

### 要解决的问题

动态操作（Dynamic Manipulation）场景中，目标物体持续运动（如传送带、旋转圆盘上的物体），机器人必须预判目标未来位置并以足够低的延迟执行，否则动作到执行时目标已移位，导致抓取失败。

### 现有方法的局限

- 现有 [[World Action Model|WAM]] 预测全帧场景演化：机器人自身运动常主导视频内容，目标物体的运动仅被间接、隐式地建模，在快速动态场景下预测精度不足。
- 多步迭代去噪（通常 10–50 步）引入显著延迟，已有方案如 [[Flash-WAM]] 通过一致性蒸馏减少步数，但缺乏显式的空间运动条件，对快速目标的适应性有限。
- 部分方法（PUMA、DynamicVLA）直接跳过视频想象，丧失了世界模型对未来推理的优势。

### 本文的动机

以物体为中心的轨迹（遮罩序列）比全帧视频更紧凑，易于预测，且能明确表达目标的未来运动。将该轨迹作为条件注入视频生成，再由动作专家从受控视频中读取动作，可以同时实现"精确预判目标"和"快速推理"两个目标。

---

## 方法详解

### 模型架构

TrajWAM 采用三阶段 **世界-动作联合生成** 架构：

- **输入**: 语言指令 $l$ + 多视角观测（固定第三方视角 + 手腕视角 RGB）+ 历史遮罩序列 $M^-$
- **阶段1 — 轨迹预测器**: 轻量级 2D [[U-Net]] $g_\phi$，预测 8 帧未来遮罩 $M^+$
- **阶段2 — 视频-动作生成器**: 预训练 5B 视频模型（冻结）+ [[Trajectory ControlNet]]（轻量化，隐藏维度 512 vs 原 3072）+ [[Action Expert]]（[[Asymmetric Attention|非对称注意力]]）
- **阶段3 — 单步推理（蒸馏后）**: [[Distribution Matching Distillation|DMD]] 压缩视频骨干 → 适配器对齐 → [[Action Rollout Distillation|动作展开蒸馏]]
- **总参数**: 5B 视频骨干（冻结），ControlNet/动作专家规模未公开

### 核心模块

#### 模块1: 以物体为中心的未来轨迹预测器

**设计动机**: 利用紧凑的遮罩序列代替全帧预测来捕获目标运动，计算成本低且物体位置信息更明确。

**具体实现**:
- 输入：来自固定第三视角摄像头的历史 4 帧遮罩（步长 4），即时刻 $t-12, t-8, t-4, t$ 的遮罩，resize 为 120×160
- 使用语言条件分割器（推理时用 [[SAM|SAM3]]）从第三方视角提取目标遮罩
- 2D U-Net $g_\phi$ 预测未来 8 帧遮罩，对应时刻 $t+4$ 至 $t+32$
- 输出经 Sigmoid 激活：$M^+ = \sigma(g_\phi(M^-))$
- 训练数据：1,575 条视频（验证 175 条），随机种子 42 划分

#### 模块2: Trajectory ControlNet

**设计动机**: 以轻量化旁路网络方式向冻结视频骨干注入轨迹条件，避免全量微调 5B 大模型。

**具体实现**:
- 将未来遮罩 $M^+$ 编码为轨迹潜码（通过视频 [[VAE]] 编码器）
- ControlNet 保持原视频模型深度，将隐藏维度从 3,072 压缩至 512
- 逐 Transformer block 产生残差 $\Delta h$，加入到视频骨干对应层的输出
- 参数随机初始化，零初始化投影层确保训练起始时不破坏视频模型
- 训练时冻结：视频骨干、[[VAE]]、文本编码器；可训练：ControlNet、零初始化投影、动作专家

#### 模块3: 非对称视频-动作注意力

**设计动机**: 让动作 token 读取受轨迹控制的视频表示，同时避免动作信号干扰视频生成质量。

**具体实现**:
- 视频 token 只与视频 token 做自注意力，不感知动作 token
- 动作 token 对视频 token 和动作 token 同时做交叉注意力
- 视频和动作共用 [[Flow Matching]] 目标函数，联合训练

#### 模块4: 解耦蒸馏（三步流程）

**步骤1 — 视频骨干蒸馏（DMD）**:
- 使用 [[Distribution Matching Distillation|分布匹配蒸馏]]，设 "真实得分模型"（冻结）和 "伪造得分模型"（可训练）
- 优化单步学生生成器，使其输出的分布匹配真实视频分布
- 训练：5,000 步，全局 batch 8，EMA 衰减 0.99

**步骤2 — 轨迹适配器对齐**:
- 将慢速 ControlNet 复制为快速适配器，在冻结的快速视频生成器上微调
- 联合损失：潜码重建 + 扩散判别器对抗损失（判别器读取 Transformer block 14、22、30 的特征）
- 训练：1,000 步

**步骤3 — 动作展开蒸馏**:
- 单步学生动作分支：匹配 10+10 步教师的动作块输出
- 使用相同观测、轨迹条件、初始噪声
- 训练：2,000 步，batch 4，学习率 1e-6

---

## 关键公式

### 公式1: [[Soft Dice Loss|遮罩预测器训练损失]]

$$
\mathcal{L}_{mask} = 0.5 \cdot \mathcal{L}_{BCE}(M^+, \hat{M}^+) + 0.5 \cdot \mathcal{L}_{Dice}(M^+, \hat{M}^+)
$$

**含义**: 以等权重组合 [[Binary Cross-Entropy|二元交叉熵]] 与软 Dice 损失，同时优化像素级精度和区域重叠度。

**符号说明**:
- $M^+$：预测的未来遮罩
- $\hat{M}^+$：真实的未来遮罩（来自分割器）
- $\mathcal{L}_{BCE}$：二元交叉熵损失
- $\mathcal{L}_{Dice}$：[[Soft Dice Loss]]，$1 - \frac{2 \sum M^+ \hat{M}^+}{\sum M^+ + \sum \hat{M}^+}$

### 公式2: [[Flow Matching|联合流匹配目标]]

$$
\mathcal{L}_{FM} = \mathbb{E}_{\tau, z_0, c}\bigl[\bigl\|v_\theta(z_\tau, \tau, c) - u_\tau(z_0 \mid z_1)\bigr\|^2\bigr]
$$

**含义**: 在视频和动作的联合潜码空间中，学习从噪声到数据的速度场，是 TrajWAM 的主训练目标。

**符号说明**:
- $\tau \sim \mathcal{U}(0, 1)$：流匹配时间步
- $z_\tau = (1-\tau) z_0 + \tau z_1$：在噪声 $z_1$ 与数据 $z_0$ 之间插值的中间状态
- $v_\theta$：模型预测的速度场（视频 + 动作联合）
- $u_\tau$：条件真实速度场
- $c$：条件信息（语言指令、历史观测、轨迹遮罩）

### 公式3: [[Trajectory ControlNet|轨迹 ControlNet 残差注入]]

$$
h_i' = h_i + \Delta h_i(\text{enc}(M^+))
$$

**含义**: 将轨迹潜码编码后，逐 Transformer block 产生残差 $\Delta h_i$，叠加到视频骨干第 $i$ 层输出上，在不修改骨干权重的前提下注入轨迹控制信号。

**符号说明**:
- $h_i$：冻结视频骨干第 $i$ 个 Transformer block 的输出
- $\Delta h_i$：ControlNet 产生的残差，维度已从 512 投影回 3,072
- $\text{enc}(M^+)$：通过视频 VAE 编码器得到的未来遮罩潜码

### 公式4: [[Distribution Matching Distillation|分布匹配蒸馏梯度]]

$$
\begin{aligned}
\nabla_\theta \mathcal{L}_{DMD} &= \mathbb{E}\bigl[w(\tau)\bigl(s_\phi^{fake}(z_\tau^g, \tau) - s_\psi^{real}(z_\tau^g, \tau)\bigr) \frac{\partial z_\tau^g}{\partial \theta}\bigr]
\end{aligned}
$$

**含义**: 通过最小化学生生成分布与真实数据分布之间的差距，用两个得分模型（冻结的真实得分、可训练的伪造得分）之差引导单步生成器 $\theta$ 的梯度更新。

**符号说明**:
- $s_\psi^{real}$：在真实数据上训练的冻结得分模型
- $s_\phi^{fake}$：在学生生成样本上训练的可训练伪造得分模型
- $z_\tau^g$：单步学生从噪声生成的样本在时刻 $\tau$ 的插值
- $w(\tau)$：时间步权重函数

### 公式5: [[Adapter Realignment|适配器对齐损失]]

$$
\mathcal{L}_{adapt} = \mathcal{L}_{recon}(z_{adapt}, z_{slow}) + \lambda_{adv} \cdot \mathcal{L}_{adv}(D_\psi(z_{adapt}))
$$

**含义**: 快速适配器在冻结的单步视频生成器上微调，同时最小化与慢速 ControlNet 输出的重建误差，并通过扩散判别器对抗损失保证生成质量。

**符号说明**:
- $z_{adapt}$：快速适配器与单步生成器的联合输出
- $z_{slow}$：多步慢速 ControlNet 的参考输出
- $\mathcal{L}_{recon}$：潜码 L2 重建损失
- $D_\psi$：扩散判别器，读取 Transformer block 14、22、30 的中间特征
- $\lambda_{adv}$：对抗损失权重

### 公式6: [[Action Rollout Distillation|动作展开蒸馏损失]]

$$
\mathcal{L}_{action} = \mathbb{E}\bigl[\bigl\|a_{student}(o, M^+, \epsilon) - a_{teacher}^{10+10}(o, M^+, \epsilon)\bigr\|^2\bigr]
$$

**含义**: 单步学生动作分支在相同观测、相同轨迹条件、相同初始噪声下，直接回归教师模型（10 步视频 + 10 步动作）的动作块输出。

**符号说明**:
- $a_{student}$：单步学生的动作块预测（一次前向传播）
- $a_{teacher}^{10+10}$：教师的 10+10 步多步展开动作块（参考目标）
- $o$：观测（多视角图像）
- $\epsilon$：初始噪声（两者共享，确保对齐）

---

## 关键图表

### Figure 1: Motivation & Performance Comparison / 动机与性能对比

![Figure 1](https://kevingeng3.github.io/TrajWAM/static/images/fig1-motivation-performance.png)

**说明**: (a) 现有 WAM 在快速运动场景下可能误判目标轨迹，导致抓取失败；TrajWAM 的显式轨迹引导纠正了这一问题。(b) DOMINO 基准上成功率 vs. 延迟的 Pareto 图，TrajWAM 在二者之间取得最佳平衡（43.54% SR，108.5 ms）。

### Figure 2: Method Overview / 方法总览

![Figure 2](https://kevingeng3.github.io/TrajWAM/static/images/fig2-method-overview.png)

**说明**: 左图为训练流水线：遮罩历史 → [[Trajectory ControlNet|轨迹 ControlNet]] → 冻结视频骨干 → [[Action Expert|动作专家]]，联合 [[Flow Matching]] 目标；右图为单步推理流水线：遮罩预测器先行预测轨迹，再一步生成视频与动作。

### Figure 3: Video Distillation & ControlNet Realignment / 视频蒸馏与适配器对齐

![Figure 3](https://kevingeng3.github.io/TrajWAM/static/images/fig3-video-distillation.png)

**说明**: (a) 视频骨干 [[Distribution Matching Distillation|DMD]] 蒸馏：真实得分模型（冻结）与伪造得分模型（可训练）协同引导单步生成器；(b) 轨迹适配器对齐：复制慢速 ControlNet 为快速适配器，在单步生成器上微调，联合重建损失与扩散判别器对抗损失。

### Figure 4: Action Rollout Distillation / 动作展开蒸馏

![Figure 4](https://kevingeng3.github.io/TrajWAM/static/images/fig4-action-rollout-distillation.png)

**说明**: 10+10 步慢速教师生成动作块 $a_{teacher}$，单步学生在相同条件下通过 L2 损失直接回归教师输出，实现动作推理从多步到单步的跨越。

### Figure 5: Real-World Setups / 真实世界实验设置

![Figure 5](https://arxiv.org/html/2610.25763v1/realworld.png)

**说明**: (a) 传送带抓取任务：目标物体在匀速运动的传送带上，三种速度（低/中/高）、四种物体类型；(b) 旋转圆盘抓取任务：目标在旋转圆盘上做圆周运动，同样三速四类配置。实验平台为双 6-DoF AgileX Piper 机械臂，手腕 + 固定 D435i 摄像头。

### Figure 6: Real-World Qualitative Comparison / 真实世界定性对比

![Figure 6](https://kevingeng3.github.io/TrajWAM/static/images/fig6-real-world-comparison.png)

**说明**: 传送带抓取任务中 π0.5、Fast-WAM、TrajWAM 的抓取过程对比（0–4 s，每秒一帧）。TrajWAM 能提前预判目标位置，在物体到达合适区域时精准抓取，而 Fast-WAM 因缺乏显式轨迹预测常常抓空。

### Figure 7: Mask Source Robustness Ablation / 遮罩来源鲁棒性消融

（图片未在 arXiv HTML 中提供外链，见论文 PDF Fig. 7）

**说明**: 在 DOMINO 3,500 次试验中，对比三种遮罩来源：地面真值遮罩、SAM3（推理时默认）、GroundedSAM2（零样本替换，无需重训练）。GroundedSAM2 仅比 SAM3 低 2.11 个百分点，说明 TrajWAM 对遮罩来源具有较强鲁棒性。

### Figure 8: Trajectory Prediction Visualization / 轨迹预测可视化

![Figure 8](https://arxiv.org/html/2610.25763v1/mask_prediction_sim_real.png)

**说明**: 在三个 DOMINO 任务和两个真实世界场景中，可视化历史遮罩输入与预测的未来 8 帧遮罩序列。预测遮罩准确跟踪目标运动轨迹，为后续视频和动作生成提供可靠的空间先验。

### Figure 9: Video Prediction on Simulation / 仿真视频预测

![Figure 9](https://arxiv.org/html/2610.25763v1/video_sim.png)

**说明**: 在 DOMINO 训练演示上，单步生成器与地面真值未来遮罩配合生成的未来帧预测，展示了轨迹控制对视频生成质量的提升。

### Figure 10: Video Prediction on Real World / 真实世界视频预测

![Figure 10](https://arxiv.org/html/2610.25763v1/video_real.png)

**说明**: 在传送带和旋转圆盘两个真实任务中，TrajWAM 的视频预测效果，验证了模型在 sim-to-real 迁移后仍能生成准确的轨迹引导视频。

### Table I: DOMINO 基准仿真结果（35 任务）

| Method | SR (%) | MS | Latency (ms) |
|--------|--------|----|--------------|
| OpenVLA | 1.54 | 6.10 | 173.6 |
| π0 | 8.17 | 23.96 | 52.9 |
| π0.5 | 9.63 | 26.17 | 59.1 |
| RDT-1B | — | — | — |
| Gr00T-N1 | — | — | — |
| PUMA | 17.20 | 34.97 | 85.4 |
| Fast-WAM | 21.97 | 37.27 | 183.7 |
| InternVLA-A1.5 | 29.30 | 42.50 | 552.2 |
| **TrajWAM** | **43.54** | **54.07** | **108.5** |

**表格说明**: TrajWAM 在成功率上超越第二名 InternVLA-A1.5 达 14.24 个百分点，延迟（108.5 ms）优于 Fast-WAM 和 InternVLA-A1.5，整体形成 Pareto 最优解。

### Table II: 真实机器人实验结果（480 次试验/方法）

| Method | Conveyor-belt SR | Rotating-disc SR | Overall SR |
|--------|-----------------|-----------------|-----------|
| π0.5 | 54.2% | 44.2% | 49.2% |
| Fast-WAM | 29.6% | 27.9% | 28.8% |
| **TrajWAM** | **77.9%** | **70.4%** | **74.2%** |

**表格说明**: TrajWAM 在综合成功率上超越 π0.5 达 25 个百分点，超越 Fast-WAM 达 45.4 个百分点。高速传送带场景 TrajWAM 达 65.0%（vs π0.5 27.5%），高速旋转圆盘达 70.0%（vs π0.5 35.0%）。瓶子在高速传送带场景下仍具挑战（7/20 成功）。

### Table III: 消融实验

**(a) 轨迹条件消融（10 任务子集）**

| 轨迹来源 | SR (%) | MS |
|---------|--------|----|
| Black（全零遮罩） | 32.00 | 44.32 |
| Static（重复当前帧遮罩） | 20.90 | 33.72 |
| **Predicted（预测遮罩）** | **47.30** | **58.32** |

**关键发现**: 预测遮罩比静态遮罩高出 26.4 个百分点，比全零遮罩高出 15.3 个百分点，证明轨迹预测质量对最终性能至关重要。

**(b) 动作蒸馏策略消融（12 任务子集）**

| 蒸馏策略 | SR (%) | MS |
|---------|--------|----|
| 无蒸馏单步 | 7.39 | 15.82 |
| DMD（仅） | 1.08 | 19.10 |
| Consistency Distillation | 18.42 | 36.33 |
| **Action Rollout（ours）** | **42.25** | **53.38** |

**关键发现**: 动作展开蒸馏策略远优于一致性蒸馏（+23.83 个百分点），说明让学生直接匹配多步教师的动作输出比传统蒸馏更有效。

**(c) 适配器对齐消融（10 任务子集）**

| 配置 | SR (%) | MS |
|-----|--------|----|
| w/o 适配器对齐 | 42.20 | 52.76 |
| **w/ 适配器对齐** | **43.20** | **54.11** |

**关键发现**: 适配器对齐带来约 1 个百分点的稳定提升，有效减少快速 ControlNet 与快速视频骨干之间的分布偏差。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| DOMINO (仿真) | 35 任务，每任务 50 条演示（共 1,750 条） | AgileX Aloha 机械臂，Level 1 动态（α=0.1） | 训练 + 测试 |
| 传送带抓取（真实） | 200 条遥操作演示 | 三速度、四物体类型，240 次评估 | 训练 + 测试 |
| 旋转圆盘抓取（真实） | 200 条遥操作演示 | 三速度、四物体类型，240 次评估 | 训练 + 测试 |

### 实现细节

- **Backbone**: 5B 参数预训练视频生成模型（冻结），ControlNet 隐藏维度 512
- **遮罩预测器**: 轻量 2D U-Net，输入 120×160 分辨率
- **训练阶段**: 联合训练（flow matching）→ 视频 DMD 蒸馏（5,000 步）→ 适配器对齐（1,000 步）→ 动作蒸馏（2,000 步）
- **动作蒸馏优化器**: AdamW，lr = 1e-6，batch = 4
- **视频 DMD**: EMA 衰减 0.99，全局 batch 8
- **硬件**: H200（训练）、A800（DOMINO 评估）、RTX 5090（真实世界推理，延迟测量）
- **延迟**: 108.5 ms（RTX 5090，含遮罩预测 + 单步推理）

### 特殊处理：接触后条件归零

在 Dynamic Interception 任务中，首次抓手-目标接触后，轨迹适配器残差归零，避免接触后的遮罩预测干扰抓取执行。

---

## 批判性思考

### 优点

1. **轨迹显式建模**: 相比全帧视频预测，物体遮罩轨迹更紧凑、更精确地表达目标运动，是动态操作场景的有效先验。
2. **解耦蒸馏设计合理**: 三阶段解耦避免了视频、轨迹适配器、动作三个组件之间的梯度干扰，各阶段可独立调优。
3. **真实世界强鲁棒性**: GroundedSAM2 替换 SAM3 仅损失 2.11 个百分点，说明系统对具体分割器选择不敏感。
4. **充分的 DOMINO 细粒度结果**: 附录提供了全部 35 个任务的逐任务结果，可重现性较强。

### 局限性

1. **实验聚焦于独立运动目标**: 现实中物体运动常与机器人动作耦合（如推动、托举），当前框架尚未处理动作依赖的目标动态。
2. **遮罩预测失效场景**: 遮罩预测器仅接受固定第三视角输入；目标遮挡、视野外运动、多目标混淆场景下预测质量待验证。
3. **物体类别依赖**: 高速传送带上的瓶子仍然困难，可能与物体形状细长导致遮罩预测困难相关。
4. **参数量和训练成本**: 5B 骨干（即使冻结）对部署资源要求高；论文未报告总训练时间。

### 潜在改进方向

1. 扩展至动作-目标联合动态建模（机器人推动或接触导致目标运动）
2. 多目标同时跟踪和轨迹预测
3. 更轻量化的视频骨干替代（如 1B/2B 规模），降低部署门槛
4. 在线实时适应：遮罩预测器在部署中持续更新

### 可复现性评估

- [ ] 代码开源（暂无）
- [ ] 预训练模型（暂无）
- [x] 训练细节完整（附录提供详细超参数）
- [x] 数据集可获取（DOMINO 为公开基准；真实世界演示数据未公开）

---

## 关联笔记

### 基于

- [[Flash-WAM]]: 一致性蒸馏减少 WAM 推理步数，TrajWAM 的蒸馏策略在此基础上进一步改进
- [[Fast-WAM]]: 动态操作 WAM 的代表基线，TrajWAM 在其基础上显式引入轨迹条件
- [[PUMA]]: 无视频想象的动态操作方法，代表了"跳过世界模型"路线的上界

### 对比

- [[π0]]: 通用机器人策略，作为动态操作性能下界参考
- [[π0.5]]: π0 的升级版，真实世界实验中的强基线
- [[InternVLA]]: 最高 DOMINO 成功率的竞争者（29.30% vs 43.54%）

### 方法相关

- [[World Action Model]]: TrajWAM 所属的方法范式
- [[Flow Matching]]: 视频和动作联合训练的核心目标函数
- [[Distribution Matching Distillation]]: 视频骨干蒸馏的核心技术
- [[ControlNet]]: 轨迹条件注入的架构来源
- [[SAM]]: 运行时目标遮罩提取工具

### 硬件/数据相关

- [[AgileX Piper]]: 真实世界实验使用的 6-DoF 机械臂平台
- [[DOMINO]]: 动态操作 35 任务仿真基准

---

## 速查卡片

> [!summary] TrajWAM
> - **核心**: 用预测的物体遮罩轨迹显式引导 WAM 的视频想象和动作预测
> - **方法**: 遮罩预测器 + Trajectory ControlNet + 非对称注意力 + 解耦三段蒸馏
> - **结果**: DOMINO 43.54% SR（+14.24pp vs SOTA），真实机器人 74.2%，延迟 108.5 ms
> - **代码**: 暂无（项目主页: https://kevingeng3.github.io/TrajWAM）

---

*笔记创建时间: 2026-10-21*
