---
title: "La-JEPA: Compact Robot Policies with Latent World Modeling"
method_name: "La-JEPA"
authors: [Mateo Clémente, George Z. Xue, Amin Abyaneh, Liheng Ma, Rui Heng Yang, Tongtong Cao, Zhanguang Zhang, Yingxue Zhang]
year: 2026
venue: arXiv
tags: [world-action-model, jepa, flow-matching, robot-manipulation, compact-policy]
zotero_collection: Robotics/World Model
image_source: mixed
arxiv_html: https://arxiv.org/html/2611.08156v1
created: 2026-11-07
---

# 论文笔记：La-JEPA: Compact Robot Policies with Latent World Modeling

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Huawei Technologies, University of Toronto, McGill University, Mila |
| 日期 | November 2026 |
| 项目主页 | — |
| 对比基线 | [[JEPA-WAM]], [[Fast-WAM]], [[Faster-WAM]], [[LeapBot-WA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2611.08156) / [Code](https://github.com/mt-clemente/lajepa) |

---

## 一句话总结

> La-JEPA 在冻结的 [[V-JEPA]] 2.1 特征空间中，从零训练紧凑的 World DiT + Action DiT，实现推理速度快、参数少、性能媲美具身预训练大模型的机器人操控策略。

---

## 核心贡献

1. **紧凑 JEPA 世界-动作模型**: 策略 Transformer 在冻结 [[V-JEPA]] 2.1 ViT-L 编码器的特征空间中从零训练，推理时参数约 0.74–1.78B，比最强竞争者少约 4 倍
2. **训练时预测未来、推理时仅输出动作**: 类似 [[Fast-WAM]]，联合训练时预测未来视觉特征+动作，推理时丢弃未来 token，显著加速（比 Fast-WAM 快 2–5×）
3. **LIBERO-Plus 指令泄露发现与修正**: 揭示官方评估器将扰动参数拼入 84.7% 的指令，重新评估多个基线，Fast-WAM 提升 +19.3pp

---

## 问题背景

### 要解决的问题
[[世界-动作模型|WAM]] 和 [[VLA]] 策略通常继承大型视频生成 Transformer（如 Wan2.2），推理成本高昂，部署参数可达 6–12B+。如何在不损失性能的前提下，大幅降低参数量和延迟？

### 现有方法的局限
- [[Fast-WAM]] / [[Faster-WAM]]: 基于 Wan2.2 视频 DiT（5B 参数），推理必须加载 VAE 和 UMT5-XXL 编码器，部署参数达 11–12B
- [[JEPA-WAM]]: 有 JEPA 特征但策略 Transformer 参数仅 0.53B，依赖 Qwen2.5 语言模型初始化
- [[LeapBot-WA]]: 需要 V-JEPA 2.1 预训练策略骨干（MoT 2.75B），部署参数 10.4B

### 本文的动机
JEPA 特征（联合嵌入预测架构）本质上是为"预测"设计的，不含不必要的像素重建约束，更适合控制。视频 VAE 潜在空间仍编码生成细节，而 JEPA 特征去除冗余信息，对分布外场景鲁棒性更好。设计专用于控制的架构（而非继承自生成模型）可以在更少参数下取得更好效果。

---

## 方法详解

### 模型架构

La-JEPA 采用**双 DiT（扩散变换器）**架构：

- **输入**: 语言指令 $l$ + 多视角视觉帧（上下文帧 + 32 帧未来目标）+ 本体感知状态 $s_t$
- **视觉编码器**: 冻结 [[V-JEPA]] 2.1 ViT-L（0.305B，不更新梯度）
- **语言编码器**: 冻结 [[FLAN-T5]]-large（0.341B）
- **World DiT**: 处理观测 token + 未来 token（训练时）
- **Action DiT**: 处理动作 chunk，接受观测 token 交叉注意力
- **输出**: [[Action Chunking|动作块]] $a_{t:t+k}$（LIBERO: k=10, RoboTwin: k=24）
- **总部署参数**: S: 0.74B / B: 0.98B / L: 1.78B

### 核心模块

#### 模块1: 视觉特征提取（V-JEPA 2.1 编码）

**设计动机**: 利用 [[V-JEPA]] 的预测性特征空间——JEPA 特征去除像素重建约束，保留对控制有意义的高层语义

**具体实现**:
- 每个相机视角独立编码，提取"上下文特征"（近期帧）和"未来目标特征"（前向 32 帧）
- 特征经归一化 + 空间降维（Patch Grouping 或 Mean Pooling）
- 对编码器应用 [[LoRA]] 微调（参数约 2–6M）

#### 模块2: Tube-Motion 时序编码

**设计动机**: 比静态位置编码更好地捕捉时序动态信息

**具体实现**:
- 对视觉 token 使用 Tube-Motion 编码，携带帧内局部运动先验
- [[RoPE|旋转位置编码]]（Rotary Positional Encoding）用于时序序列
- 设计研究显示 Tube-Motion 比静态编码在 RoboTwin 上高 +3.4pp（clean）/ +3.7pp（randomized）

#### 模块3: 联合 [[Flow Matching]] 训练

**设计动机**: 同时训练未来特征预测和动作生成，让预测任务形成更有利于控制的表示

**具体实现**:
- 对未来特征和动作分别采样独立的时间步 $\tau_f, \tau_a \sim \mathcal{U}(0, 1)$ 和噪声
- World DiT 预测未来特征速度，Action DiT 预测动作速度
- 注意力掩码防止动作 token 关注未来 token（避免推理时不一致）

#### 模块4: 相机/本体感知 Dropout 鲁棒化

**设计动机**: 提高策略对传感器缺失的鲁棒性

**具体实现**:
- 随机 Either-Camera Dropout：LIBERO-Plus +10.2pp，RoboTwin 随机化场景 +7.1pp
- 随机 Proprioception Dropout：RoboTwin clean +3.8pp / randomized +4.2pp

---

## 关键公式

### 公式1: [[Flow Matching|联合流匹配训练目标]]

$$
\mathcal{L} = \mathcal{L}_{\text{future}} + \mathcal{L}_{\text{action}}
$$

$$
\mathcal{L}_{\text{future}} = \mathbb{E}_{\tau_f, \epsilon_f}\left[\|v_\theta^{W}(x_{\tau_f}^f, \tau_f, c) - (x_1^f - x_0^f)\|^2\right]
$$

$$
\mathcal{L}_{\text{action}} = \mathbb{E}_{\tau_a, \epsilon_a}\left[\|v_\theta^{A}(x_{\tau_a}^a, \tau_a, c) - (x_1^a - x_0^a)\|^2\right]
$$

**含义**: 分别对未来视觉特征和动作 chunk 施加独立的均方误差流匹配损失，联合优化 World DiT（$v_\theta^W$）和 Action DiT（$v_\theta^A$）

**符号说明**:
- $\tau_f, \tau_a \sim \mathcal{U}(0, 1)$: 未来特征和动作各自独立采样的流时间步
- $x_0^f, x_0^a$: 噪声初始状态（从标准正态采样）
- $x_1^f, x_1^a$: 目标（未来 JEPA 特征 / 真值动作）
- $c$: 条件（语言指令 + 观测特征 + 本体感知）
- $v_\theta^W, v_\theta^A$: World DiT 和 Action DiT 的速度预测

### 公式2: [[Flow Matching|推理时 Euler 积分]]

$$
x_{\tau + \Delta\tau}^a = x_\tau^a + \Delta\tau \cdot v_\theta^A(x_\tau^a, \tau, c_{\text{obs}})
$$

**含义**: 推理时仅运行 Action DiT，对动作噪声做 Euler 步积分，未来 token 被完全丢弃

**符号说明**:
- $c_{\text{obs}}$: 仅含观测特征的条件（无未来 token）
- $\Delta\tau$: 积分步长（由 denoising steps 决定，默认 10 步）

### 公式3: JEPA-WAM 全任务外推（附录回归方程）

$$
\hat{s}_{\text{clean}} = -33.09 + 1.234 \cdot s_{20}
$$

$$
\hat{s}_{\text{rand}} = -1.14 + 0.710 \cdot s_{20}
$$

**含义**: 由 JEPA-WAM 在 20 任务子集的成功率，外推估算其在全套 50 任务 RoboTwin 2.0 上的性能。估计 JEPA-WAM 全套成绩约为：clean 65.5%，randomized 25.0%，均值 45.3%（La-JEPA 全部超越）

**符号说明**:
- $s_{20}$: JEPA-WAM 在 20 任务子集的成功率
- $\hat{s}$: 估算的全套成功率

---

## 关键图表

### Figure 1: 成功率 vs 部署参数

![[La-JEPA_fig1.png|600]]

**说明**: LIBERO-Plus（修正指令）和 RoboTwin 2.0 的成功率 vs 部署参数（含冻结编码器）对数坐标散点图。La-JEPA（蓝色点）位于参数更少但性能更高的帕累托前沿。紫色=VLM 初始化，棕色=生成器初始化。

### Figure 2: La-JEPA 整体架构

![[La-JEPA_fig2.png|600]]

**说明**: 左侧为整体架构（World DiT + Action DiT 联合训练，推理时 World DiT 仅提供 KV cache）；右侧展示 V-JEPA 2.1 的上下文帧 + 未来目标帧的 Tube-Motion 编码方式。

### Figure 3: 设计消融增益

![[La-JEPA_fig3.png|600]]

**说明**: 各设计选择（A–F 组）相对基线在 LIBERO 和 RoboTwin 上的成功率增益（百分点）。未来预测（+7.0 clean）、JEPA 特征（+15.9 clean）和 Dropout 增强（+16.0 randomized）贡献最大。

### Figure 4: Clean vs Randomized 对比

![[La-JEPA_fig4.png|600]]

**说明**: 仅用 clean 数据训练的策略在 clean 场景 vs 随机化场景的成功率对比。La-JEPA B 与具身预训练基线和非预训练基线对比，展示分布外鲁棒性。

### Figure 5: 真实机器人任务

| 任务 | 图片 |
|------|------|
| Cube 放置 | ![Cube](https://arxiv.org/html/2611.08156v1/figures/robot_pick_place_cube.png) |
| Fruit 拾取 | ![Fruit](https://arxiv.org/html/2611.08156v1/figures/robot_fruit_picking.png) |
| Push T | ![PushT](https://arxiv.org/html/2611.08156v1/figures/robot_push_t.png) |
| 餐具摆放 | ![Dish](https://arxiv.org/html/2611.08156v1/figures/robot_dish_utensil.png) |

**说明**: 四个真实机器人任务场景，La-JEPA 在 Push T 上最优，餐具摆放略逊于 π0.5。

### Table 1: RoboTwin 2.0 主要结果（clean-trained, 50 tasks）

| 方法 | 具身预训练 | Clean (%) | Randomized (%) | Mean (%) |
|------|-----------|-----------|----------------|----------|
| Fast-WAM | 否 | 77.8 | 1.9 | 39.9 |
| Qwen-RobotManip-scratch | 否 | 71.6 | 22.6 | 47.1 |
| **La-JEPA S** | **否** | **85.5** | **33.9** | **59.7** |
| **La-JEPA B** | **否** | **86.2** | **35.4** | **60.8** |
| **La-JEPA L** | **否** | **86.0** | **32.1** | **59.0** |

**关键发现**: La-JEPA 在无具身预训练策略中取得 SOTA；Fast-WAM 随机化场景几乎失败（1.9%），而 La-JEPA 随机化场景约 33%，说明 JEPA 特征分布外鲁棒性显著更好。

### Table 2: LIBERO-Plus 主要结果（修正指令）

| 方法 | 具身预训练 | LIBERO-Plus Rev. (%) |
|------|-----------|----------------------|
| Fast-WAM | 否 | 70.8 |
| Cosmos Policy | 否 | 84.9 |
| NowWAM | 否 | 86.2 |
| JEPA-WAM | 否 | 77.8 |
| **La-JEPA S** | **否** | **86.3** |
| **La-JEPA B** | **否** | **87.3** |
| **La-JEPA L** | **否** | **88.0** |
| π0.5 | 是 | 85.7 |

**关键发现**: La-JEPA L 达到 88.0%，在无具身预训练策略中排名第一，甚至超过具身预训练的 π0.5（85.7%）。

### Table 3: 真实机器人结果（20 次尝试，4 任务）

| 策略 | Fruit | Cube | Push T | Dish | 均值 |
|------|-------|------|--------|------|------|
| π0.5 | 5/5 | 5/5 | 3/5 | 4.25/5 | 17.25/20 |
| Fast-WAM | 3/5 | 2/5 | 4/5 | 1/5 | 10/20 |
| Faster-WAM | 3/5 | 3/5 | 4/5 | 1.5/5 | 11.5/20 |
| LingBot-VA | 1/5 | 2/5 | 1/5 | 0.25/5 | 4.25/20 |
| **La-JEPA** | **5/5** | **4/5** | **5/5** | **3.5/5** | **17.5/20** |

**关键发现**: La-JEPA 均值（17.5/20）高于所有非具身预训练策略，与 π0.5（17.25/20）相当；Push T 任务 La-JEPA 满分，餐具任务略逊于 π0.5。

### Table 4: 策略延迟对比（ms，batch=1，10 步去噪）

| 策略 | 优化版 | CUDA Graphs | Eager |
|------|--------|-------------|-------|
| La-JEPA S | 17.78 | 34.20 | 99.96 |
| La-JEPA B | 25.16 | 42.35 | 98.21 |
| La-JEPA L | 59.72 | 88.87 | 167.93 |
| Fast-WAM | 88.09 | OOM | 193.44 |
| Faster-WAM | 50.96 | OOM | 62.70 |
| π0.5 | 71.89 | — | — |

**关键发现**: La-JEPA S 仅需 17.78ms，比 Fast-WAM（88.09ms）快 5 倍；CUDA graphs 对 Fast-WAM 爆显存，La-JEPA 两者均可用。

### Table 5: 消融实验汇总（S 模型，RoboTwin + LIBERO-Plus 增益）

| 设计选择（vs 基线） | RoboTwin Clean Δ | RoboTwin Rand Δ | LIBERO-Plus Δ |
|-------------------|-----------------|-----------------|--------------|
| A: 未来预测（vs 仅动作） | +7.0 | +3.7 | +2.1 |
| B: JEPA 特征（vs Wan2.2 VAE） | +15.9 | +16.0 | +6.0 |
| C: Tube-Motion 编码（vs 静态） | +3.4 | +3.7 | — |
| D: Either-Camera Dropout | — | +7.1 | +10.2 |
| E: Proprioception Dropout | +3.8 | +4.2 | — |
| F: 增强 + LoRA | — | +16.0 | +7.2 |

### Table 6: LIBERO-Plus 指令修正影响

| 策略 | 原始指令 (%) | 修正后 (%) | Δ |
|------|------------|-----------|---|
| Fast-WAM | 51.5 | 70.8 | **+19.3** |
| LeapBot-WA | 71.7 | 79.7 | **+8.1** |
| La-JEPA S | 80.0 | 86.3 | +6.3 |
| La-JEPA B | 82.5 | 87.3 | +4.9 |
| La-JEPA L | 84.1 | 88.0 | +3.9 |
| NowWAM | 86.2 | 86.2 | 0.0 |
| π0.5 | 84.3 | 85.7 | +1.4 |

**关键发现**: 视频生成器 WAM（Fast-WAM、LeapBot-WA）对指令更敏感，修正后提升更大；说明语言理解能力更强的模型反而受错误指令影响更大。

### Table 7: 参数量详细对比

| 方法 | 可训练参数 (B) | 推理时部署参数 (B) |
|------|--------------|-----------------|
| Fast-WAM | 6.02 | 12.4 |
| Cosmos Policy | 1.96 | 6.95 |
| π0.5 | 3.35 | 3.35 |
| JEPA-WAM | 0.55 | 1.35 |
| **La-JEPA S** | **0.096** | **0.74** |
| **La-JEPA B** | **0.337** | **0.98** |
| **La-JEPA L** | **1.14** | **1.78** |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[RoboTwin 2.0]] | 50 任务，各 100 eps/seed×3 | 有 clean 和 randomized 场景 | 主要仿真 benchmark |
| [[LIBERO-Plus]] | 10,030 变体（6 种扰动×LIBERO 任务） | 包含相机/机器人/光照/背景/噪声/布局扰动 | 鲁棒性 benchmark |
| LIBERO | 90 任务 | 标准 in-domain 评估 | 对比实验 |
| Real Robot | 4 任务（Fruit, Cube, Push T, Dish） | 实际 Franka 机器人 | 真机验证 |

### 实现细节

- **视觉编码器**: 冻结 V-JEPA 2.1 ViT-L（0.305B），+ LoRA 微调
- **语言编码器**: 冻结 FLAN-T5-large（0.341B），cross-attention 条件注入
- **架构尺寸**: S（0.094B 策略）/ B（0.335B）/ L（1.131B）
- **训练**: [[Flow Matching]]，分别对未来特征和动作采样独立时间步
- **去噪步数**: 10 步（推理），3 步性能几乎不降
- **动作 Chunk 大小**: LIBERO: k=10, RoboTwin: k=24
- **数据增强**: 颜色抖动 + Camera/Proprioception Dropout
- **硬件**: 未详细说明（代码已开源于 github.com/mt-clemente/lajepa）

### 可视化结果

La-JEPA 在 JEPA 特征 vs Wan2.2 VAE 对比中，分布外（randomized）差距最为显著（+16.0pp），说明生成 VAE 特征包含冗余纹理信息，导致 OOD 失败；而 JEPA 特征聚焦预测相关信息，更具泛化性。

---

## 批判性思考

### 优点
1. **架构设计哲学清晰**: 明确提出"控制导向架构"优于"继承生成架构"，并以实验验证
2. **系统性消融**: 6 组（A–F）消融设计，每个选择都有清晰的量化贡献
3. **基础设施贡献**: 发现并修正了 LIBERO-Plus 评估器的指令泄露 bug，重新评估多个基线，提升了社区 benchmark 可靠性

### 局限性
1. **超参数未充分调优**: 论文自认 recipe 超参数（如 Dropout 率、LoRA rank）未穷举搜索
2. **随机化场景性能仍差距大**: La-JEPA 随机化约 33%，具身预训练模型可超过 50%，差距明显
3. **真机评估规模有限**: 仅 4 任务、单一实验室设置，泛化性未充分验证
4. **无大规模具身预训练**: 未与在大规模机器人数据集上预训练的策略对比（依赖人工演示数据）

### 潜在改进方向
1. 在大规模机器人演示数据（如 Open X-Embodiment）上预训练 La-JEPA 策略骨干
2. 结合 V-JEPA 视频预测能力，探索更深层的 ViT 层特征
3. 将 World DiT 的预测能力用于在线 planning（MBRL 范式）

### 可复现性评估
- [x] 代码开源（github.com/mt-clemente/lajepa）
- [ ] 预训练模型（未提及）
- [x] 训练细节完整（附录包含完整 hyper-parameter 表格）
- [x] 数据集可获取（RoboTwin 2.0 和 LIBERO 均公开）

---

## 关联笔记

### 基于
- [[V-JEPA]]: 核心视觉编码器（冻结 ViT-L 权重）
- [[Flow Matching]]: 训练和推理框架

### 对比
- [[JEPA-WAM]]: 同样基于 JEPA 特征，但策略从 Qwen 初始化，参数更少但本文超越
- [[Fast-WAM]]: 视频 DiT 生成器 WAM，参数大、延迟高
- [[LeapBot-WA]]: 具身预训练 MoT 策略，本文在无预训练条件下可超越
- [[Faster-WAM]]: Fast-WAM 的加速版本

### 方法相关
- [[World-Action Model]]: 本文所属方法族
- [[DiT（扩散变换器）]]: World DiT 和 Action DiT 的基础架构
- [[Action Chunking]]: 输出动作块而非单步动作
- [[RoPE|旋转位置编码]]: 时序序列位置编码方式

### 硬件/数据相关
- [[RoboTwin 2.0]]: 主要仿真评估 benchmark
- [[LIBERO-Plus]]: 鲁棒性 benchmark（本文发现指令泄露 bug）

---

## 速查卡片

> [!summary] La-JEPA: Compact Robot Policies with Latent World Modeling
> - **核心**: 在冻结 V-JEPA 2.1 特征空间中从零训练紧凑 World+Action DiT
> - **方法**: 联合流匹配训练（预测未来 JEPA 特征 + 动作），推理时丢弃未来 token
> - **结果**: 无具身预训练 SOTA（RoboTwin 2.0 mean 60.8%，LIBERO-Plus 88.0%），比 Fast-WAM 快 5×，真机媲美 π0.5
> - **代码**: https://github.com/mt-clemente/lajepa

---

*笔记创建时间: 2026-11-07*
