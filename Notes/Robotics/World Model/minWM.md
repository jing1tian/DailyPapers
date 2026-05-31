---
title: "minWM: A Full-Stack Open-Source Framework for Real-Time Interactive Video World Models"
method_name: "minWM"
authors: [Min Zhao, Hongzhou Zhu, Bokai Yan, Zihan Zhou, Yimin Chen, Wenqiang Sun, Kaiwen Zheng, Guande He, Xiao Yang, Chongxuan Li, Fan Bao, Jun Zhu]
year: 2026
venue: arXiv
tags: [video-world-model, diffusion-distillation, camera-control, autoregressive-generation, real-time-inference]
zotero_collection: Robotics/World Model
image_source: local
arxiv_html: https://arxiv.org/html/2605.30263
created: 2026-05-31
---

# 论文笔记：minWM: A Full-Stack Open-Source Framework for Real-Time Interactive Video World Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Tsinghua University / Shengshu AI |
| 日期 | May 2026 |
| 项目主页 | [GitHub](https://github.com/shengshu-ai/minWM) |
| 对比基线 | [[HunyuanVideo]] / [[Wan2.2\|Wan2.1]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.30263) / [Code](https://github.com/shengshu-ai/minWM) |

---

## 一句话总结

> minWM 是一个端到端开源框架，将现有的双向视频扩散基础模型（T2V/TI2V）转化为支持相机控制的实时交互式自回归世界模型，实现高达 236× 的延迟压缩。

---

## 核心贡献

1. **全栈框架**: 覆盖数据构建、可控微调、[[自回归扩散|AR 扩散]]训练、少步蒸馏、低延迟推理的完整 pipeline，首个完整开源实现
2. **PRoPE 相机控制**: 通过[[PRoPE|投影旋转位置编码]]将相机内参/外参注入自注意力机制，实现精确的相机轨迹跟随
3. **Causal Forcing++**: 升级版因果蒸馏方案，用在线一致性蒸馏替代离线 ODE 轨迹生成，消除存储开销并保持等价性
4. **极致延迟压缩**: HY1.5-8B 实现 223.75× 加速，Wan2.1-1.3B 实现 236.64× 加速（单 A800 GPU）

---

## 问题背景

### 要解决的问题

将预训练的大规模[[视频扩散模型|视频扩散基础模型]]（如 [[HunyuanVideo]]、[[Wan2.2|Wan2.1]]）转化为适合实时交互的[[世界模型]]。实时交互要求**可控、因果、低延迟**的逐帧生成能力。

### 现有方法的局限

- 现有[[视频生成世界模型|视频生成方法]]采用双向注意力，无法实现因果自回归生成
- 多步扩散采样（DDPM/DDIM）延迟极高，单帧首帧延迟高达数百秒
- 缺乏统一、可复现、可扩展的完整转化框架；已有工作（如 GameGen-O、Oasis）未完整开源

### 本文的动机

通过分阶段 pipeline（相机控制微调 → AR 扩散训练 → 少步蒸馏）逐步将双向模型转化为实时可用的世界模型，同时保持基础模型的视频质量。

---

## 方法详解

### 整体架构

![[minWM_fig1_overview.png]]

**图说明**: minWM 的完整 pipeline。从 T2V/TI2V 基础模型出发，经过相机控制微调（Phase 1）和 AR 扩散蒸馏（Phase 2，共 3 个阶段），最终产出支持相机控制的少步实时 AR 世界模型。

minWM 采用**两阶段转化**架构：
- **输入**: 视频帧序列 + 相机参数（内参 $K_i$、外参 $T_i^{cw}$）
- **Phase 1**: 在相机标注视频上微调双向扩散模型，注入 [[PRoPE]] 实现相机可控
- **Phase 2**: 三阶段 [[因果蒸馏|AR 扩散蒸馏]]，将多步双向模型转化为少步 AR 生成器
- **输出**: 4 步实时自回归生成，每块 4 帧，分辨率 480×832

### Phase 1: PRoPE 相机控制注入

#### PRoPE（Projective Rope Embeddings）

**设计动机**: 用[[旋转位置编码|RoPE]]的思路将相机投影矩阵编码进自注意力，使模型学习 token 间的相对相机变换。

**具体实现**:

每帧的相机参数被编码为**提升投影矩阵**（lifted projective matrix）：

$$
\tilde{P}_i = \begin{bmatrix} K_i & \mathbf{0} \\ T_i^{cw} \\ \mathbf{e}_4^T \end{bmatrix} \in \mathbb{R}^{4 \times 4}
$$

其中 $K_i$ 为相机内参矩阵，$T_i^{cw} \in SE(3)$ 为世界到相机的外参变换，$\mathbf{e}_4 = (0,0,0,1)^T$。

将其整合进块对角变换：

$$
D_t^{\text{PRoPE}} = \begin{bmatrix} I_{d/8} \otimes \tilde{P}_{i(t)} & \mathbf{0} \\ \mathbf{0} & \begin{bmatrix} \text{RoPE}_{d/4}(x_t) & \mathbf{0} \\ \mathbf{0} & \text{RoPE}_{d/4}(y_t) \end{bmatrix} \end{bmatrix}
$$

注意力计算改写为 GTA（Grouped Token Attention）形式：

$$
\text{Attn}_{\text{PRoPE}}(Q,K,V) = D^{\text{PRoPE}} \odot \text{Attn}\bigl((D^{\text{PRoPE}})^T \odot Q,\ (D^{\text{PRoPE}})^{-1} \odot K,\ (D^{\text{PRoPE}})^{-1} \odot V\bigr)
$$

**效果**: token 对之间隐式编码了相对投影变换 $\tilde{P}_{i(t_1)} \tilde{P}_{i(t_2)}^{-1}$，同时包含相对内参和相对位姿。

### Phase 2: AR 扩散蒸馏（3 阶段）

#### Stage 1: AR 扩散训练（Teacher Forcing）

用[[Teacher Forcing]]将双向模型转化为自回归模型：给定已生成的历史帧（干净 + 含噪拼接），在因果注意力掩码下进行标准扩散训练。

- **优点**: 保留了相机可控能力
- **局限**: 仍需多步采样；存在**曝光偏差**（exposure bias）——训练时用真实前缀，推理时用生成前缀

#### Stage 2a: Causal ODE 初始化

训练少步生成器，回归 PF-ODE 轨迹上的干净帧：

$$
\theta^* = \arg\min_\theta \mathbb{E}\Bigl[\|G_\theta(x_t^i,\, x_{\text{gt}}^{<i},\, t) - x_0^i\|^2\Bigr]
$$

**符号说明**:
- $G_\theta$: 少步生成器
- $x_t^i$: 第 $i$ 帧在时刻 $t$ 的含噪版本
- $x_{\text{gt}}^{<i}$: 来自真实数据的历史前缀帧
- $x_0^i$: 干净目标帧
- $t \sim \mathcal{S}$: 从预定义少步集合中采样的时间步

**含义**: 通过 AR 扩散教师生成 ODE 轨迹，少步学生在这些轨迹上做回归，获得初步的少步能力。缺点是需要大量离线 ODE 数据。

#### Stage 2b: Causal CD（一致性蒸馏，Causal Forcing++）

用在线[[一致性蒸馏|Consistency Distillation]]消除离线数据依赖：

$$
\theta^* = \arg\min_\theta \mathbb{E}\Bigl[w(t)\, d\bigl(G_\theta(x_t^i,\, x_{\text{gt}}^{<i},\, t),\; G_{\theta^-}(\hat{x}_{t-\Delta t}^i,\, x_{\text{gt}}^{<i},\, t-\Delta t)\bigr)\Bigr]
$$

**符号说明**:
- $w(t)$: 时间步依赖的权重函数
- $d(\cdot, \cdot)$: 预定义范数下的距离度量
- $\hat{x}_{t-\Delta t}^i$: 单步 ODE 求解得到的帧
- $\theta^-$: $\theta$ 的 EMA（指数移动平均），带 stop-gradient

**含义**: 用 EMA 学生网络作为一致性目标，实现在线蒸馏，理论上等价于 Causal ODE 但无需离线数据。

#### Stage 3: Asymmetric DMD（分布匹配蒸馏）

用双向教师模型对少步 AR 学生做分布对齐，通过 [[Distribution Matching Distillation|DMD]] 梯度回传：

$$
\nabla_\theta \mathbb{E}_t\bigl[D_{\text{KL}}(p_{\theta,t}(\tilde{x}_t) \| p_{\text{data},t}(\tilde{x}_t))\bigr] = -\mathbb{E}\bigl[(s_{\text{real}}(\tilde{x}_t, t) - s_{\text{fake}}(\tilde{x}_t, t))\, \tfrac{\partial \tilde{x}}{\partial \theta}\bigr]
$$

**符号说明**:
- $\tilde{x}$: 少步 AR 模型自回归展开的视频
- $s_{\text{real}}$: 来自冻结双向扩散教师的 score
- $s_{\text{fake}}$: 来自在线训练扩散判别器的 score
- $D_{\text{KL}}$: KL 散度，对齐学生分布与真实数据分布

**含义**: 用高质量双向教师提升少步 AR 模型的生成质量，解决 Stage 1/2 中曝光偏差和质量上限问题。

---

## 关键图表

### Figure 2: 相机控制生成效果

![[minWM_fig2_camera_control.png]]

**说明**: 蒸馏后的少步 AR 模型在不同相机动作（前进、后退、旋转等）下的生成结果，验证蒸馏过程有效保留了相机可控能力。

### Figure 3: 训练数据对相机控制的影响

| 子图 | 数据来源 | 结果 |
|------|----------|------|
| (a) | SpatialVid（感知估计位姿） | 控制效果不稳定，可靠性差 |
| (b) | DL3DV（3D 重建 + 渲染轨迹） | 成功学习控制 |
| (c) | WorldPlay 生成轨迹 | 有效，适合开源版本 |

![[minWM_fig3_training_data.png]]

**关键发现**: **地面真值相机位姿**对可控性至关重要；基于感知估计的位姿（SpatialVid）在当前设定下无法有效训练。

### Figure 4: 训练步数对相机控制的影响

| 步数 | 现象 |
|------|------|
| 1K–2K | 完全不可控 |
| ~5K | 开始出现相机可控性 |
| 8K | 强且可靠的控制 |

![[minWM_fig4_training_steps.png]]

**关键发现**: 相机控制能力需要充足的训练步数才能涌现，HY1.5 需要约 8K 步。

### Figure 5: Batch Size 对相机控制的影响

| Batch Size | 结果 |
|-----------|------|
| < 4 | 持续失败，无法学习控制 |
| 8 | 显著改善，但仍有不稳定性 |
| 16 | 成功训练，高可控性 |

![[minWM_fig5_batch_size.png]]

**关键发现**: 相机控制训练对 batch size 有最低要求（Wan2.1 至少需要 16）；过小 batch 导致梯度方差过大无法收敛。

### Table 1: 首帧延迟对比（单 A800 GPU，不含 VAE）

| 基础模型 | 模型类型 | 首帧延迟 (s) | 加速比 |
|----------|----------|------------|--------|
| HY1.5-8B | 多步双向 | 1.041 | 1.00× |
| HY1.5-8B | 多步 AR | 0.109 | 9.52× |
| HY1.5-8B | **少步 AR（minWM）** | **0.00465** | **223.75×** |
| Wan2.1-1.3B | 多步双向 | 0.269 | 1.00× |
| Wan2.1-1.3B | 多步 AR | 0.0286 | 9.39× |
| Wan2.1-1.3B | **少步 AR（minWM）** | **0.00114** | **236.64×** |

**关键发现**: 少步 AR 模型相比多步双向基线实现 200× 以上延迟压缩；模型规模越大加速效果越显著。

---

## 实验

### 数据集

| 数据集 | 来源 | 相机位姿质量 | 用途 |
|--------|------|------------|------|
| DL3DV | 3D 重建 + 重渲染 | 高（GT 轨迹） | 相机控制训练 |
| OpenVid + WorldPlay | 视频生成轨迹 | 中（生成轨迹） | 开源版本训练数据 |
| SpatialVid | 感知估计 | 低（预测值） | 消融实验（失败案例） |

### 实现细节

**HY1.5-TI2V-8B**:
- Batch size: 32，学习率: $1\times10^{-5}$
- 双向微调: 8K 步；Stage 1: 4K 步；Stage 2: 1.5K 步；Stage 3: 500 步

**Wan2.1-T2V-1.3B**:
- Batch size: 32，学习率: $2\times10^{-6}$
- 双向微调: 5K 步；Stage 1: 4K 步；Stage 2: 2K 步；Stage 3: 200 步

**通用设置**:
- 输出分辨率: 480×832，视频长度: 77 帧
- AR chunk 大小: 4 latent 帧
- 少步推理: 4 步

### 可视化结果

Figure 2 展示了 minWM 在多种相机运动下的生成质量，包括平移（前/后/左/右）和旋转，视觉效果与多步双向模型相当，但推理速度提升 200× 以上。

---

## 批判性思考

### 优点

1. **完整性**: 覆盖全流程的开源框架，填补了视频世界模型端到端实现的空白
2. **通用性**: 支持不同架构（MMDiT、Cross-Attention DiT），在 1.3B 和 8B 两种规模上均有验证
3. **Causal Forcing++**: 消除了离线 ODE 轨迹的存储/计算开销，工程价值高
4. **消融实验详细**: 数据质量、训练步数、batch size 的影响都有系统分析

### 局限性

1. **仅评估延迟**: 缺乏对生成质量（FVD、FID、感知评测）的定量比较，难以评估质量-速度 trade-off
2. **感知位姿失败未解释**: SpatialVid 失败的原因未深入分析，是否可以通过位姿精度过滤改善未知
3. **当前仅支持相机控制**: 其他动作控制（pose、手势等）计划中但未实现
4. **延迟不含 VAE**: 实际部署时 VAE 解码延迟可能显著，完整延迟数据缺失

### 潜在改进方向

1. 将位姿估计质量过滤引入 SpatialVid 数据，扩展可用训练数据规模
2. 增加对象级别的动作条件（姿态、手势控制）
3. 多 GPU 序列并行已在仓库中支持，可进一步优化分布式推理

### 可复现性评估

- [x] 代码开源（Apache-2.0）
- [x] 预训练模型（checkpoints 公开）
- [x] 训练细节完整（步数/LR/batch size）
- [x] 数据集可获取（DL3DV 公开，WorldPlay 配套）

---

## 关联笔记

### 基于

- [[HunyuanVideo]]: 使用 HY1.5-TI2V-8B（MMDiT 架构）作为基础模型之一
- [[Wan2.2|Wan2.1]]: 使用 Wan2.1-T2V-1.3B（Cross-Attention DiT）作为另一基础模型
- [[DiffusionForcing]]: 因果蒸馏方法的前置工作（Causal Forcing 概念来源）

### 对比

- [[CausVid]]: 另一个因果视频生成方法
- [[Distribution Matching Distillation]]: Stage 3 所用 DMD 的原始方法

### 方法相关

- [[PRoPE]]: 核心相机控制编码方法
- [[Distribution Matching Distillation|DMD]]: Stage 3 分布匹配蒸馏
- [[RoPE]]: PRoPE 的设计基础
- [[Teacher Forcing]]: Stage 1 AR 训练的核心技术
- [[一致性蒸馏|Consistency Distillation]]: Stage 2b 的核心方法
- [[自回归扩散|AR Diffusion]]: 整个 Phase 2 的核心范式

### 硬件/数据相关

- [[DL3DV]]: 主要训练数据集

---

## 速查卡片

> [!summary] minWM (arXiv 2026)
> - **核心**: 将双向视频扩散模型转化为相机可控实时 AR 世界模型的全栈开源框架
> - **方法**: PRoPE 相机注入 + 三阶段 AR 蒸馏（Causal Forcing++）
> - **结果**: HY1.5-8B 223×加速，Wan2.1-1.3B 236×加速（单 A800，不含 VAE）
> - **代码**: https://github.com/shengshu-ai/minWM

---

*笔记创建时间: 2026-05-31*
