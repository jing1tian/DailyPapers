---
title: "OSCAR: Omni-Embodiment Action-Conditioned World Model for Robotics"
method_name: "OSCAR"
authors: [Zhuoyuan Wu, Jun Gao]
year: 2026
venue: arXiv
tags: [world-model, action-conditioned, cross-embodiment, video-generation, policy-evaluation]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2606.04463
created: 2026-06-05
---

# 论文笔记：OSCAR: Omni-Embodiment Action-Conditioned World Model for Robotics

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Peking University; University of Michigan / NVIDIA |
| 日期 | June 2026 |
| 项目主页 | https://wuzy2115.github.io/oscar-project-page/ |
| 对比基线 | [[Kinema4D]], [[IRASim]], [[TesserAct]], [[Cosmos-Predict2.5]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.04463) |

---

## 一句话总结

> OSCAR 以 2D 运动学骨架渲染为统一条件表示，在单卡 GH200 上训练出跨多种机器人本体的动作条件视频世界模型，可作为真实策略评估的高相关代理。

---

## 核心贡献

1. **统一条件表示（Skeleton Rendering）**: 将不同机械臂和人手的运动学链投影为 2D 骨架图，无需机器人特定纹理或资产，实现天然跨本体泛化。
2. **标准化数据流水线**: 从 7 个异构来源（5 个机器人 + 2 个人类第一视角数据集）整合 2.16M 片段，经四阶段过滤与语义去重后得到 180,657 高质量片段。
3. **高相关策略评估代理**: 在 [[RoboArena]] 基准上，OSCAR 的虚拟评估结果与真实机器人评估的 Pearson 相关系数达 r = 0.750，超越所有基线。

---

## 问题背景

### 要解决的问题

如何构建一个能够**跨不同机器人本体**精确执行动作条件、并可替代真实机器人进行策略评估的视频世界模型。

### 现有方法的局限

- **纯文本条件**（TesserAct, Cosmos-Predict2.5）：语言无法精确描述帧级动作序列，动作追踪精度差。
- **隐式动作嵌入**（[[IRASim]], Ctrl-World）：[[Latent Action]] 信号隐晦，在不同本体间存在分布偏移，泛化性弱。
- **密集几何渲染**（Kinema4D）：使用 mesh / 点图过拟合于特定机器人外观，参数量高达 14B 且跨本体能力有限。
- **数据稀缺性**：机器人数据集场景多样性有限，训练集规模受限于相机标定和运动学标注的可用性。

### 本文的动机

2D 骨架投影只依赖运动学规格，不依赖机器人纹理，可以同时处理机械臂和 MANO 人手模型，从而无缝融合机器人和人类第一视角数据，解决数据稀缺性与跨本体泛化两大瓶颈。

---

## 方法详解

### 模型架构

OSCAR 采用 **[[Diffusion Transformer]]（DiT）** 架构：

- **基础模型**: [[Cosmos-Predict2.5]]-2B（Diffusion Transformer + VAE）
- **输入**: 第一帧 RGB 图像 + 动作条件骨架序列 $\{S_t\}_{t=1}^T$
- **条件注入**: 骨架序列与视频帧并行通过 [[VAE]] 编码，**在 latent 空间直接相加后**送入 DiT 去噪
- **输出**: 预测的 RGB 视频序列
- **总参数**: 2B

![Figure 2: OSCAR 架构总览](https://arxiv.org/html/2606.04463v2/x3.png)

**说明**：三个核心组件：① 条件编码（第一帧 + 骨架 VAE latent），② 条件注入（latent 求和），③ 视频生成（DiT 去噪 + VAE 解码）。

### 核心模块

#### 模块1: 2D 骨架渲染（Skeleton Rendering）

**设计动机**: 利用 [[Forward Kinematics]] 将运动学链投影到 2D 像素空间，作为不依赖外观的统一表示，实现机械臂与人手的跨本体统一处理。

**具体实现**:
- 从机器人 URDF 模型计算各关节的世界坐标变换 $\{T_{k,t}\}_{k=1}^K$
- 使用相机内参 $K_\text{cam}$ 和外参 $T^\text{cam}_\text{world}$ 投影到像素坐标 $(u_{k,t}, v_{k,t})$
- 将关节点绘制为圆圈、边绘制为线段，渲染在黑色画布上
- 对人类手部扩展：使用 [[MANO]] 模型拓扑，相同投影流程直接适用

#### 模块2: 数据流水线（4 阶段）

**设计动机**: 从 7 个异构数据源（含不同分辨率、标注格式、帧率）构建统一高质量训练集。

**具体实现**:
1. **数据汇聚（Curation）**: 合并 5 个机器人数据集（RH20T, InternData, DROID, AgiBot, AIROA-MoMa）+ 2 个人类第一视角数据集（EgoDex, EPIC-Kitchens）
2. **数据过滤（Filtering）**: 最小长度 ≥70 帧、静态相机、有意义动作、骨架可见性阈值
3. **语义去重（Deduplication）**: 两阶段去重——用 [[SigLIP]] 嵌入做视觉聚类（阈值 0.95）+ 轨迹 RMS 距离验证
4. **自动标注（Captioning）**: 使用 Qwen3-VL-30B 以 15fps 采样，生成强调运动和场景的详细文本描述

#### 模块3: 频率调和批次加权（Frequency-Tempered Batch Weighting）

**设计动机**: 平衡来自不同大小数据集的采样频率，避免大数据集主导训练。

**具体实现**: 按每个数据集帧数的 $1/T$ 次幂比例采样，温度 $T=3$。

---

## 关键公式

### 公式1: [[Rectified Flow|修正流训练目标]]

$$
\mathcal{L}_\text{RF} = \mathbb{E}_{t,z_0,\epsilon} \left\| v_\theta(z_t, t, c) - (\epsilon - z_0) \right\|^2_2
$$

**含义**: 训练速度场网络 $v_\theta$ 预测从噪声到数据的流速，作为 OSCAR 的基础生成目标。

**符号说明**:
- $z_t = (1-t)z_0 + t\epsilon$：$t$ 时刻的插值 latent，$t \sim \mathcal{U}(0,1)$
- $z_0$：真实视频的 VAE latent
- $\epsilon \sim \mathcal{N}(0, I)$：高斯噪声
- $c$：条件信息（骨架 + 第一帧）
- $v_\theta$：待学习的速度场网络

### 公式2: [[Forward Kinematics|正向运动学骨架投影]]

$$
\{T_{k,t}\}_{k=1}^K = \text{FK}(q_t, \mathcal{M})
$$

$$
(u_{k,t}, v_{k,t}) = \pi\!\left(K_\text{cam},\, T^\text{cam}_\text{world} \cdot T_{k,t} \cdot o_k\right)
$$

$$
S_t = \text{Rasterise}\!\left(\{(u_{k,t}, v_{k,t})\}_{k=1}^K,\; \mathcal{E}(\mathcal{M})\right)
$$

**含义**: 将机器人关节角 $q_t$ 通过正向运动学转为世界坐标，再投影到像素空间，最终光栅化为骨架图 $S_t$。

**符号说明**:
- $q_t$：$t$ 时刻的关节角向量
- $\mathcal{M}$：机器人运动学模型（来自 URDF）
- $K_\text{cam}$：相机内参矩阵
- $T^\text{cam}_\text{world}$：相机外参（世界到相机变换）
- $T_{k,t}$：第 $k$ 个关节在 $t$ 时刻的世界坐标变换
- $o_k$：第 $k$ 个关节的本体偏移
- $\mathcal{E}(\mathcal{M})$：运动学模型的拓扑边集
- $S_t \in \mathbb{R}^{H \times W \times 3}$：渲染的骨架图

### 公式3: [[MANO|人手骨架投影（MANO 扩展）]]

$$
S^{\text{human}}_t = \text{Rasterise}\!\left(\left\{\pi\!\left(K_\text{cam},\, T^\text{cam}_\text{world} \cdot T^{\text{MANO}}_{k,t} \cdot o^{\text{MANO}}_k\right)\right\}_k,\; \mathcal{E}(\mathcal{M}^{\text{MANO}})\right)
$$

**含义**: 对人类手部使用 MANO 模型拓扑，与机械臂完全相同的投影流程生成骨架图，实现统一条件编码。

**符号说明**:
- $T^{\text{MANO}}_{k,t}$：MANO 模型第 $k$ 个关节在 $t$ 时刻的变换
- $\mathcal{M}^{\text{MANO}}$：MANO 手部运动学模型

### 公式4: [[Frequency-Tempered Sampling|频率调和批次加权]]

$$
w_i \propto n_{\text{frames},i}^{1/T}, \quad T = 3
$$

**含义**: 按帧数的 $1/T$ 次幂对各数据集加权采样，温度 $T=3$ 平衡大小数据集的贡献。

**符号说明**:
- $w_i$：第 $i$ 个数据集的采样权重
- $n_{\text{frames},i}$：第 $i$ 个数据集的总帧数
- $T=3$：温度参数，控制平衡程度

---

## 关键图表

### Figure 1: OSCAR 作为策略评估代理

![Figure 1 左：OSCAR rollout vs 真实执行](https://arxiv.org/html/2606.04463v2/x1.png)

![Figure 1 右：七种策略成功率相关性](https://arxiv.org/html/2606.04463v2/x2.png)

**说明**: 左图对比 OSCAR 生成视频（上）与 π₀-FAST 策略真实执行（下）的三帧；右图展示 7 种通用策略在虚拟与真实评估中均值成功率的高度相关性（r = 0.750）。

### Figure 2: 方法总览

![Figure 2: OSCAR 三阶段架构](https://arxiv.org/html/2606.04463v2/x3.png)

**说明**: 三组件架构——条件编码（第一帧 + 骨架 VAE latent 求和）→ 条件注入（latent 空间相加送入 DiT）→ 视频生成（DiT 去噪 + VAE 解码）。

### Figure 3: 跨本体骨架覆盖（8 个训练源）

![Figure 3a - DROID](https://arxiv.org/html/2606.04463v2/x4.png)

![Figure 3b - RH20T-cfg5](https://arxiv.org/html/2606.04463v2/x5.png)

![Figure 3c - RH20T-cfg7](https://arxiv.org/html/2606.04463v2/x6.png)

![Figure 3d - InternData](https://arxiv.org/html/2606.04463v2/x7.png)

![Figure 3e - AgiBot G1](https://arxiv.org/html/2606.04463v2/x8.png)

![Figure 3f - AIROA-MoMa](https://arxiv.org/html/2606.04463v2/x9.png)

![Figure 3g - EgoDex](https://arxiv.org/html/2606.04463v2/x10.png)

![Figure 3h - EPIC-Kitchens](https://arxiv.org/html/2606.04463v2/x11.png)

**说明**: 同一骨架渲染流水线在 4 个机器人数据集（DROID、RH20T-cfg5、RH20T-cfg7、InternData）和 4 个人类/人形数据集（AgiBot G1、AIROA-MoMa、EgoDex、EPIC-Kitchens）上的覆盖，验证条件表示的跨本体统一性。

### Figure 4: 定性对比（vs 5 个基线）

![Figure 4 左：AgiBot G1 对比](https://arxiv.org/html/2606.04463v2/x12.png)

![Figure 4 右：DROID 对比](https://arxiv.org/html/2606.04463v2/x13.png)

**说明**: 在 AgiBot G1 和 DROID 本体上，OSCAR 相比 5 个基线（文本条件、潜在动作、Kinema4D、Genie Envisioner、EnerVerse-AC）生成了更准确的动作执行和更高视觉质量的视频。

### Figure 5: 附录 — 其余形态定性结果

![Figure 5a - AIROA-MoMa](https://arxiv.org/html/2606.04463v2/x14.png)

![Figure 5b - InternData](https://arxiv.org/html/2606.04463v2/x15.png)

![Figure 5c - RH20T-cfg5](https://arxiv.org/html/2606.04463v2/x16.png)

![Figure 5d - RH20T-cfg7](https://arxiv.org/html/2606.04463v2/x17.png)

**说明**: OSCAR 在 AIROA-MoMa（Toyota HSR）、InternData（KUKA iiwa 合成）、RH20T 两种配置上的定性结果，进一步验证跨本体泛化。

### Table 1: 数据集统计

| 来源 | 本体 | 原始片段 | 过滤后 |
|------|------|---------|--------|
| RH20T-cfg5 | Franka Panda | — | 1,261 |
| RH20T-cfg7 | Franka Panda | — | 2,044 |
| InternData-A1 | KUKA iiwa（合成） | — | 9,805 |
| DROID | Franka Panda | — | 21,904 |
| AgiBot World Beta | AgiBot G1 | — | 65,720 |
| AIROA-MoMa | Toyota HSR | — | 96 |
| EgoDex | Human hand | — | 78,273 |
| EPIC-Kitchens | Human hand | — | 7,554 |
| **合计** | — | **2,165,359** | **180,657** |

**说明**: 原始 2.16M 片段经四阶段过滤后仅保留约 8.3%，确保高质量训练数据。

### Table 2: 定量对比（主要结果）

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | tLPIPS↓ | FVD↓ | FID↓ |
|------|-------|-------|--------|---------|------|------|
| TesserAct | — | — | — | — | — | — |
| Cosmos-Predict2.5（文本） | — | — | — | — | — | — |
| IRASim | — | — | — | — | — | — |
| Genie Envisioner | 23.29 | 0.838 | 0.140 | 0.007 | 15.37 | 22.92 |
| EnerVerse-AC | 20.47 | 0.746 | 0.223 | 0.021 | 33.70 | 38.23 |
| **OSCAR（ours）** | **24.24** | **0.846** | **0.094** | **0.015** | **7.08** | **15.07** |

**说明**: OSCAR 在 PSNR、SSIM、LPIPS、FVD、FID 上均取得最佳，且参数量（2B）远小于 Kinema4D（14B）。

### Table 3: 消融——条件表示

| 条件方式 | PSNR↑ | SSIM↑ | LPIPS↓ | FVD↓ |
|----------|-------|-------|--------|------|
| 隐式动作（Latent Action） | 19.22 | — | — | — |
| Mesh 渲染 | 23.11 | — | — | 0.106 |
| **骨架渲染（canonical）** | **23.48** | — | — | **0.106** |

**说明**: 骨架渲染与 Mesh 渲染性能相当，但骨架渲染无需机器人特定 URDF 外观资产，可直接扩展至人类数据，泛化性更强。

### Table 4: 消融——数据策略

| 训练策略 | FVD↓ |
|----------|------|
| 仅机器人数据 | 7.69 |
| + 人类数据（从头训练） | 7.65 |
| + 人类数据（热启动） | **7.08** |

**说明**: 加入人类数据可改善所有指标；热启动（先在机器人数据上预训练）比从头联合训练更快收敛且效果更好。

### Table 5: 策略评估指标（RoboArena）

| 条件方式 | MMRV↓ | Pearson r↑ | Spearman ρ↑ | SISR_Δ↓ |
|----------|-------|-----------|------------|---------|
| 隐式动作（Latent Action） | 1.429 | 0.867 | 0.643 | — |
| Mesh 渲染 | 0.714 | 0.781 | 0.679 | — |
| **骨架渲染（OSCAR）** | **0.571** | **0.852** | **0.750** | **1.73pp** |

**说明**: 骨架渲染在策略排序相关性（Spearman ρ = 0.750）和相对排名误差（MMRV = 0.571）上均优于其他条件方式，确认其为更优的策略评估代理。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| DROID | 21,904 片段 | 86 任务，564 真实场景，Franka Panda | 训练 + 测试 |
| AgiBot World Beta | 65,720 片段 | 217 任务，AgiBot G1 | 训练 + 测试 |
| RH20T | 3,305 片段 | 接触丰富桌面操作，Franka Panda | 训练 + 测试 |
| InternData-A1 | 9,805 片段 | 合成，KUKA iiwa | 训练 + 测试 |
| EgoDex | 78,273 片段 | 194 个手部任务，第一视角 | 训练 |
| EPIC-Kitchens | 7,554 片段 | 无脚本烹饪，第一视角 | 训练 |
| RoboArena | 65 个测试 session | 7 种通用策略 | 策略评估 |

### 实现细节

- **Backbone**: [[Cosmos-Predict2.5]]-2B（Diffusion Transformer）
- **优化器**: AdamW，学习率 $3 \times 10^{-5}$
- **Batch Size**: 16
- **训练迭代**: 第一阶段 15k iter（仅机器人），第二阶段继续 + 人类数据
- **总训练迭代**: ~150k
- **硬件**: 单张 NVIDIA GH200 GPU
- **Guidance Scale**: $w = 6$（CFG）
- **采样策略**: 频率调和批次加权（$T=3$）

### 可视化结果

OSCAR 在 AgiBot G1 和 DROID 本体的定性样本上，运动轨迹与骨架条件高度一致，生成的视频背景和物体纹理保真度优于所有基线；对于 OOD 数据（Ego4D，训练时未见），仍能生成合理的手部运动视频，体现了骨架表示的跨本体泛化能力。

---

## 批判性思考

### 优点

1. **简洁有效的统一表示**: 2D 骨架投影轻量、可解释，且天然覆盖机械臂和人手，无需复杂的跨域对齐模块。
2. **高效的计算需求**: 2B 参数在单卡 GH200 上训练，成本远低于竞品（Kinema4D 使用 14B 参数且需多卡）。
3. **真实-虚拟高相关**: Pearson r = 0.750 的策略评估相关性在实用层面有意义，且 GPT-5 VLM 评判与人类评分 78% 一致。

### 局限性

1. **数据规模受限**: 骨架渲染依赖相机标定和运动学标注，可用数据集范围受此制约，难以大规模扩展。
2. **2B 参数瓶颈**: 当前模型规模可能限制更复杂场景的生成质量，扩展到更大模型需更多算力。
3. **依赖特定条件通道**: 推理时需要完整的关节角序列，而非所有策略都会暴露这些底层动作信号。

### 潜在改进方向

1. 扩展至更大模型（如 7B/14B）以验证 scaling 带来的提升。
2. 在无精确运动学标注时，利用视觉估计（pose estimation）自动生成骨架条件，降低数据依赖。
3. 探索更多人类第一视角数据（如 Ego4D）以进一步提升场景多样性。

### 可复现性评估

- [x] 代码开源（项目主页声明提供）
- [x] 预训练模型（项目主页声明提供）
- [x] 训练细节完整（论文附录详细描述）
- [x] 数据集可获取（使用公开数据集）

---

## 关联笔记

### 基于

- [[Cosmos-Predict2.5]]: 基础生成模型，使用其 2B DiT 架构
- [[Rectified Flow]]: 训练目标，用于视频生成

### 对比

- [[IRASim]]: 使用隐式动作嵌入的机器人视频世界模型
- [[TesserAct]]: 纯文本条件的机器人世界模型
- [[Kinema4D]]: 使用密集几何渲染的 14B 参数世界模型
- [[Genie Envisioner]]: 对比基线，Genie 系列的视频预测模型

### 方法相关

- [[Forward Kinematics]]: 骨架渲染的核心计算
- [[MANO]]: 人类手部运动学模型，OSCAR 对其的扩展
- [[Diffusion Transformer]]: OSCAR 使用的生成骨干架构
- [[Classifier-Free Guidance]]: 推理时使用的条件增强策略
- [[SigLIP]]: 语义去重中用于视觉嵌入的模型

### 硬件/数据相关

- [[RoboArena]]: 策略评估基准
- [[DROID]]: 主要训练数据集之一
- [[EgoDex]]: 人类手部第一视角训练数据集

---

## 速查卡片

> [!summary] OSCAR: Omni-Embodiment Action-Conditioned World Model for Robotics
> - **核心**: 以 2D 骨架渲染为统一条件表示，在 2B 参数模型上实现跨本体动作条件视频生成
> - **方法**: Cosmos-Predict2.5-2B + 骨架 latent 求和注入 + 7 源异构数据流水线
> - **结果**: PSNR 24.24 / FVD 7.08（SOTA），策略评估 Pearson r = 0.750
> - **代码**: https://wuzy2115.github.io/oscar-project-page/

---

*笔记创建时间: 2026-06-05*
