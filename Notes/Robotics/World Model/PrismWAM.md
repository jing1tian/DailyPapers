---
title: "PrismWAM: Decoupling State Representations in World Action Models"
method_name: "PrismWAM"
authors: [Tianyu Xin, Wanqi Zhong, Bowen Hu, Zhangchi Guo, Lijie Wang, Yiming Wang, Haoran Liao, Heyang Yu, Xiaoyu Tian, Zhiding Yu, Yiming Li]
year: 2026
venue: arXiv
tags: [world-action-model, depth-prediction, manipulation, flow-matching, mixture-of-transformers, decoupled-representation, robot-learning]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2610.26564v1
created: 2026-10-21
---

# 论文笔记：PrismWAM: Decoupling State Representations in World Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Tsinghua University; NVIDIA |
| 日期 | October 2026 |
| 项目主页 | [prism-wam.github.io](https://prism-wam.github.io/) |
| 对比基线 | [[OpenWAM]], [[FastWAM]], [[starVLA]], [[4D-WAM]], [[pi0.5]] |
| 链接 | [arXiv](https://arxiv.org/abs/2610.26564) |

---

## 一句话总结

> PrismWAM 将 [[World Action Model]] 的状态表示解耦为语义规划（RGB）和物理展开（深度），无需动作预训练即可超越有预训练的基线模型。

---

## 核心贡献

1. **状态解耦原则**: 提出将世界状态的语义外观（RGB）与物理配置（深度/位姿）分离，避免现有 WAM 将两者混为一体的弊端
2. **链式注意力架构（Chain Attention）**: 设计结构化的单向注意力掩码，实现 RGB → 深度 → 动作的有向信息流，防止信息短路
3. **两阶段训练策略**: 先用深度数据预训练深度 VAE 和 Backbone，再联合训练 RGB 稀疏目标、深度密集展开和动作生成，无需机器人动作预训练

---

## 问题背景

### 要解决的问题

现有 [[World Action Model]]（WAM）将世界状态表示为视频模型的潜在嵌入，这些嵌入把外观（纹理、颜色）和物理配置（占据、位姿）混合在一起，导致模型在随机化场景中的泛化能力受限。

### 现有方法的局限

- **[[RGB-based WAM]]**（如 FastWAM、OpenWAM-α）：在 RGB 空间进行世界状态展开，难以区分外观变化和物理运动，随机化场景下成功率急剧下降（FastWAM 在随机化场景仅 1.90%）
- **[[空间感知 WAM]]**（如 4D-WAM、Spatial-Forcing、X-WAM）：将几何信息与 RGB 并行输入，但仍在联合表示中学习，未真正解耦
- **动作预训练模型**（如 π0.5、OpenWAM-α）：依赖大规模机器人动作数据预训练，数据获取成本高

### 本文的动机

世界运动的本质是物理的，与外观无关。将语义规划（"做什么"，用 RGB 表达）与物理展开（"怎么变化"，用深度表达）分离，可以让物理预测更精准、更少受外观干扰，从而同时提升泛化性和精度控制能力。

---

## 方法详解

### 模型架构

PrismWAM 采用 **[[Mixture-of-Transformers]]（MoT）** 架构，基于 Wan2.2-5B 骨干网络：

- **输入**: 语言指令 $\ell$ + RGB 历史帧 $r_{\leq 0}$ + 当前深度帧 $d_0$ + 本体感知 $s$
- **Backbone**: [[Wan]] 视频生成模型（RGB 流和深度流共享视觉块参数 $B_V$）
- **核心模块**: [[Chain Attention]] 实现 RGB → Depth → Action 单向信息流
- **输出**: [[Action Chunking|动作块]] $a_{1:H}$（绝对关节角度指令）
- **总参数**: ~6B（PrismWAM）/ ~15B（PrismWAM∗，基于 Wan-2.1-14B）

### 核心模块

#### 模块1: RGB 语义规划流（Sparse Goal Supervision）

**设计动机**: 利用 [[预训练视频模型]] 的语义理解能力，为深度展开提供目标语义引导

**具体实现**:
- 输入当前 RGB 历史帧 $r_{\leq 0}$，通过冻结的 [[RGB VAE]] 编码为潜在表示
- 仅在**训练时**监督预测一帧目标 RGB 图像 $r_F$（稀疏监督）
- 推理时 RGB 流只处理已观测帧，不生成未来 RGB
- 在 MoT 中，RGB 流使用共享视觉块 $B_V$，但不读取深度或动作信息

#### 模块2: 深度物理展开流（Dense Physical Rollout）

**设计动机**: 在几何空间进行密集的精细预测，捕捉物体的物理运动轨迹，避免外观干扰

**具体实现**:
- 输入当前深度帧 $d_0$，通过预训练深度 VAE 编码（单通道，来自 Wan VAE 迁移）
- 密集监督预测未来 $F$ 帧深度序列 $d_{1:F}$
- 通过 [[Chain Attention]] 读取 RGB 流的观测特征（$h_{R,\leq 0}$），获取语义引导
- 推理时深度潜变量与动作联合去噪

#### 模块3: 链式注意力（Chain Attention）

**设计动机**: 控制信息流向，避免 RGB 外观特征直接污染动作预测，同时保留语义引导深度展开的能力

**具体实现**（结构化注意力掩码）:
- **RGB 流** → 只读自身观测 token 和上下文 $C$（语言、本体感知）
- **深度流** → 读 RGB 观测 token + 自身 token + 上下文 $C$
- **动作流** → 读深度（观测 + 未来）token + 自身 + 上下文 $C$
- **禁止连接**: RGB 不读深度/动作；深度不读动作；动作不直接读 RGB

### 两阶段训练

**Stage 1（深度适配）**: 用深度数据（AgiBotWorld-Beta 75% + RoboTwin 25%，约 2300 小时）预训练深度 VAE 和 Backbone，仅用深度损失

**Stage 2（联合训练）**: 同时训练 RGB 目标预测、深度序列预测、动作生成三个分支，三者损失联合优化

---

## 关键公式

### 公式1: [[Factorized Policy|因子化策略]]

$$
\pi_\theta(a_{1:H} \mid x_0, c) = \int p_\theta(a_{1:H} \mid x_{1:F}, x_0, c) \cdot p_\theta(x_{1:F} \mid x_0, c) \, dx_{1:F}
$$

**含义**: 将动作生成策略分解为"给定未来状态轨迹生成动作"和"预测未来状态轨迹"两个子问题的乘积

**符号说明**:
- $x_0$: 当前世界状态
- $x_{1:F}$: 未来状态轨迹（中间变量）
- $a_{1:H}$: 动作序列
- $c = (\ell, s)$: 任务上下文（语言指令 + 本体感知）

### 公式2: [[Decoupled Policy|解耦策略]]

$$
\pi_\theta(a_{1:H} \mid r_{\leq 0}, d_0, c) = \iint p_\theta(a_{1:H} \mid z_D, c) \cdot p_\theta(z_D \mid d_0, z_R, c) \cdot p_\theta(z_R \mid r_{\leq 0}, c) \, dz_R \, dz_D
$$

**含义**: 将状态表示进一步解耦为语义规划特征 $z_R$（来自 RGB）和物理展开特征 $z_D$（来自深度），形成嵌套积分的三阶段推断链

**符号说明**:
- $r_{\leq 0}$: 当前及历史 RGB 帧
- $d_0$: 当前深度帧
- $z_R$: RGB 流产生的语义规划潜变量
- $z_D$: 深度流产生的物理展开潜变量

### 公式3: [[Mixture-of-Transformers|MoT 特征更新]]

$$
h_R^{(j+1)} = B_V^{(j)}(h_R^{(j)} \mid C, \sigma_R)
$$

$$
h_D^{(j+1)} = B_V^{(j)}(h_D^{(j)} \mid h_{R,\leq 0}^{(j)}; C, \sigma_D)
$$

$$
h_A^{(j+1)} = B_A^{(j)}(h_A^{(j)} \mid h_D^{(j)}; C, \sigma_D, \sigma_A)
$$

**含义**: 第 $j$ 层的三个流的特征更新规则，体现链式信息流：RGB → Depth → Action

**符号说明**:
- $B_V^{(j)}$: 第 $j$ 层共享视觉块（RGB 和深度流共享参数）
- $B_A^{(j)}$: 第 $j$ 层动作块（独立参数）
- $\sigma_R, \sigma_D, \sigma_A$: 各模态的 [[Flow Matching]] 噪声水平
- $C$: 上下文（语言 + 本体感知投影）

### 公式4: [[Flow Matching|流匹配噪声扰动]]

$$
\tilde{X}_m = (1 - \sigma_m) X_m + \sigma_m \varepsilon_m, \quad \varepsilon_m \sim \mathcal{N}(0, I)
$$

目标速度场：

$$
v_m = \varepsilon_m - X_m
$$

**含义**: 对每个模态 $m$（RGB、深度、动作）独立采样噪声水平 $\sigma_m$，线性插值生成带噪输入，训练模型预测去噪方向

**符号说明**:
- $X_m$: 模态 $m$ 的干净目标
- $\sigma_m \in [0,1]$: 噪声水平（各模态独立采样）
- $\varepsilon_m$: 标准高斯噪声

### 公式5: [[Joint Training Objective|联合训练目标]]

$$
\mathcal{L} = \mathcal{L}_R + \mathcal{L}_D + \mathcal{L}_A
$$

其中每个模态损失为：

$$
\mathcal{L}_m = \mathbb{E}_{\sigma_m}\left[w(\sigma_m) \cdot \left\| \hat{v}_m - v_m \right\|^2 \right]
$$

**含义**: 三个模态的噪声加权 MSE 损失之和，各模态独立加权

**符号说明**:
- $\hat{v}_m$: 模型预测的速度场
- $v_m$: 目标速度场（$\varepsilon_m - X_m$）
- $w(\sigma_m)$: 噪声权重函数（见公式6）

### 公式6: [[Noise Weighting|噪声权重函数]]

$$
w(\sigma) = \frac{y(\sigma) - y_{\min}}{\bar{y} - y_{\min} + 10^{-10}}, \quad y(\sigma) = \exp\left[-2\left(\sigma - \frac{1}{2}\right)^2\right]
$$

**含义**: 对中间噪声水平（$\sigma \approx 0.5$）给予更大权重，抑制极端噪声水平下的不稳定梯度

**符号说明**:
- $y(\sigma)$: 以 $\sigma=0.5$ 为峰值的高斯权重
- $y_{\min}$: 在 1000 个均匀网格点上取得的最小值
- $\bar{y}$: 对应均值

---

## 关键图表

### Figure 1: 整体架构与链式注意力

![Figure 1](https://arxiv.org/html/2610.26564v1/method_architecture_and_attention.png)

**说明**: 左图：PrismWAM 的整体框架，RGB 流（稀疏目标监督）和深度流（密集展开监督）并行，通过动作塔输出动作块。右图：[[Chain Attention]] 的具体注意力路由，展示 RGB → Depth → Action 的单向信息流。

### Figure 2: 跨基准性能对比

（HTML 中无可访问图片 URL，见原文）

**说明**: PrismWAM 和 PrismWAM∗ 在多个基准（RoboTwin-C2R、LIBERO、LIBERO-Plus、LIBERO-Pro、RoboDojo）上与基线对比，显示在无动作预训练情况下仍能达到或超越有预训练基线的性能。

### Figure 3: 深度噪声鲁棒性测试

![Figure 3](https://arxiv.org/html/2610.26564v1/depth_corruption.png)

**说明**: 对三种深度传感器噪声（Metric Jitter、Edge Missing、Patch Hole）分 4 个强度级别进行压力测试（基于 Orbbec Gemini 335 和 RealSense D435 传感器规格）。最强噪声级别下 RoboTwin 随机化场景平均成功率仅下降 2.45 个百分点，说明深度流具有较强的噪声鲁棒性。

### Figure 4: RoboDojo 得分与成功率

（HTML 中无可访问图片 URL，见原文）

**说明**: PrismWAM（25.10 分）和 PrismWAM∗（36.23 分）在 RoboDojo 综合得分上显著领先 OpenWAM-α（17.18 分），尤其在精度控制类别（Precision: 19.50% vs 9.25% SR）上优势明显。

### Figure 5: 真实世界实验评估

![Figure 5](https://arxiv.org/html/2610.26564v1/realworld.png)

**说明**: 三个真实机械臂任务（Pick Up Plate 双臂、Ranking Block Size 单臂、Open Microwave 双臂多步骤）在分布内和分布外条件下的轨迹与成功率对比，PrismWAM 综合 OOD 成功率比 OpenWAM 高 20 个百分点。

### Table 1: RoboTwin-C2R 和 RoboDojo 主要结果

| 方法 | 动作预训练 | RT 干净 | RT 随机 | RT 平均 | RoboDojo 得分 | RoboDojo SR |
|------|-----------|--------|--------|--------|--------------|------------|
| π0.5 | ✓ | 70.70 | 46.00 | 58.35 | 11.44 | 6.93% |
| OpenWAM-α | ✓ | 89.40 | 48.70 | 69.05 | 17.18 | 11.92% |
| starVLA | ✗ | 46.52 | 3.16 | 24.84 | 6.40 | 3.24% |
| FastWAM | ✗ | 77.80 | 1.90 | 39.85 | 3.48 | 2.03% |
| X-WAM | ✗ | 70.00 | 25.80 | 47.90 | 7.69 | 3.83% |
| Spatial-Forcing | ✗ | 77.20 | 26.74 | 51.97 | 12.38 | 8.04% |
| 4D-WAM | ✗ | 81.50 | 41.80 | 61.65 | 12.71 | 7.97% |
| **PrismWAM** | **✗** | **87.98** | **53.16** | **70.57** | **25.10** | **18.55%** |
| **PrismWAM∗** | **✗** | **91.84** | **75.50** | **83.67** | **36.23** | **29.30%** |

**关键发现**: PrismWAM（无动作预训练）在 RoboTwin 平均成功率上超过 OpenWAM-α（有动作预训练），随机化场景差距尤其显著（53.16% vs 48.70%）。RoboDojo 得分更是大幅领先所有基线。

### Table 2: LIBERO、LIBERO-Plus、LIBERO-Pro 结果

| 方法 | LIBERO 总体 | LIBERO-Plus 布局 | LIBERO-Pro 总体 |
|------|------------|-----------------|----------------|
| π0.5 | 96.85 | — | 57.16 |
| OpenWAM-α | 99.30 | 77.38 | 55.71 |
| starVLA | 96.60 | 80.95 | 47.43 |
| FastWAM | 97.60 | 66.49 | 40.51 |
| 4D-WAM | 98.60 | 79.21 | 48.34 |
| **PrismWAM** | **99.35** | **81.70** | **54.48** |

**关键发现**: PrismWAM 在 LIBERO 分布内任务取得最高成功率（99.35%），在 LIBERO-Plus 布局扰动分布外测试也最优（81.70%），展示了深度表示对泛化能力的贡献。

### Table 3: 消融实验（RoboTwin 50 任务）

| 变体 | 干净 | 随机 | 平均 | 说明 |
|------|------|------|------|------|
| **PrismWAM（完整）** | 87.40 | 54.10 | 70.75 | — |
| w/o RGB 流 | 79.10 | 40.10 | 59.60 | −11.15，语义引导缺失 |
| w/o 深度流 | 83.00 | 2.40 | 42.70 | −28.05，物理展开缺失，随机化场景崩溃 |
| 观测 RGB → 动作（直连）| 89.10 | 40.30 | 64.70 | 干净提升但泛化下降 |
| 未来 RGB → 深度（暴露目标）| 87.00 | 50.00 | 68.50 | 轻微下降，信息污染 |
| w/o 未来 RGB 监督 | 81.60 | 51.40 | 66.50 | −4.25，语义引导变弱 |
| w/o 深度预热阶段 | 89.30 | 24.70 | 57.00 | −13.75，物理展开质量下降 |

**关键发现**: 深度流的贡献最大（去掉后随机化场景仅 2.40%），两阶段训练策略对深度质量至关重要（−13.75 均值），Chain Attention 防止 RGB 直接访问动作对泛化至关重要。

### Table 4: 真实世界实验结果

| 任务 | OpenWAM ID | OpenWAM OOD | PrismWAM ID | PrismWAM OOD |
|------|-----------|-------------|------------|-------------|
| Pick Up Plate（双臂）| 4/5 (80.0%) | 10/15 (66.7%) | 5/5 (100%) | 13/15 (86.7%) |
| Ranking Block Size（单臂）| 7/12 (58.3%) | 3/10 (30.0%) | 11/12 (91.7%) | 6/10 (60.0%) |
| Open Microwave（双臂多步）| 9/10 (90.0%) | 4/10 (40.0%) | 8/10 (80.0%) | 5/10 (50.0%) |
| **汇总** | **74.1%** | **48.6%** | **88.9%** | **68.6%** |

**关键发现**: PrismWAM 综合 ID 成功率比 OpenWAM 高 14.8 个点，OOD 高 20 个点。Open Microwave 任务（多步骤）OpenWAM 略优，说明 PrismWAM 在复杂长程任务仍有改进空间。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| AgiBotWorld-Beta | ~75% 权重 | 大规模深度视频 | 深度 VAE + Stage 1 预训练 |
| RoboTwin | ~25% 权重，约 2300h 总计 | 机器人操作深度序列 | 深度预训练 |
| RoboTwin-C2R | 50 任务，干净/随机化场景 | 操作场景泛化测试 | 主要基准 |
| LIBERO | 分布内 4 套件 | 标准操作任务 | 分布内评测 |
| LIBERO-Plus | OOD 视角/布局扰动 | 泛化测试 | 分布外评测 |
| LIBERO-Pro | OOD 物体属性变化 | 泛化测试 | 分布外评测 |
| RoboDojo | 精度/泛化综合 | 多维度评测 | 综合基准 |

### 实现细节

- **Backbone**: [[Wan]]2.2-5B（PrismWAM）/ Wan-2.1-14B（PrismWAM∗）
- **优化器**: AdamW，$\beta=(0.9, 0.95)$
- **学习率**: $10^{-4}$ → $10^{-6}$ 余弦衰减
- **Flow Matching 时间步**: $1000\sigma$，其中 $\sigma = \frac{5u}{1+4u}$，$u \sim \mathcal{U}(0,1)$
- **深度 VAE**: 单通道，48/16 潜在通道，损失 = L1 + L2 + KL
- **真实世界硬件**: Piper-X 机械臂 + Orbbec Gemini 335 RGB-D 相机
- **单任务演示数**: 100–200 条

---

## 批判性思考

### 优点

1. **解耦原则清晰**: 语义（RGB）与物理（深度）分离的设计思路有明确的理论动机，消融实验验证充分
2. **无需动作预训练**: 仅用约 2300 小时深度数据即可超越需要大规模机器人动作预训练的模型（OpenWAM-α）
3. **泛化能力突出**: 随机化场景（53.16% vs 1.90%/3.16%）和真实世界 OOD（+20 点）的提升尤其显著
4. **模块设计合理**: Chain Attention 的单向路由避免了信息污染，消融证明每个方向都重要

### 局限性

1. **深度数据依赖**: 需要同步采集 RGB-D 数据，真实世界大规模深度数据集匮乏，限制了预训练规模
2. **复杂长程任务**: Open Microwave（多步骤）上略逊于 OpenWAM，说明深度展开对长时序依赖的支持仍需改进
3. **深度腐蚀分析局限**: 压力测试基于传感器规格设计，并非真实标定的传感器误差模型
4. **Figure 2、4 无图片 URL**: HTML 版本部分图表无法直接获取图片链接

### 潜在改进方向

1. 将深度扩展到其他物理表示（占据、光流、触觉、本体感知）
2. 在真实世界收集大规模 RGB-D 操作数据，扩展预训练规模
3. 引入记忆机制增强长程任务的状态追踪能力

### 可复现性评估

- [ ] 代码开源（项目主页存在但代码暂未开放）
- [ ] 预训练模型（未公开）
- [x] 训练细节完整（论文附录详细描述）
- [x] 数据集可获取（RoboTwin 公开；AgiBotWorld-Beta 可访问）

---

## 关联笔记

### 基于

- [[Wan]]: 基础视频生成模型骨干（RGB VAE + Transformer）
- [[Flow Matching]]: 生成建模框架，用于联合去噪
- [[Mixture-of-Transformers]]: 多流并行 Transformer 架构

### 对比

- [[OpenWAM]]: 最强有动作预训练基线，PrismWAM 在随机化场景上超越
- [[FastWAM]]: 无预训练基线，RGB 空间展开，随机化场景仅 1.90%
- [[4D-WAM]]: 空间感知 WAM，联合 RGB+几何但未解耦
- [[starVLA]]: VLA 基线，无预训练
- [[pi0.5]]: 大规模动作预训练 VLA，PrismWAM 在 RoboDojo 上大幅超越

### 方法相关

- [[World Action Model]]: 核心方法类别
- [[Chain Attention]]: 本文核心注意力机制
- [[Depth Prediction]]: 物理状态表示的核心手段
- [[Decoupled Representation]]: 本文核心设计原则

### 硬件/数据相关

- [[Orbbec Gemini 335]]: 真实世界实验使用的 RGB-D 传感器
- [[AgiBotWorld-Beta]]: 深度预训练主要数据源

---

## 速查卡片

> [!summary] PrismWAM: Decoupling State Representations in World Action Models
> - **核心**: 将 WAM 的状态解耦为语义（RGB，稀疏目标）和物理（深度，密集展开），通过 Chain Attention 实现单向信息流
> - **方法**: MoT 架构（基于 Wan），两阶段训练（深度预热 + 联合训练），无动作预训练
> - **结果**: RoboTwin 随机 53.16%（vs FastWAM 1.90%），RoboDojo 25.10 分（vs OpenWAM 17.18），真实 OOD +20 点
> - **代码**: [prism-wam.github.io](https://prism-wam.github.io/)

---

*笔记创建时间: 2026-10-21*
