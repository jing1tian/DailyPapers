---
title: "Point Tracking Improves World Action Models"
method_name: "JOPAT"
authors: [Jiarui Guan, Wenshuai Zhao, Yue Pei, Ziliang Chen, Arno Solin, Juho Kannala]
year: 2026
venue: arXiv
tags: [world-action-model, point-tracking, diffusion-transformer, robot-manipulation, action-free-pretraining]
zotero_collection: Robotics/World Model
image_source: mixed
arxiv_html: https://arxiv.org/html/2605.23856
created: 2026-05-26
---

# 论文笔记：Point Tracking Improves World Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Aalto University, ELLIS Institute Finland, University of Oulu, Sun Yat-sen University, Peng Cheng Laboratory, Beihang University |
| 日期 | May 2026 |
| 项目主页 | 暂无 |
| 对比基线 | [[World Action Model\|UWM]]、[[扩散策略\|ACT/Diffusion Policy]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.23856) |

---

## 一句话总结

> JOPAT 通过在 [[World Action Model]] 中联合生成 2D 点轨迹（含可见性）与像素潜变量和动作，显著提升了机器人在遮挡、物体交互和超视野运动等复杂任务中的长视野操作性能。

---

## 核心贡献

1. **联合像素-轨迹-动作生成框架**: 提出 JOPAT，在单一 [[Diffusion Transformer (DiT)|扩散 Transformer]] 中同步去噪未来视觉潜变量、2D 点轨迹（含可见性）和机器人动作，三模态之间通过 self-attention 相互约束。
2. **轨迹即视频编码（Track-as-Video）**: 将空间点网格重整为时空体素，利用 [[3D 卷积]] 进行 patchification，实现紧凑的轨迹 token 表示，无需额外预训练的轨迹编码器。
3. **动作无关预训练可迁移**: JOPAT 的视觉-轨迹分支可从机器人域（DROID）或通用视频（OpenVid-1M）预训练，在低数据场景下带来显著收益，验证了显式运动先验的通用性。

---

## 问题背景

### 要解决的问题

[[World Action Model]]（WAM）将视频预测与动作生成耦合在一起，但现有像素级视觉预测对任务无关的视觉变化（光照、背景、遮挡）脆弱，难以捕获长时序中的运动一致性。在遮挡、物体交互或超视野运动等场景下，像素外观模型无法可靠编码物体去向。

### 现有方法的局限

- **纯像素 WAM**（如 UWM）：视觉 token 需隐式编码运动，对外观变化敏感；长视野任务误差累积严重。
- **光流 / 中间运动表示**：通常作为策略输入或辅助目标，而非作为生成未来状态的一部分与动作共同采样。
- **动作无关预训练研究**：发现单纯像素级预训练并不总能迁移到操作任务，运动层面的显式监督更有效。

### 本文的动机

显式 2D 点轨迹（含可见性）提供了运动的对应级约束，对遮挡和超视野具有天然鲁棒性；将轨迹作为联合生成状态的一部分，而非后处理，使动作与运动在 attention 中直接交互，理论上更有助于长视野规划。

---

## 方法详解

### 模型架构

JOPAT 采用 **[[Diffusion Transformer (DiT)|DiT]] 风格的统一扩散架构**：

- **输入**: 语言指令 + 当前两帧 RGB 观测 $o_{t-1:t}$
- **Conditioning**: 观测编码器 $E_c$ 生成全局特征 $\mathbf{c}_t = E_c(o_{t-1:t})$，通过 [[AdaLN]] 注入每个 DiT 块（各模态独立时间步 $\tau_a, \tau_o, \tau_p$）
- **核心模块**: [[CoTracker3]] 提供离线轨迹监督；Track-as-Video 编码器 $E_p$ 将点轨迹 token 化
- **输出**: 动作块 $\hat{a}_{t:t+K-1}$（主要），未来视觉潜变量与点轨迹（作为内部状态）
- **总参数**: DiT 12 层，隐藏维度 768，12 注意力头，8 个可学习 register token

### 核心模块

#### 模块 1：联合序列去噪

**设计动机**: 利用 [[Self-Attention]] 让动作、视觉、轨迹三模态 token 在共享序列中互相约束，避免各自独立预测的信息孤立。

**具体实现**:

联合序列构成如下：

$$\mathbf{Z} = [\tilde{\mathbf{z}}^{a}_{t:t+K-1},\ \tilde{\mathbf{z}}^{o}_{t+H:t+H+1},\ \tilde{\mathbf{z}}^{p}_{t:t+H_p-1},\ \mathbf{r}]$$

其中 $\mathbf{r}$ 为 8 个可学习 register token，$H=16$ 为未来观测偏移，$K=H_p=19$ 为动作和轨迹预测视野。

DiT 对整个序列做全局双向注意力，同时预测三模态噪声：

$$\hat{\boldsymbol{\epsilon}}^{a},\hat{\boldsymbol{\epsilon}}^{o},\hat{\boldsymbol{\epsilon}}^{p} = T_\theta(\mathbf{Z};\ \mathbf{c}_t,\ \tau_a,\ \tau_o,\ \tau_p)$$

#### 模块 2：轨迹构建与编码（Track-as-Video）

**设计动机**: 将 2D 点轨迹视为时空视频，利用 [[3D 卷积]] patchification 得到紧凑 token，而非逐点 MLP 编码，保留空间邻域结构。

**具体实现**:

- 在参考帧上初始化 $N=625$ 个均匀格点查询点（$25\times25$ 网格）
- 用 [[CoTracker3]] 前向追踪得到 $\mathbf{P}^{(t)} \in \mathbb{R}^{H_p \times N \times 2}$，可见性标签 $\mathbf{V}^{(t)} \in \{0,1\}^{H_p \times N}$
- 将点维度重整为空间网格：$\mathbf{G}^p \in \mathbb{R}^{2 \times H_p \times H_g \times W_g}$（$H_g W_g = N$）
- 3D 卷积 patchify（时空 patch 大小 $(2,5,5)$）：$\mathbf{z}^p = E_p(\mathbf{P}^{(t)}) = \text{Patchify}_{3D}(\mathbf{G}^p) \in \mathbb{R}^{L_p \times d}$
- 去噪后分别解码坐标和可见性：$\hat{\mathbf{P}}^{(t)} = D_p(\hat{\mathbf{z}}^p)$，$\hat{\mathbf{V}}^{(t)} = D_v(\hat{\mathbf{z}}^p)$
- 可见性**仅作为输出**（不输入编码器），避免去噪过程中的真值泄露

---

## 关键公式

### 公式 1：[[World Action Model|WAM 基础架构]]（像素-潜变量版本）

$$
[\hat{\boldsymbol{\epsilon}}^{a}, \hat{\boldsymbol{\epsilon}}^{o}] = T^{\mathrm{WAM}}_\theta([\tilde{\mathbf{z}}^{a}, \tilde{\mathbf{z}}^{o}, \mathbf{r}];\ \mathbf{c}_t,\ \tau_a,\ \tau_o)
$$

**含义**: 原始 WAM 仅联合去噪动作 token 和视觉潜变量 token，JOPAT 在此基础上新增轨迹分支。

**符号说明**:
- $\hat{\boldsymbol{\epsilon}}^{a}$: 预测的动作噪声
- $\hat{\boldsymbol{\epsilon}}^{o}$: 预测的观测潜变量噪声
- $\tilde{\mathbf{z}}^{a}, \tilde{\mathbf{z}}^{o}$: 含噪动作 / 视觉 token
- $\mathbf{r}$: 可学习 register token
- $\mathbf{c}_t$: 条件特征（来自观测编码器）
- $\tau_a, \tau_o$: 各模态独立的扩散时间步

### 公式 2：[[点轨迹追踪|轨迹初始化]]

$$
\mathbf{P}^{(t)} \in \mathbb{R}^{H_p \times N \times 2}, \qquad \mathbf{V}^{(t)} \in \{0,1\}^{H_p \times N}
$$

**含义**: 定义轨迹张量形状——$H_p$ 个未来时间步，$N$ 个追踪点，每点 2D 坐标（$x,y$）；可见性为二值标签。

**符号说明**:
- $H_p$: 轨迹预测视野（19 步）
- $N$: 查询点数量（625，$25\times25$ 网格）
- 上标 $(t)$: 以当前帧为参考锚点

### 公式 3：[[3D 卷积|Track-as-Video 编码]]

$$
\mathbf{P}^{(t)} \rightarrow \mathbf{G}^p \in \mathbb{R}^{2 \times H_p \times H_g \times W_g}, \qquad H_g W_g = N
$$

$$
\mathbf{z}^p = E_p(\mathbf{P}^{(t)}) = \mathrm{Patchify}_{3D}(\mathbf{G}^p), \qquad \mathbf{z}^p \in \mathbb{R}^{L_p \times d}
$$

**含义**: 将点轨迹重整为时空网格后做 3D 卷积 patch 化，得到紧凑 token 序列以输入 DiT。

**符号说明**:
- $\mathbf{G}^p$: 重整后的时空点坐标网格
- $H_g, W_g$: 网格空间维度（$H_g = W_g = 25$）
- $E_p$: 3D 卷积 patchifier 编码器
- $L_p$: patch 数量
- $d$: 隐藏维度（768）

### 公式 4：[[扩散模型|统一扩散去噪目标]]

$$
\mathbf{x}^m_{\tau_m} = \sqrt{\bar{\alpha}_{\tau_m}}\,\mathbf{x}^m_0 + \sqrt{1-\bar{\alpha}_{\tau_m}}\,\boldsymbol{\epsilon}^m, \qquad \mathcal{L}_m = \left\|\hat{\boldsymbol{\epsilon}}^m - \boldsymbol{\epsilon}^m\right\|_2^2
$$

**含义**: 对每个模态 $m \in \{a, o, p\}$ 独立采样扩散时间步，标准加噪后计算噪声预测 MSE 损失。

**符号说明**:
- $m \in \{a, o, p\}$: 动作、观测、点轨迹三模态
- $\mathbf{x}^m_0$: 干净目标（动作序列 / 视觉潜变量 / 轨迹坐标）
- $\bar{\alpha}_{\tau_m}$: 累积噪声调度系数
- $\boldsymbol{\epsilon}^m \sim \mathcal{N}(0, I)$: 高斯噪声
- $\hat{\boldsymbol{\epsilon}}^m$: 模型预测的噪声

### 公式 5：[[二元交叉熵|可见性监督损失]]

$$
\mathcal{L}_{\mathrm{vis}} = \frac{1}{H_p N} \sum_{\tau=0}^{H_p-1} \sum_{i=1}^{N} \mathrm{BCE}(\hat{\mathbf{V}}_{t+\tau,i},\ \mathbf{V}_{t+\tau,i})
$$

**含义**: 对每个点、每个未来时间步的可见性做二分类监督，处理遮挡和超视野情况。

**符号说明**:
- $\hat{\mathbf{V}}_{t+\tau,i}$: 模型预测的第 $i$ 点在时刻 $t+\tau$ 的可见性 logit
- $\mathbf{V}_{t+\tau,i}$: 由 [[CoTracker3]] 生成的真值可见性标签
- $\mathrm{BCE}$: 二元交叉熵

### 公式 6：[[模仿学习|有动作标注训练目标]]

$$
\mathcal{L}_{\mathcal{D}} = \mathcal{L}_a + \lambda_o \mathcal{L}_o + \lambda_p \mathcal{L}_p + \lambda_{\mathrm{vis}} \mathcal{L}_{\mathrm{vis}}
$$

**含义**: 对有动作标注的示教数据，同时监督三模态去噪和可见性分类，各权重均设为 1。

**符号说明**:
- $\mathcal{L}_a$: 动作去噪损失
- $\mathcal{L}_o$: 视觉潜变量去噪损失
- $\mathcal{L}_p$: 轨迹坐标去噪损失
- $\lambda_o, \lambda_p, \lambda_{\mathrm{vis}}$: 损失权重（均=1）

### 公式 7：[[行为克隆|动作无关视频训练目标]]

$$
\mathcal{L}_{\mathcal{V}} = \lambda_o \mathcal{L}_o + \lambda_p \mathcal{L}_p + \lambda_{\mathrm{vis}} \mathcal{L}_{\mathrm{vis}}
$$

**含义**: 对无动作标注的视频，屏蔽动作分支，仅监督视觉-轨迹动力学，实现无动作预训练。

---

## 关键图表

### Figure 1：JOPAT 点轨迹预测示意

![[JOPAT_fig1.png|600]]

**说明**: 从参考帧的查询点出发，JOPAT 预测跨越遮挡的长视野 2D 轨迹和可见性 logit。蓝色轨迹线展示了在物体遮挡时仍能追踪的关键点。

### Figure 2：JOPAT 模型总览

![[JOPAT_fig2.png|600]]

**说明**: (a) 滑动窗口轨迹构建——以当前帧为参考图像初始化格点查询；(b) JOPAT 在共享 [[Diffusion Transformer (DiT)|DiT]] 中联合去噪未来视觉潜变量、点轨迹坐标和机器人动作；(c) Track-as-Video 编码器将点轨迹重整为时空网格并做 3D 卷积 patchification，同时预测坐标噪声和可见性 logit。

### Figure 3：真实机器人任务场景

**说明**: LeRobot SO-101 平台上的四个真实任务：Insert-Peg（插销）、Cook-Soup（煮汤）、Push-Tomato（推番茄）、Pick-Grocery（抓取杂货）。展示了 JOPAT 需要应对遮挡和精细操作的实际场景。

### Figure 4：动作无关预训练消融

![[JOPAT_fig3.png|600]]

**说明**: 比较有无 DROID 动作无关视频预训练的平均真实机器人成功率。预训练带来显著提升，在 Pick-Grocery 等遮挡任务上收益最大。

### Figure 5：预测视野敏感性分析

![[JOPAT_fig4.png|600]]

**说明**: 不同未来观测偏移 $H$ 下的平均真实机器人成功率。$H=16$ 为最优配置，过短（无预见）或过长（过度外推）均导致性能下降。

### Table 1：LIBERO Benchmark 结果

| Suite | JOPAT | 排名 |
|-------|-------|------|
| Spatial | 97.2% | 4 |
| Object | 98.9% | **1** |
| Goal | 98.4% | **1** |
| Long | **96.4%** | **1** |
| **Average** | **97.8%** | **1** |

**说明**: JOPAT 在 40 个 LIBERO 任务上取得 97.8% 平均成功率，在长视野任务（LIBERO-Long）上提升最为显著，体现了显式轨迹建模对长时序任务的核心价值。

### Table 2：真实机器人 LeRobot 对比结果

| Task | JOPAT | UWM | ACT |
|------|-------|-----|-----|
| Cook-Soup | **60%** | 10% | 40% |
| Insert-Peg | **10%** | 0% | 0% |
| Push-Tomato | **100%** | 80% | 70% |
| Pick-Grocery | **60%** | 40% | 50% |
| **Average** | **57.5%** | 32.5% | 40% |

**说明**: JOPAT 在所有四个真实任务上均超越 UWM 和 ACT，平均成功率 57.5% vs UWM 的 32.5%，尤其在 Cook-Soup（涉及遮挡）上优势巨大（60% vs 10%）。

### Table 3：单模态 vs 联合建模消融（LIBERO-Long）

| 配置 | 平均成功率 | 说明 |
|------|-----------|------|
| 仅视觉潜变量（Latent-only） | 77.4% | 纯像素 WAM |
| 仅点轨迹（Track-only） | 26.2% | 无视觉语义锚 |
| **联合（JOPAT）** | **96.4%** | 轨迹 + 视觉互补 |

**关键发现**: 两模态具有互补性——仅轨迹时缺乏语义信息，仅视觉时缺乏运动一致性约束，联合建模大幅超过单独任一模态。

### Table 4：可见性预测影响（真实机器人）

| 配置 | 平均成功率 |
|------|-----------|
| 无可见性预测 | 47.5% |
| 有可见性预测 | **57.5%** |

**关键发现**: 显式可见性监督在遮挡和超视野运动场景下至关重要，贡献约 10 个百分点。

### Table 5：动作无关预训练消融（LIBERO-Long, 50 demos）

| 预训练数据 | 平均成功率 |
|-----------|-----------|
| 无预训练 | 66.1% |
| DROID（机器人域） | **96.4%** |
| OpenVid-1M（通用视频） | 95.1% |

**关键发现**: 从机器人域和通用视频预训练均有效；低数据场景下预训练收益最显著（+30 个百分点）。

### Table 6：OOD 鲁棒性（修改版 LIBERO-Long）

| 方法 | 平均成功率 |
|------|-----------|
| JOPAT | **0.66** |
| π₀.₅ | 0.53 |
| UWM | 0.34 |
| Diffusion Policy | 0.32 |

**说明**: 测试设置包含扩大物体范围（+5cm）、未见物体、干扰物等 OOD 变化，JOPAT 在所有配置下均最优。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| LIBERO | 40 任务，各 50 demo | 模拟基准，含 Spatial/Object/Goal/Long 4 子集 | 主要训练/测试 |
| LeRobot SO-101 | 4 任务，各约 50 demo | 真实机器人，含遮挡和精细操作 | 真实环境测试 |
| DROID | 大规模机器人视频 | 无动作标注，机器人域 | 动作无关预训练 |
| OpenVid-1M | 百万级通用视频 | 无动作标注，非机器人域 | 动作无关预训练 |

### 实现细节

- **Backbone**: 12 层 DiT，768 隐藏维度，12 注意力头
- **优化器**: AdamW，学习率 $1\times10^{-4}$，权重衰减 $1\times10^{-6}$
- **Batch Size**: 预训练 144，微调 72
- **扩散步数**: 训练 100 步，推理 DDIM 10 步
- **观测输入**: 2 帧 RGB，分辨率 224×224
- **未来观测偏移**: $H=16$ 步
- **动作/轨迹视野**: $K=H_p=19$ 步，执行前 8 步后重规划
- **点追踪**: 25×25 格点，$N=625$，由 [[CoTracker3]] 离线标注
- **硬件**: 预训练 4×NVIDIA H200 GPU（约 5 天），微调 1×H200（约 1 天），推理 RTX 4090 约 10 Hz

---

## 批判性思考

### 优点

1. **互补性验证充分**: 通过严格消融证明了轨迹与视觉潜变量的互补性（77.4% vs 26.2% vs 96.4%），逻辑清晰。
2. **可见性建模创新**: 将可见性仅作为输出而非输入的设计避免了真值泄露，技术细节严谨。
3. **通用视频预训练可行**: 证明无动作标注的通用视频（OpenVid-1M）也能有效迁移运动先验，扩大了预训练数据来源。
4. **真实机器人验证充足**: 在 LeRobot SO-101 平台上做了实际部署，成功率提升显著（57.5% vs 32.5%）。

### 局限性

1. **格点追踪粒度不足**: 基于格点的稀疏轨迹可能遗漏亚厘米级形变，限制了对精细灵巧操作（如毫米级插销）的支持（Insert-Peg 仅 10%）。
2. **依赖外部追踪器**: 训练时需要 [[CoTracker3]] 离线标注，追踪质量上限制约了轨迹监督质量。
3. **静态相机假设**: 当前设计仅适用于固定相机，无法直接推广到移动平台。
4. **推理速度限制**: 约 10 Hz，实时性有限；对延迟敏感的任务（如高速抓取）存在挑战。

### 潜在改进方向

1. 引入自适应点采样（如关注高不确定性区域），取代固定格点以提升精细操作的支持。
2. 端到端训练轨迹追踪器，摆脱对离线 CoTracker3 的依赖。
3. 扩展到移动相机场景（补偿自我运动），提升跨平台通用性。

### 可复现性评估

- [ ] 代码开源（暂无公开）
- [ ] 预训练模型（暂无）
- [x] 训练细节完整（超参数、架构均有详细说明）
- [x] 数据集可获取（LIBERO、DROID 均为公开数据集）

---

## 关联笔记

### 基于

- [[World Action Model]]: JOPAT 在其框架上扩展轨迹分支
- [[CoTracker3]]: 提供离线 2D 点轨迹监督
- [[Diffusion Transformer (DiT)]]: 主干架构
- [[DDIM]]: 推理时使用 DDIM 采样器

### 对比

- [[World Action Model|UWM]]: 像素级 WAM 基线，JOPAT 在所有任务上超越
- [[扩散策略]]: 非 WAM 基线，缺乏视觉预测
- [[TraceVLA]]: 同样使用点轨迹但作为策略输入而非生成状态

### 方法相关

- [[点轨迹追踪]]: 核心运动表示
- [[3D 卷积]]: Track-as-Video 编码的关键组件
- [[AdaLN]]: 条件注入机制
- [[行为克隆]]: 有标注数据的监督方式

### 硬件/数据相关

- [[LeRobot]]: 真实机器人实验平台（SO-101）
- [[LIBERO]]: 主要仿真 Benchmark

---

## 速查卡片

> [!summary] Point Tracking Improves World Action Models (JOPAT)
> - **核心**: 在 World Action Model 中联合生成 2D 点轨迹（含可见性）与视觉潜变量和动作
> - **方法**: DiT 统一去噪框架 + Track-as-Video 3D 卷积编码 + 可见性 BCE 监督
> - **结果**: LIBERO 97.8%（SOTA），真实机器人 57.5% vs UWM 32.5%
> - **代码**: 暂无公开

---

*笔记创建时间: 2026-05-26*
