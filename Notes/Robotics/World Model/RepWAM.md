---
title: "RepWAM: World Action Modeling with Representation Visual-Action Tokenizers"
method_name: "RepWAM"
authors: [Junke Wang, Qihang Zhang, Shuai Yang, Yiming Luo, Yujun Shen, Zuxuan Wu, Yu-Gang Jiang, Yinghao Xu]
year: 2026
venue: arXiv
tags: [world-action-model, visual-tokenization, latent-action, robot-manipulation, flow-matching, diffusion-transformer]
zotero_collection: Robotics/World Model
image_source: local
arxiv_html: https://arxiv.org/html/2606.13674
created: 2026-06-13
---

# 论文笔记：RepWAM: World Action Modeling with Representation Visual-Action Tokenizers

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Fudan University（可信具身智能研究所）/ Ant Group / HKUST |
| 日期 | June 2026 |
| 项目主页 | [wdrink.github.io/RepWAM](https://wdrink.github.io/RepWAM) |
| 对比基线 | [[pi0.5]]、[[Motus]]、[[Lingbot-VA]]、[[LAPA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.13674) / [Code](https://github.com/wdrink/RepWAM) |

---

## 一句话总结

> RepWAM 用语义对齐的视觉-动作分词器替代像素重建分词器，在共享语义潜空间中联合建模视觉预测与潜在动作，实现了更强的机器人操控泛化能力。

---

## 核心贡献

1. **RepViTok（表示视觉分词器）**: 将视频潜变量与冻结的视觉基础模型对齐，获得语义丰富的视觉 token，而非传统的像素重建 token
2. **语义潜在动作 Tokenizer**: 在语义视觉潜空间中用[[逆动力学模型|IDM]]+[[前向动力学模型|FDM]]捕获帧间转换，使每个动作 token 成为语义状态之间的变换
3. **因果世界动作模型（Causal WAM）**: 基于[[条件流匹配]]的[[扩散 Transformer]]，块因果注意力机制对视觉-动作 chunk 进行联合生成，两阶段训练策略实现预训练到机器人控制的高效迁移

---

## 问题背景

### 要解决的问题

现有[[世界动作模型|World Action Model (WAM)]]继承了视频生成领域的像素重建分词器，这类分词器专为视频重建设计，不适合机器人控制任务的语义理解需求。机器人需要理解状态变化的语义含义，而非像素级别的视觉细节。

### 现有方法的局限

- **Motus、DreamZero、Lingbot-VA** 等 WAM 直接沿用预训练视频生成器的分词器（如 WAN2.2 VAE），这些分词器优化目标是重建质量而非语义对齐
- 现有[[潜在动作模型]]（LAPA、LAPO）学习的潜在动作与视觉语义空间解耦，导致跨场景迁移能力不足
- 像素重建优化会引入大量低层纹理偏差，对机器人控制任务造成噪声干扰

### 本文的动机

在语义对齐的视觉潜空间中定义动作 token（即相邻语义状态间的变换），可以使动作表示更具泛化性、更接近任务语义，同时减少视频分类器自由引导（CFG）对生成质量的依赖。

---

## 方法详解

### 模型架构

![[RepWAM_fig1_overview.png]]

RepWAM 采用三阶段 **表示中心** 架构：

- **输入**: 语言指令 $c$ + 观测帧序列 $o_{1:T}$
- **视觉编码**: [[RepViTok]]（[[Vision Transformer]] 自编码器）将帧编码为语义视觉 token $z_t$
- **动作编码**: [[逆动力学模型|IDM]] + [[前向动力学模型|FDM]] 将帧转换编码为潜在动作 token $\ell_t$
- **预测主干**: [[因果扩散 Transformer]] 在 chunk 级别联合预测 $z$ 和 $\ell$
- **输出**: 解码的未来视觉帧 + 通过冻结 IDM 适配的真实机器人动作
- **总参数**: 1.3B / 5B（两个规模）

---

### 核心模块

#### 模块 1: RepViTok（表示视觉分词器）

**设计动机**: 利用[[视觉基础模型]]（如 [[Perception Encoder]]）提供语义监督，使视频潜变量具备语义对齐能力

**具体实现**:
- 将初始帧分割为 $16 \times 16$ 图像 patch；后续帧分割为 $4 \times 16 \times 16$ 时空管元（tubelet）
- 编码器 + 解码器采用 12 层 Transformer，隐藏维度 768，时序因果掩码 + 空间注意力
- 解码器通过转置卷积层的对称架构实现重建
- 双损失监督：重建损失 + **语义对齐损失**（将视频潜变量对齐到冻结教师模型）

#### 模块 2: 潜在动作 Tokenizer（IDM + FDM）

**设计动机**: 在已经对齐语义的视觉潜空间中，捕获帧转换的语义动作表示

**具体实现**:
- **[[逆动力学模型|IDM]]** $q_\phi$: 给定相邻潜变量 $(z_t, z_{t+1})$，压缩得到紧凑潜在码 $\ell_t$
- **[[前向动力学模型|FDM]]** $f_\psi$: 给定 $(z_t, \ell_t)$ 输出变换算子 $K_t$ 和残差 $\delta_t$，预测下一状态
- 双向一致性损失（前向预测 + 后向一致）保证训练稳定性

#### 模块 3: 因果世界动作模型

**设计动机**: 联合建模视觉与动作的生成过程，支持语言条件的闭环机器人控制

**具体实现**:
- 将视觉 token 和动作 token 组织为时序 chunk：$u_{t:t+k} = [z_{t:t+k}, \ell_{t:t+k-1}]$
- 块因果掩码（block-causal masking）：每个 chunk 只关注历史 chunk $s_{<t}$
- 模态共享注意力权重 + 模态特定前馈网络
- 语言条件来自冻结文本编码器
- 推理时：chunk 大小设为 2，注意力窗口 32，视频去噪步数 5，动作去噪步数 10

---

## 关键公式

### 公式 1: [[视觉重建损失|视觉 Tokenizer 重建损失]]

$$
\mathcal{L}_\text{rec} = \lambda_1 \|o - \hat{o}\|_1 + \lambda_\text{perc} \mathcal{L}_\text{perc}(o, \hat{o}) + \lambda_\text{gan} \mathcal{L}_\text{gan}(\hat{o})
$$

**含义**: 多项损失联合监督视觉分词器的重建质量，包括像素级 L1、感知相似度和对抗判别损失

**符号说明**:
- $o$: 原始帧
- $\hat{o} = D_\theta(z)$: 解码器重建帧
- $\lambda_1, \lambda_\text{perc}, \lambda_\text{gan}$: 各损失项权重系数
- $\mathcal{L}_\text{perc}$: [[感知损失]]（perceptual loss）
- $\mathcal{L}_\text{gan}$: [[对抗损失]]（GAN discriminator loss）

### 公式 2: [[语义对齐损失|视觉语义对齐损失]]

$$
\mathcal{L}_\text{align} = \left\| \text{avg}(W_\text{align} z) - \text{avg}(G(o)) \right\|_2^2
$$

**含义**: 将视频潜变量对齐至冻结的视觉基础模型教师，赋予视觉 token 语义信息

**符号说明**:
- $z$: 视觉编码器输出的潜变量
- $G$: 冻结的视觉基础模型（教师模型，如 [[Perception Encoder]]）
- $W_\text{align}$: 线性投影层，将 $z$ 维度对齐至教师特征维度
- $\text{avg}(\cdot)$: 对空间/时序维度做平均池化

### 公式 3: [[潜在动作 Tokenization|IDM 与 FDM 耦合]]

$$
\ell_t = q_\phi(z_t, z_{t+1}), \qquad (K_t, \delta_t) = f_\psi(z_t, \ell_t), \qquad \hat{z}_{t+1} = K_t z_t + \delta_t
$$

**含义**: IDM 从相邻语义帧压缩得到潜在动作码；FDM 用变换算子和残差实现前向预测

**符号说明**:
- $\ell_t$: 第 $t$ 步潜在动作 token
- $q_\phi$: [[逆动力学模型]]（IDM）参数化为 $\phi$
- $K_t$: 变换算子（transport operator），捕获状态间的主要变换
- $\delta_t$: 残差修正项，捕获细粒度变化
- $f_\psi$: [[前向动力学模型]]（FDM）参数化为 $\psi$

### 公式 4: [[前向一致性损失|IDM+FDM 训练损失]]

$$
\mathcal{L}_\text{fwd} = \sum_{t=1}^{T'-1} \|\hat{z}_{t+1} - z_{t+1}\|_2^2, \qquad \mathcal{L}_\text{cons} = \sum_{t=1}^{T'-1} \|\hat{z}_t - z_t\|_2^2
$$

**含义**: 前向预测损失确保 FDM 能正确预测下一状态；后向一致性损失增强双向稳定性

**符号说明**:
- $\mathcal{L}_\text{fwd}$: 前向预测损失
- $\mathcal{L}_\text{cons}$: 后向一致性损失
- $T'$: 序列长度

### 公式 5: [[Visual-Action Chunk|视觉-动作 Chunk 组织]]

$$
u_{t:t+k} = [z_{t:t+k},\ \ell_{t:t+k-1}], \qquad s = [c,\ z_1,\ u_{t_1:t_1+k},\ \ldots,\ u_{t_n:t_n+k}]
$$

**含义**: 将视觉 token 和动作 token 打包为时序 chunk，构成因果 Transformer 的完整输入序列

**符号说明**:
- $u_{t:t+k}$: 第 $t$ 到 $t+k$ 的视觉-动作 chunk
- $k$: chunk 大小（训练时从 $[1,4]$ 采样，推理时设为 2）
- $c$: 语言指令条件
- $z_1$: 初始帧的视觉 token（作为历史锚点）

### 公式 6: [[条件流匹配损失|WAM 流匹配训练目标]]

$$
x_\alpha = (1-\alpha)\varepsilon_{t:t+k} + \alpha \cdot u_{t:t+k}, \qquad \dot{x}_\alpha = u_{t:t+k} - \varepsilon_{t:t+k}
$$

$$
\mathcal{L}_\text{FM} = \mathbb{E}\!\left[\left\|F_\theta^v(x_\alpha, \alpha, s_{<t}) - \dot{x}_\alpha^v\right\|_2^2 + \lambda_a \left\|F_\theta^a(x_\alpha, \alpha, s_{<t}) - \dot{x}_\alpha^a\right\|_2^2\right]
$$

**含义**: 用[[条件流匹配]]训练因果 Transformer，同时回归视觉分量和动作分量的速度场目标

**符号说明**:
- $\alpha \in [0,1]$: 流匹配插值系数
- $\varepsilon_{t:t+k}$: 标准高斯噪声
- $x_\alpha$: 噪声插值样本
- $\dot{x}_\alpha$: 速度目标（插值方向）
- $F_\theta^v, F_\theta^a$: Transformer 的视觉和动作输出头
- $\lambda_a$: 动作损失权重

---

## 关键图表

### Figure 1: 系统概览 — RepViTok 与 Latent Action Tokenizer

![[RepWAM_fig1_overview.png]]

**说明**: RepWAM 的表示视觉-动作分词器概览。对比三种方案：(a) Video VAE（像素重建）、(b) Latent Action Models（视觉动作分离）、(c) RepWAM 的 Representation Visual-Action Tokenizer。RepViTok 将视觉帧对齐至冻结基础模型；[[逆动力学模型|IDM]] 和 [[前向动力学模型|FDM]] 在同一语义空间中捕获帧间变换 $\ell_t$。

### Figure 2: 真实机器人任务成功率对比

![[RepWAM_fig2_success_rate.png]]

**说明**: 三项真实 Franka 双臂操控任务（Pick Fruit / Push Drawer / Insert Tube）上的成功率对比（各 10 次）。RepWAM-5B 在 Push Drawer 和 Insert Tube 任务上显著超越 [[pi0.5]] 和 [[Lingbot-VA]]，长时域任务提升尤为明显。

### Figure 3: 真实机器人执行可视化

![[RepWAM_fig3_robot_execution.png]]

**说明**: 三项任务的代表性成功执行帧序列。从左到右分别为：Pick Fruit（抓取水果）、Push Drawer（推抽屉）、Insert Tube（插管），展示模型在精细操控和长时域任务中的稳定性。

### Figure 4: 潜在动作可视化与 IDM 适配损失

![[RepWAM_fig4_latent_actions.png]]

**说明**: 左图对比 [[LAPA]] 与 RepViTok 的潜在动作可视化——RepViTok 的动作在语义空间中呈现更结构化的聚类。右图展示以冻结动作 latent 进行 IDM 适配时的损失曲线，验证两阶段迁移策略的有效性。

### Figure 5: 视频 CFG 尺度对成功率的影响

![[RepWAM_fig5_cfg.png]]

**说明**: 在 RoboTwin 2.0 Easy/Hard 两个难度下，不同视频[[分类器自由引导|CFG]] 尺度（1.0 / 1.25 / 2.0）的影响对比。RepWAM 在 CFG=1.0（无额外 CFG 外推）时表现最佳，证明语义潜空间已内在包含语言对齐信息，无需 CFG 增强。

### Figure 6: ImageNet 和 UCF101 上的重建质量对比

![[RepWAM_fig6_reconstruction.png]]

**说明**: RepViTok 在 ImageNet（图像）和 UCF101（视频）上的定性重建示例。保留语义细节的同时维持时序一致性，优于纯重建基线。

---

### Table 1: RoboTwin 2.0 Benchmark 定量对比

| 方法 | Easy hor=2 | Hard hor=2 | Easy hor=3 | Hard hor=3 | Avg Easy (50任务) | Avg Hard (50任务) |
|------|-----------|-----------|-----------|-----------|-----------------|-----------------|
| π₀.₅ | 79.3 | 73.0 | 78.6 | 67.4 | 82.7 | 76.8 |
| Motus | 85.2 | 80.9 | 85.0 | 84.2 | 88.7 | 87.0 |
| Lingbot-VA | 85.3 | 86.9 | 89.6 | 90.6 | 92.9 | 91.6 |
| **RepWAM-1.3B** | **85.7** | **84.0** | **92.0** | **85.4** | **86.6** | **83.1** |
| **RepWAM-5B** | **87.4** | **87.6** | **88.0** | **90.4** | **89.3** | **88.4** |

*注：π₀.₅、Motus、Lingbot-VA 使用了预训练骨干网络，RepWAM 从头训练*

**关键发现**: RepWAM-5B 在多数设置下超越 Motus 和 π₀.₅，与 Lingbot-VA 竞争；RepWAM-1.3B 在 hor=3 Easy 上达到最高 92.0%。

### Table 2: 视觉 Tokenizer 设计消融

| 模型 | gFVD (Seen) ↓ | OLS (Seen) ↑ | gFVD (Unseen) ↓ | OLS (Unseen) ↑ | PickFruit 成功率 |
|------|--------------|-------------|----------------|---------------|----------------|
| WAN2.2 VAE | 67.42 | 13.68 | 83.98 | 11.21 | 20% |
| ViTok | 69.23 | 16.29 | 80.14 | 13.81 | 10% |
| **RepViTok** | **61.01** | **18.82** | **72.91** | **14.15** | **30%** |

**关键发现**: RepViTok 相比 WAN2.2 VAE 将 gFVD 降低 9.5%/13.2%，OLS 从 13.68 提升到 18.82，真实机器人成功率提升 10 个百分点。

### Table 3: VAE 对比（RoboTwin 2.0）

| 指标 | WAN2.2 VAE Easy | WAN2.2 VAE Hard | RepViTok Easy | RepViTok Hard |
|------|----------------|----------------|---------------|---------------|
| Avg hor=1 | 81.1 | 78.4 | 86.2 | 83.1 |
| Avg hor=2 | 75.5 | 73.9 | 85.7 | 84.0 |
| Avg hor=3 | 67.2 | 68.0 | 92.0 | 85.4 |
| **Avg 50 Tasks** | **78.0** | **76.0** | **86.6** | **83.1** |

**关键发现**: RepViTok 在所有 horizon 设置下均超越 WAN2.2 VAE，平均提升约 8 个百分点。

### Table 4: 潜在动作训练策略消融

| 配置 | gFVD (Seen) ↓ | OLS (Seen) ↑ | gFVD (Unseen) ↓ | OLS (Unseen) ↑ | PickFruit 成功率 |
|------|--------------|-------------|----------------|---------------|----------------|
| w/o latent actions | 61.01 | 18.82 | 72.91 | 14.15 | 30% |
| Joint Prediction | 94.25 | 18.52 | 98.77 | 15.22 | 20% |
| **Two Stages（提出）** | **48.23** | **19.87** | **58.83** | **16.98** | **50%** |

**关键发现**: 两阶段训练将 gFVD 从 61.01 降至 48.23，PickFruit 成功率从 30% 提升到 50%；联合预测反而会降低生成质量（gFVD 升至 94.25）。

### Table 5: 重建质量评估

| 模型 | ImageNet 256 PSNR ↑ | ImageNet 512 PSNR ↑ | UCF101 256×17 rFVD ↓ | UCF101 512×17 rFVD ↓ |
|------|--------------------|--------------------|---------------------|---------------------|
| WAN2.2 VAE | 28.16 | 30.48 | 4.28 | 0.68 |
| ViTok | 28.65 | 30.77 | 1.23 | 0.16 |
| **RepViTok** | **28.90** | **31.00** | **1.09** | **0.16** |

**关键发现**: RepViTok 在 PSNR 和 rFVD 上均达到最优，在提升语义质量的同时并未牺牲重建精度。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| AgiBot | ~100G video-action token | 真实机器人操控演示 | 预训练 |
| 混合真实机器人演示 | ~300G | 多任务、多场景 | 适配训练 |
| RoboTwin 2.0 | 50 任务（Easy+Hard） | 仿真操控基准 | 仿真测试 |
| Franka 双臂 | 3 任务 × 10次 rollout | Pick Fruit / Push Drawer / Insert Tube | 真实机器人测试 |
| ImageNet | 标准图像集 | 重建质量评估 | 分词器评估 |
| UCF101 | 256×17 / 512×17 | 视频重建质量评估 | 分词器评估 |

### 实现细节

- **视觉分词器**: 12 层 Transformer，隐藏维度 768，时序因果掩码 + 空间注意力
- **模型规模**:
  - 1.3B: 30 层 Transformer，隐藏维度 1536
  - 5B: 30 层 Transformer，隐藏维度 3072
- **优化器**: [[Muon optimizer]]，峰值学习率 $1 \times 10^{-2}$
- **精度**: bfloat16
- **硬件**: H20 GPU（1.3B 用 64 卡，5B 用 128 卡）
- **推理参数**: chunk 大小 2，注意力窗口 32，视频去噪 5 步，动作去噪 10 步

### 可视化结果

RepViTok 在 ImageNet 和 UCF101 上保留了语义细节（如物体边缘、纹理），同时维持时序一致性；相比 WAN2.2 VAE 的像素重建偏差，RepViTok 的重建结果更接近语义真值。真实机器人任务中，RepWAM-5B 在长时域和精细操控（Push Drawer / Insert Tube）上的提升最为显著。

---

## 批判性思考

### 优点

1. **语义分词的清晰动机**: 从"为什么用像素重建 token"的反问出发，设计目标明确，理论动机清晰
2. **两阶段训练策略的有效性**: Table 4 显示联合预测反而有害，两阶段策略更优，消融设计合理
3. **对 CFG 依赖性低**: CFG=1.0 时表现最优，说明语义潜空间已内在语言对齐，减少推理计算开销

### 局限性

1. **与 Lingbot-VA 对比不公平**: Lingbot-VA 使用了预训练骨干，RepWAM 从头训练，但在 50 任务平均成功率上仍略低于 Lingbot-VA（89.3% vs 92.9%），说明预训练知识仍有价值
2. **预训练数据受限**: 当前仅在 AgiBot 数据上预训练，作者承认未来需扩展到互联网视频和以人为中心的视频以提升泛化性
3. **计算开销大**: 5B 模型需要 128 张 H20 GPU，对大多数研究机构不友好

### 潜在改进方向

1. 扩展预训练数据至大规模互联网/以人为中心的视频数据
2. 探索更高效的推理策略（如蒸馏至更小模型、减少去噪步数）
3. 将语义对齐思路应用于非操控类任务（如导航、整体运动控制）

### 可复现性评估

- [x] 代码开源（GitHub 已承诺）
- [x] 预训练模型（承诺发布）
- [x] 训练细节完整（优化器、学习率、硬件均有说明）
- [ ] 数据集可获取（AgiBot 数据集需联系获取）

---

## 关联笔记

### 基于

- [[Lingbot-VA]]: 同类 World Action Model 基线，采用因果视频生成
- [[LAPA]]: 潜在动作预训练前置工作，RepWAM 的动作 tokenizer 设计受其启发并改进
- [[Motus]]: 另一 WAM 基线，使用预训练视频生成器骨干

### 对比

- [[pi0.5]]: VLA 基线，在真实机器人和 RoboTwin 上均作对比
- [[Flash-WAM]]: 同为 WAM 系列，关注推理效率

### 方法相关

- [[条件流匹配]]: RepWAM 的核心训练目标
- [[逆动力学模型]]: 用于从帧对压缩潜在动作
- [[前向动力学模型]]: 与 IDM 耦合实现动作一致性
- [[Vision Transformer]]: RepViTok 的骨干架构
- [[扩散 Transformer]]: WAM 的预测主干

### 硬件/数据相关

- [[Franka 机械臂]]: 真实机器人实验平台（双臂配置）
- [[RoboTwin 2.0]]: 仿真操控基准（50 任务）

---

## 速查卡片

> [!summary] RepWAM (2026)
> - **核心**: 用语义对齐的 RepViTok 替代像素重建分词器，在语义潜空间中联合建模视觉与潜在动作
> - **方法**: RepViTok（语义对齐） + IDM/FDM（语义动作 token） + 因果流匹配 Transformer（两阶段训练）
> - **结果**: RoboTwin 2.0 Easy/Hard 平均 89.3%/88.4%，真实机器人三任务平均成功率超越 π₀.₅ 和 Lingbot-VA
> - **代码**: [github.com/wdrink/RepWAM](https://github.com/wdrink/RepWAM)

---

*笔记创建时间: 2026-06-13*
