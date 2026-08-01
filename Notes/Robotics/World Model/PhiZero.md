---
title: "PhiZero: A World Model Built Around Physical Language"
method_name: "PhiZero"
authors: [Shuyao Shang, Yuqi Wang, Ruopeng Gao, Xu Chen, Tieniu Tan, Lue Fan, Zhaoxiang Zhang]
year: 2026
venue: arXiv
tags: [world-model, physical-reasoning, discrete-representation, video-generation, self-supervised-learning, robotics, autonomous-driving]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2607.28624
created: 2026-08-01
---

# 论文笔记：PhiZero: A World Model Built Around Physical Language

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | CASIA / UCAS |
| 日期 | July 2026 |
| 项目主页 | [phi-zero.github.io](https://phi-zero.github.io) |
| 对比基线 | [[Wan 2.2-5B]]、[[Cosmos3]] |
| 链接 | [arXiv](https://arxiv.org/abs/2607.28624) |

---

## 一句话总结

> PhiZero 提出"物理语言"——一种从无标注视频中自监督学习的紧凑离散状态转移表示，采用"先推理后渲染"范式实现物理一致的视频世界模型。

---

## 核心贡献

1. **物理语言（Physical Language）**: 从互联网无标注视频中通过[[自监督学习|Self-Supervised Learning]]学习的紧凑离散表示，捕捉世界状态转移模式，充当显式推理的中间空间。
2. **先推理后渲染（Reason-then-Render）**: PhiZero 先用[[自回归模型|Autoregressive VLM]]在物理语言空间中推断未来状态转移，再由[[扩散模型|Diffusion Decoder]]渲染为视频，将动力学推断与像素级合成解耦。
3. **全面验证与迁移能力**: 在物理生成（Physics-IQ、PhyGround、WorldModelBench）和物理理解（IntPhys2、LikePhys、YoCausal）六大 benchmark 上取得 SOTA，并实现零样本跨体态运动迁移和 sim-to-real 迁移。

---

## 问题背景

### 要解决的问题

当前视频世界模型主要在像素空间中直接预测未来帧，导致底层物理动力学被隐式嵌入高维视觉表示中，容易产生物理不一致的生成结果（如碰撞后物体不发生位移、液体不遵循重力等）。

### 现有方法的局限

- **隐式动力学**：像素级直接预测将动力学建模与外观合成混为一谈，无法显式推理物理规律。
- **自然语言粒度不足**：自然语言描述物理状态转移过于粗糙，无法忠实表达视觉经验中的复杂动力学。
- **可控性差**：缺乏细粒度的状态转移接口，难以支持动作条件控制和跨体态迁移。

### 本文的动机

受人类智能启发——人类从视觉经验中抽象出可泛化的世界演化知识，以自然语言进行符号推理。PhiZero 类比于此，学习一种比自然语言更精细的"物理语言"作为物理世界状态转移的符号空间，将推理（What will happen）与渲染（How it looks）分开处理。

---

## 方法详解

### 模型架构

PhiZero 采用 **Tokenizer-Reasoner-Renderer** 双组件架构，以[[物理语言|Physical Language]] $\mathbf{z}$ 为核心接口：

- **输入**: 首帧 $I^0$（当前世界状态）+ 文本动作意图 $c$
- **Physical Language Tokenizer**: 将视频状态转移压缩为离散物理语言序列（编码器）；[[扩散先验解码器|Diffusion-prior Decoder]]从物理语言重建视频（解码器）
- **Physical Language Reasoner**: [[Qwen3-VL]] 初始化的[[自回归VLM|Autoregressive VLM]]，从 $I^0, c$ 预测物理语言序列
- **输出**: 未来视频 $\mathbf{V}$，256 [[有限标量量化|FSQ]] token 对应 33 帧视频

### 核心模块

#### 模块 1: Physical Language Tokenizer（物理语言分词器）

**设计动机**: 利用[[过渡级 Q-Former|Transition-level Q-Former]]捕捉相邻帧间的转移模式，而非全局视频表示，引入局部时序归纳偏置。

**具体实现**:

1. **时空编码器**: 将视频 $\mathbf{V} \in \mathbb{R}^{B \times 3 \times T \times H \times W}$ 编码为隐特征 $\mathbf{x} \in \mathbb{R}^{B \times C \times t \times h \times w}$
2. **[[Q-Former]] 提取转移特征**: 对每对相邻状态 $(\mathbf{x}^i, \mathbf{x}^{i+1})$，共享 Q-Former 提取 $M$ 个转移特征 $\mathbf{q}_i$，拼接得 $\mathbf{q} \in \mathbb{R}^{B \times N \times D_q}$，其中 $N = (t-1)M$
3. **[[有限标量量化|FSQ]] 离散化**: 将连续特征量化为离散物理语言序列，词表大小 25K（量化等级 $8 \times 5^5$），无需单独学习 codebook
4. **[[扩散先验解码器|Diffusion-prior Decoder]]**: 以 Wan2.2-5B 预训练扩散模型为解码器，物理语言上下文 $P_c$ 替换原文本条件，首帧 $I^0$ 提供静态外观锚点，使离散瓶颈专注于编码状态变化

**Pure-noise Warm-up（纯噪声预热）**: 训练初期将所有未来帧隐变量初始化为纯噪声，强制解码器依赖物理语言条件，防止解码器依赖预训练去噪先验"抄近路"。

#### 模块 2: Physical Language Reasoner（物理语言推理器）

**设计动机**: 利用[[预训练VLM|Pretrained VLM]]丰富的视觉语义与常识先验，在物理语言空间中自回归推断未来状态转移。

**具体实现**:

- 以 Qwen3-VL-4B 初始化，词表扩展 FSQ 索引作为原子符号
- 给定 $I^0$ 和文本提示 $c$，在 FSQ 词表（大小 $K=25K$）上自回归预测长度 $N=256$ 的物理语言序列
- 训练数据由冻结 Tokenizer 离线编码生成监督信号，在 [[Teacher Forcing|teacher forcing]] 下以自回归交叉熵优化
- **两阶段训练**: Stage 1 在 5M 通用视频片段上继续预训练；Stage 2 在 1M 运动丰富、物理信息密集片段上 SFT，提升物理合理性

#### 模块 3: 分层数据策划流程

**Tokenizer 数据**: 从约 50K 小时互联网视频池过滤得 10K 小时用于预训练，再经审美质量、运动幅度、状态转移可观测性二次过滤得 5M 四秒片段（含真实+仿真）用于 SFT。

**Reasoner 数据**: 复用 5M 片段做继续预训练；VLM 动态过滤器筛出 1M 运动丰富片段做 SFT，VLM 生成的 caption 仅描述高层动作意图（不含转移细节），防止文本条件泄露结果。

---

## 关键公式

### 公式 1: [[物理世界模型|未来预测联合建模]]

$$
p_{\theta,\psi}(\mathbf{V}, \mathbf{z} \mid I^0, c) = p_\theta(\mathbf{z} \mid I^0, c) \cdot p_\psi(\mathbf{V} \mid I^0, \mathbf{z})
$$

**含义**: PhiZero 将未来预测分解为物理语言推理（$p_\theta$）和视频渲染（$p_\psi$）两个独立步骤，实现动力学推断与像素合成解耦。

**符号说明**:
- $\mathbf{V}$: 未来视频
- $I^0$: 首帧，代表当前世界状态
- $c$: 文本动作意图
- $\mathbf{z}$: 离散物理语言序列（状态转移描述）
- $\theta$: Physical Language Reasoner 参数
- $\psi$: Diffusion Decoder 参数

---

### 公式 2: [[Q-Former|过渡级 Q-Former 特征提取]]

$$
\mathbf{q}_i = \text{QFormer}(\mathbf{Q};\, \mathbf{x}^i,\, \mathbf{x}^{i+1})
$$

**含义**: 共享 Q-Former 联合注意两个相邻隐状态，提取 $M$ 个转移特征，引入局部时序归纳偏置。

**符号说明**:
- $\mathbf{Q} \in \mathbb{R}^{M \times D_q}$: $M$ 个可学习转移查询
- $\mathbf{x}^i \in \mathbb{R}^{B \times C \times h \times w}$: 时间步 $i$ 的隐状态
- $\mathbf{q}_i \in \mathbb{R}^{B \times M \times D_q}$: 第 $i$ 个时间区间的转移特征

---

### 公式 3: [[有限标量量化|FSQ 离散化]]

$$
\mathbf{z} = \text{FSQ}(\text{Proj}_\text{down}(\mathbf{q}))
$$

**含义**: 将连续转移特征投影到低维量化空间后离散化为物理语言序列，FSQ 词表为标量量化等级的笛卡尔积，无需额外 codebook。

**符号说明**:
- $\mathbf{q} \in \mathbb{R}^{B \times N \times D_q}$: 所有时间区间拼接后的转移特征
- $\mathbf{z}$: 离散物理语言序列，词表大小 $K = 8 \times 5^5 = 25000$

---

### 公式 4: [[Flow Matching|扩散先验解码器流匹配目标]]

$$
\mathcal{L}_\text{FM} = \mathbb{E}_{x,\varepsilon,\tau}\left[\left\|v_\psi(x_\tau, \tau;\, I^0, P_c) - (\varepsilon - x_0)\right\|_2^2\right]
$$

$$
x_\tau = (1-\tau)x_0 + \tau\varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)
$$

**含义**: 标准流匹配损失训练扩散解码器，以物理语言上下文 $P_c$ 和首帧 $I^0$ 为条件重建目标视频。

**符号说明**:
- $\tau \sim \mathcal{U}(0, 1)$: 扩散时间步
- $x_0$: 目标视频隐变量（clean latent）
- $\varepsilon \sim \mathcal{N}(0, I)$: 纯噪声
- $x_\tau$: 时间步 $\tau$ 下的加噪隐变量
- $v_\psi$: 扩散 Transformer 速度场预测网络
- $P_c \in \mathbb{R}^{B \times N \times d}$: 物理语言上下文

---

### 公式 5: [[自回归模型|物理语言自回归预测]]

$$
p_\theta(\mathbf{z} \mid I^0, c) = \prod_{j=1}^{N} p_\theta(z_j \mid I^0, c, \mathbf{z}_{<j})
$$

**含义**: 物理语言推理器在时间顺序下自回归分解条件分布，逐步预测每个物理语言 token。

**符号说明**:
- $N = 256$: 物理语言序列长度
- $z_j$: 第 $j$ 个物理语言 token
- $\mathbf{z}_{<j}$: 前 $j-1$ 个已预测 token

---

### 公式 6: [[Teacher Forcing|VLM 自回归训练目标]]

$$
\mathcal{L}_\text{VLM} = -\sum_{j=1}^{N} \log p_\theta(z_j \mid I^0, c, \mathbf{z}_{<j})
$$

**含义**: Teacher Forcing 下的自回归交叉熵损失，监督信号由冻结 Tokenizer 离线编码生成。

**符号说明**:
- 所有符号同公式 5
- 监督标签 $z_j$ 来自 $\mathbf{z} = \mathcal{T}_\phi(\mathbf{V})$（冻结 Tokenizer 编码）

---

### 公式 7: [[物理语言|似然评估]]

$$
s(\mathbf{V}^\pm, c_\text{pair}) = \sum_{j=1}^{N} \log p_\theta(z_j^\pm \mid I_\pm^0, c_\text{pair}, \mathbf{z}_{<j}^\pm)
$$

**含义**: 通过计算两段视频在推理器下的序列似然差异，判断哪段视频更符合物理规律，用于视频理解 benchmark（如 IntPhys2、LikePhys）。

**符号说明**:
- $\mathbf{V}^\pm$: 一对对比视频（物理合理 vs. 不合理）
- $c_\text{pair}$: 对应的判断提示

---

## 关键图表

### Figure 1: PhiZero 整体动机与能力展示

![Figure 1](https://arxiv.org/html/2607.28624v1/x1.png)

**说明**: PhiZero 从互联网视频中学习紧凑离散[[物理语言|Physical Language]]，在物理语言空间中推断世界演化后渲染为视频，支持物理一致的视频生成、细粒度动作条件仿真、零样本运动迁移等能力。

---

### Figure 2: PhiZero 整体流水线

![Figure 2](https://arxiv.org/html/2607.28624v1/x2.png)

**说明**: (a) Physical Language Tokenizer：[[Q-Former|过渡级 Q-Former]] 提取相邻帧转移特征，[[有限标量量化|FSQ]] 离散化为物理语言，[[扩散模型|Diffusion Decoder]] 以物理语言和首帧为条件重建视频。(b) Physical Language Reasoner：[[自回归VLM|Autoregressive VLM]] 从首帧和文本意图预测物理语言序列，再由扩散解码器渲染为未来视频。

---

### Figure 3: 分层数据策划流水线

![Figure 3](https://arxiv.org/html/2607.28624v1/x3.png)

**说明**: 从约 50K 小时真实世界视频池和 1K 小时仿真视频池出发，逐级过滤得到：10K 小时用于 Tokenizer 预训练，5M 四秒片段用于 Tokenizer SFT 和 Reasoner 预训练，1M 运动丰富片段用于 Reasoner SFT。

---

### Figure 4: Physics-IQ Verified 定性对比

![Figure 4](https://arxiv.org/html/2607.28624v1/x4.png)

**说明**: 与 Wan2.2-5B 基线对比，PhiZero 更准确地捕捉碰撞引起的物体位移、重力驱动的变形、链式反应和动态阴影变化等物理后果。

---

### Figure 5: 物理语言的可迁移性

![Figure 5](https://arxiv.org/html/2607.28624v1/x5.png)

**说明**: 将源视频状态转移编码为物理语言序列，在视觉外观被编辑为不同物体/场景的首帧下解码同一序列，迁移后视频保留源视频的转移模式（液体流动、粘性扩散等），验证物理语言与视觉外观的解耦。

---

### Figure 6: 物理语言的语义结构

![Figure 6 (自动驾驶域)](https://arxiv.org/html/2607.28624v1/x6.png)

![Figure 6 (机器人操作域)](https://arxiv.org/html/2607.28624v1/x7.png)

**说明**: 将转移特征聚合为片段级表示，经 PCA 降至 20 维再用 UMAP 投影到 3D。结果显示：物理语言自然按运动模式而非视觉外观组织——自动驾驶中静止片段形成紧致簇、转向模式沿连续流形排列；机器人操作中四种夹取转移模式形成紧致且大体分离的簇。

---

### Figure 7: PhiZero 作为可控交互式世界模型

![Figure 7](https://arxiv.org/html/2607.28624v1/x8.png)

**说明**: PhiZero 生成物理一致的世界演化，遵循细粒度动作条件，支持顺序控制下的交互式 rollout，验证其作为可控世界模型的能力。

---

### Figure 8: 零样本跨体态与 Sim-to-Real 迁移

![Figure 8](https://arxiv.org/html/2607.28624v1/x9.png)

**说明**: 将源视频状态转移编码为物理语言，以指定新体态或视觉域的编辑首帧解码，实现零样本跨体态（人→机器人）和 sim-to-real 迁移，无需目标特定训练数据。

---

### Table 1: Physics-IQ Verified 结果

| 模型 | S-IoU ↑ | ST-IoU ↑ | WS-IoU ↑ | IQ-Score ↑ |
|------|---------|---------|---------|-----------|
| Wan2.2-5B | 24.7 | 22.6 | 13.3 | 21.2 |
| Sora 2 | 37.3 | 27.0 | 26.9 | 26.5 |
| Cosmos3-Nano | 40.4 | 22.0 | 24.6 | 29.1 |
| Wan2.2-14B | 51.1 | 20.5 | 28.5 | 32.2 |
| Hunyuan-Video | 47.1 | 26.9 | 29.7 | 33.4 |
| Grok-Video | 52.7 | 21.4 | 35.7 | 34.8 |
| Cosmos3-Super | — | — | — | 39.5 |
| **PhiZero** | **58.2** | **36.8** | **27.6** | **41.2** |

**关键发现**: PhiZero IQ-Score 41.2，超越 Cosmos3-Super（39.5）和 Wan2.2-14B（32.2）等强基线，在 S-IoU 和 ST-IoU 上均领先，说明"先推理后渲染"范式对物理一致性有显著提升。

---

### Table 2: PhyGround 结果

| 模型 | General Quality ↑ | Physics Score ↑ | Overall ↑ |
|------|-------------------|-----------------|-----------|
| LTX-Video-19B | 2.56 | 2.54 | 2.55 |
| LTX-Video-22B | 2.59 | 2.56 | 2.57 |
| Wan2.2-5B | 2.67 | 2.65 | 2.66 |
| Cosmos2.5-2B | 2.79 | 2.70 | 2.74 |
| OmniWeaving | 2.89 | 2.78 | 2.84 |
| Veo3.1 | 3.01 | 2.85 | 2.93 |
| Wan2.2-14B | 3.00 | 2.90 | 2.95 |
| **PhiZero** | **2.93** | **3.01** | **2.97** |

**关键发现**: PhiZero Physics Score 3.01 排名第一，超越 Veo3.1（2.85）和 Wan2.2-14B（2.90），尽管 General Quality 略低于 Veo3.1，但总体 Overall 最优（2.97）。

---

### Table 3: WorldModelBench 结果

| 模型 | Physics Adherence ↑ | Common Sense ↑ | Total ↑ |
|------|-------------------|----------------|---------|
| Pandora | 4.05 | 0.95 | 6.57 |
| OpenSora-Plan | 3.85 | 1.01 | 6.62 |
| CogVideoX | 3.88 | 0.99 | 6.75 |
| Mochi | 4.14 | 1.26 | 7.62 |
| Luma | 4.13 | 1.57 | 7.72 |
| Wan2.2-5B | 4.51 | 1.41 | 7.98 |
| Runway | 4.27 | 1.65 | 8.08 |
| **PhiZero** | **4.88** | **1.71** | **8.19** |

**关键发现**: PhiZero 在 Physics Adherence（4.88）和 Total（8.19）均位列第一，大幅超越 Wan2.2-5B（4.51 / 7.98）。

---

### Table 4: IntPhys2 直觉物理理解结果

| 模型 | Easy ↑ | Medium ↑ | Hard ↑ | Overall ↑ |
|------|---------|----------|---------|------------|
| Cosmos-4B | 46.00 | 52.00 | 48.05 | 49.41 |
| Qwen2.5-VL | 50.96 | 53.25 | 51.49 | 52.27 |
| Gemini-1.5 Pro | 58.65 | 53.00 | 52.67 | 52.27 |
| GPT-4o | 57.69 | 54.75 | 54.17 | 53.75 |
| VideoMAEv2 | 46.00 | 58.50 | 52.73 | 53.75 |
| V-JEPA | 52.00 | 53.00 | 57.42 | 53.75 |
| Gemini-2.5 Flash | 64.42 | 56.75 | 54.46 | 55.63 |
| **PhiZero** | **60.98** | **60.50** | **52.38** | **56.34** |

**关键发现**: PhiZero Overall 56.34% 超越 Gemini-2.5 Flash（55.63%）等强判别模型，验证物理语言蕴含物理合理性判断能力。

---

### Table 5: LikePhys 物理似然辨别结果（误差越低越好）

| 模型 | Rigid ↓ | Fluid ↓ | Optical ↓ | Avg. Error ↓ |
|------|---------|---------|-----------|---------------|
| AnimateDiff-SDXL | 61.44 | 53.33 | 37.65 | 56.0 |
| ZeroScope | 57.32 | 49.23 | 41.00 | 53.3 |
| ModelScope | 59.46 | 47.23 | 35.65 | 52.9 |
| Mochi | 54.22 | 47.33 | 49.15 | 51.9 |
| CogVideoX-5B | 59.00 | 43.10 | 26.65 | 49.8 |
| CogVideoX-2B | 56.98 | 42.00 | 25.15 | 48.2 |
| Wan2.1-1.3B | 44.66 | 57.10 | 41.65 | 48.0 |
| LTX-Video-2B | 43.44 | 33.43 | 61.50 | 44.7 |
| CogVideoX1.5-5B | 44.14 | 47.90 | 42.65 | 43.8 |
| Wan2.1-14B | 43.34 | 45.00 | 38.35 | 43.8 |
| Hunyuan-Video | 36.34 | 48.67 | 41.15 | 43.6 |
| **PhiZero** | **29.14** | **53.15** | **37.50** | **41.7** |

**关键发现**: PhiZero Avg. Error 41.7 排名最优，在刚体（Rigid）物理辨别上以 29.14 大幅领先。

---

### Table 6: YoCausal 真实世界因果理解结果

| 模型 | RSI (%) ↑ | CCI (%) ↑ | Agg. Rank ↓ |
|------|-----------|-----------|-------------|
| CogVideoX-2B | 41.50 | 0.93 | 10.0 |
| Wan2.2-5B | 51.91 | -2.12 | 9.0 |
| Hunyuan-Video | 52.05 | -0.29 | 8.0 |
| Wan2.1-1.3B | 45.51 | 5.36 | 7.5 |
| Mochi | 49.12 | 3.85 | 8.0 |
| CogVideoX1.5-5B | 46.83 | 4.85 | 8.0 |
| CogVideoX-5B | 49.92 | 5.09 | 6.5 |
| LTX-Video-13B | 56.48 | -4.32 | 7.0 |
| LTX-Video-2B | 58.86 | -0.20 | 5.0 |
| Wan2.1-14B | 53.24 | 5.91 | 3.5 |
| Wan2.2-14B | 54.19 | 5.51 | 3.5 |
| **PhiZero** | **55.54** | **6.20** | **2.0** |

**关键发现**: PhiZero Agg. Rank 2.0 排名第一（越低越好），CCI 6.20 最高，说明物理语言理解因果关系的能力最强。

---

### Table 7: Physical Language Tokenizer 重建性能对比

| 模型 | Tokens ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|----------|---------|---------|----------|
| Wan2.2-5B VAE | 44800 | 37.7 | 0.957 | 0.042 |
| Video-LaVIT | 270 | 23.6 | 0.835 | 0.175 |
| VideoFlexTok (k=32) | 288 | 25.2 | 0.858 | 0.177 |
| VideoFlexTok (k=64) | 576 | 26.5 | 0.870 | 0.167 |
| **Ours** | **256** | **28.9** | **0.903** | **0.087** |

**关键发现**: PhiZero Tokenizer 仅用 256 离散 token（相比 VAE 的 44800 连续 token，压缩比 **175×**），PSNR 28.9 / SSIM 0.903 / LPIPS 0.087，在极度紧凑的离散表示下重建质量最优，大幅超越 Video-LaVIT 和 VideoFlexTok。

---

### Table 8: Tokenizer 消融实验

| 配置 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|---------|---------|----------|
| w/o Diffusion Decoder | 26.6 | 0.875 | 0.119 |
| w/o Transition Q-Former | 28.2 | 0.896 | 0.089 |
| w/o Pure-noise Warm-up | 27.9 | 0.894 | 0.091 |
| **Full** | **28.9** | **0.903** | **0.087** |

**关键发现**: 移除扩散解码器退化最大（LPIPS +0.032），证明扩散先验对恢复细粒度外观细节至关重要；过渡级 Q-Former 和纯噪声预热均有显著贡献。

---

### Table 9: Reasoner 消融实验

| 方法 | IQ-Score ↑ |
|------|-----------|
| Wan2.2-5B（基线） | 21.2 |
| + Prompt Enhancement | 26.6 |
| Ours w/o Simulation Data | 37.7 |
| Ours w/o Two-stage Training | 39.2 |
| **Ours（Full）** | **41.2** |

**关键发现**: 仅靠 Prompt Enhancement 提升有限（26.6），说明自然语言推理不足以捕捉细粒度状态转移；仿真数据（+1.5 IQ）和两阶段训练（+2.0 IQ）均有显著增益。

---

## 实验

### 数据集与训练规模

| 数据集/阶段 | 规模 | 用途 |
|------------|------|------|
| 真实世界视频池 | ~50K 小时 | Tokenizer 过滤起点 |
| Tokenizer 预训练数据 | ~10K 小时 | Tokenizer Stage 1 |
| Tokenizer SFT 数据 | 5M 四秒片段（真实+仿真） | Tokenizer Stage 2 |
| Reasoner 预训练数据 | 5M 四秒片段 | Reasoner Stage 1 |
| Reasoner SFT 数据 | 1M 运动丰富片段 | Reasoner Stage 2 |
| 仿真视频池 | ~1K 小时 | 辅助 Tokenizer SFT + Reasoner SFT |

### 实现细节

- **Tokenizer 编码器**: Wan2.2 VAE 编码器（预训练权重初始化）
- **Tokenizer 解码器**: Wan2.2-5B 扩散模型（[[LoRA]] rank=32 微调）
- **FSQ 配置**: 量化等级 $(8, 5, 5, 5, 5, 5)$，词表 $K=25000$
- **Q-Former 查询数**: $M=32$，产生 256-token 序列（对应 33 帧视频）
- **Reasoner 基座**: Qwen3-VL-4B
- **分辨率课程**: 预训练 256×448 → SFT 512×896

### 评估 Benchmark

- **生成质量**: Physics-IQ Verified、PhyGround、WorldModelBench
- **物理理解**: IntPhys2（直觉物理）、LikePhys（物理似然辨别）、YoCausal（因果理解）
- **Tokenizer 重建**: PSNR / SSIM / LPIPS（33 帧视频）

---

## 批判性思考

### 优点

1. **显式物理推理**: "先推理后渲染"范式将动力学建模显式化，而非隐式嵌入像素空间，有理论依据。
2. **极度紧凑**: 256 离散 token 表示 33 帧视频（175× 压缩比），计算效率高，利于序列建模。
3. **强泛化性**: 从无标注视频学习，支持开放域场景；物理语言与外观解耦实现零样本跨体态迁移，无需目标特定数据。
4. **全面验证**: 同时覆盖生成和理解两类 benchmark，验证物理语言的双向能力。

### 局限性

1. **仅限于 4 秒片段**: 当前训练和评估基于 4 秒视频，长时序物理推理能力未经验证。
2. **流体物理表现一般**: LikePhys Fluid 误差 53.15 不及部分基线，说明流体动力学难以被离散物理语言充分捕捉。
3. **推理速度**: 需要先自回归生成 256 个物理语言 token，再扩散解码，推理延迟高于直接扩散方法。
4. **词表设计依赖**: FSQ 词表大小 25K 的选择对性能影响未充分消融。

### 潜在改进方向

1. 层级物理语言（粗粒度全局 + 细粒度局部）以更好捕捉流体等复杂动力学。
2. 将物理语言作为 VLA 的动作表示，探索 sim-to-real 控制迁移的闭环应用。
3. 多帧条件（不只首帧）以支持更长时序的世界模型 rollout。

### 可复现性评估

- [ ] 代码开源（项目主页存在但代码链接未明确）
- [ ] 预训练模型（未公开）
- [x] 训练细节完整（论文 Appendix 包含超参和数据细节）
- [ ] 数据集可获取（内部策划数据集，未公开）

---

## 关联笔记

### 基于

- [[Wan 2.2-5B]]: 扩散解码器基座，提供强生成先验
- [[Qwen]]: Physical Language Reasoner 初始化基座（Qwen3-VL-4B）
- [[Q-Former]]: 过渡级特征提取核心组件（来自 BLIP-2）
- [[有限标量量化|FSQ]]: 离散化方案，构建物理语言词表

### 对比

- [[Wan 2.2-5B]]: 直接像素级扩散预测基线，IQ-Score 21.2 vs. PhiZero 41.2
- [[Cosmos3]]: 竞争性世界模型，Cosmos3-Super IQ-Score 39.5 vs. PhiZero 41.2

### 方法相关

- [[Flow Matching]]: Tokenizer 扩散解码器的训练目标
- [[有限标量量化|FSQ]]: 核心离散化方法
- [[自监督学习|Self-Supervised Learning]]: 物理语言学习范式
- [[Curriculum Learning]]: Tokenizer 时序-空间课程训练策略
- [[Teacher Forcing]]: Reasoner 自回归训练策略
- [[LoRA]]: 扩散解码器微调策略
- [[世界模型 (World Model)]]: 本文所属方向

### 硬件/数据相关

- [[Video Diffusion Model]]: Wan2.2-5B 扩散解码器所属类别

---

## 速查卡片

> [!summary] PhiZero: A World Model Built Around Physical Language
> - **核心**: 学习离散"物理语言"表示世界状态转移，先在语言空间推理再渲染为视频
> - **方法**: Transition Q-Former + FSQ + 扩散先验解码器 + Qwen3-VL-4B 推理器
> - **结果**: Physics-IQ 41.2 SOTA，256 token 实现 175× 压缩，零样本跨体态迁移
> - **代码**: [phi-zero.github.io](https://phi-zero.github.io)

---

*笔记创建时间: 2026-08-01*
