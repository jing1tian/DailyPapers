---
title: "TS-Mask VLA: 2D Temporal-Spatial Masking for Vision-Language-Action Model with Effective Bridging"
method_name: "TS-Mask VLA"
authors: [Shengzhuo Yang, Ronghao Yu, Chuanjie Lv, Linpeng Peng, Hang Yu, Jie Ren, Jiajun Lv, Yong Liu]
year: 2026
venue: IROS 2026
tags: [vla, discrete-diffusion, masked-diffusion, robotic-manipulation, temporal-spatial-masking]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2607.09818v1
created: 2026-07-15
---

# 论文笔记：TS-Mask VLA: 2D Temporal-Spatial Masking for Vision-Language-Action Model with Effective Bridging

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | （未明确列出） |
| 日期 | July 2026 |
| 项目主页 | 无 |
| 对比基线 | [[π₀]], [[OpenVLA-OFT]], [[FlowVLA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2607.09818) |

---

## 一句话总结

> 提出 TS-Mask VLA：用 [[离散扩散]] 动作专家 + [[Bridge Attention]] 条件注入 + [[时序-空间2D掩码]] 策略，以 0.5B 参数在 LIBERO 和 CALVIN 上超越多个 7B+ VLA。

---

## 核心贡献

1. **离散扩散动作专家 + Bridge Attention**: 通过对 VLM 各层隐藏状态进行层次化对齐注入，克服自回归动作生成的误差累积问题
2. **时序-空间 2D 掩码策略**: 将动作 token 矩阵（T×D）分两阶段掩码，同时建模跨时间帧依赖和维度间耦合
3. **小参数高性能**: 仅用 0.5B 参数，LIBERO 达到 95.7%，CALVIN 平均序列长度 4.19，超越多个 7B 量级模型

---

## 问题背景

### 要解决的问题

[[VLA（视觉-语言-动作模型）]] 中主流的**自回归动作 token 生成**存在两大缺陷：(1) 误差累积导致长序列动作失稳；(2) 自回归单向依赖无法充分建模动作维度间的耦合结构。

### 现有方法的局限

- 基于 VQ-VAE 的离散化（如部分 VLA 方法）引入强压缩，丢失精细时空信息
- 1D 掩码扩散只在一个维度（时间或动作维）上掩码，忽略另一维度的结构信息
- 多数高性能 VLA（[[π₀]] 3B、[[FlowVLA]] 8.5B）参数量大，推理成本高

### 本文的动机

将 [[离散扩散]] 与结构化 2D 掩码相结合：离散化动作 token 让去噪过程更加可控；2D 掩码同时强化时间和维度两个轴向的上下文感知；Bridge Attention 让 VLM 的层次化特征有效驱动动作生成。

---

## 方法详解

### 模型架构

![Figure 1: TS-Mask VLA 系统概览](https://arxiv.org/html/2607.09818v1/x1.png)

**说明**：输入为第三视角图像、夹爪图像、语言指令和 Action Query token，经 [[Qwen]] 2.5-0.5B 产生 L 层隐藏状态 $\mathbf{H}$，注入到对齐深度的离散扩散动作专家，输出离散动作 token 序列。

TS-Mask VLA 采用 **VLM + 离散扩散专家并行** 架构：

- **输入**: 第三视角图像 + 夹爪图像 + 语言指令 $l$ + Action Query (AQ) token
- **Backbone**: [[Qwen]] 2.5-0.5B（配合 [[LoRA]] 微调）
- **多模态编码**: 视觉特征提取器（DINOv2 / SigLIP 风格）+ 文本 tokenizer
- **核心模块**: [[离散扩散]] 动作专家（含 [[Bridge Attention]]），层数与 VLM 对齐
- **动作表示**: [[动作分块|Action Chunking]]，$T$ 步 $\times$ $D$ 维，均匀量化为 256 个离散 bin
- **总参数**: ~0.5B

### 核心模块

#### 模块 1：Bridge Attention 条件注入

**设计动机**：利用 VLM 各层的层次化语义特征（低层几何/高层语义）条件化 [[离散扩散]] 动作专家，同时避免过强的任务条件在训练初期导致不稳定。

**三路信息流融合**：

- **Self tokens**: 动作 token 的自注意力，建模帧内动作依赖
- **Action Query (AQ) tokens**: 可学习的动作查询 token，提供以动作为中心的条件
- **Task tokens**: VLM 对应层的视觉-语言隐藏状态，提供任务约束

注意力权重计算：

$$
\alpha = \text{softmax}\!\left(\frac{[\, Q K_{\text{self}}^\top,\; Q K_{\text{AQ}}^\top,\; \tanh(g)\, Q K_{\text{task}}^\top \,]}{\sqrt{d_h}}\right)
$$

Bridge 输出：

$$
Z_{\text{bridge}} = \alpha \, [V_{\text{self}};\; V_{\text{AQ}};\; V_{\text{task}}]
$$

**含义**：三路 key-value 沿 key 维度拼接后联合 softmax 归一化；$\tanh(g)$ 是可学习门控，在训练初期自动压制任务条件强度，随训练稳定后逐渐开放。

**符号说明**：

- $Q, K_{\cdot}, V_{\cdot}$：来自 self / AQ / task 三路的查询、键、值矩阵
- $d_h$：注意力头维度
- $g$：可学习标量门控参数

#### 模块 2：时序-空间 2D 掩码策略（[[时序-空间2D掩码]]）

**设计动机**：1D 掩码（仅按时间或仅按维度）忽略动作的另一结构轴，导致长序列任务中跨帧一致性较弱。

**两阶段掩码过程**，如下图所示：

![Figure 2: 时序-空间 2D Token 掩码](https://arxiv.org/html/2607.09818v1/x2.png)

将离散动作 token 重塑为 2D 矩阵 $\mathbf{A} \in \{0, \ldots, V-1\}^{T \times D}$：

1. **时间掩码（Temporal Masking）**：随机选择 $\lceil r \cdot T \rceil$ 个时间帧，将这些帧中所有 $D$ 个动作维度全部掩码
2. **空间掩码（Spatial Masking）**：在剩余未掩码帧中，每帧随机掩码 $\lceil r \times D \rceil$ 个 token

**掩码比例余弦调度**：

$$
r = \cos\!\left(\frac{\pi t}{2}\right), \quad t \sim \text{Uniform}(0, 1)
$$

**含义**：$t=0$ 时 $r=1$（全掩码），$t=1$ 时 $r=0$（无掩码），余弦调度使低掩码率（高信噪比）样本更多，有利于学习精细动作。

---

## 关键公式

### 公式 1：[[Qwen|VLM 隐藏状态]]提取

$$
\mathbf{H} = \{ H^{(l)} \}_{l=1}^{L}, \quad H^{(l)} \in \mathbb{R}^{N \times d}
$$

**含义**：VLM 的 $L$ 层 transformer 各自输出 $N$ 个 token 的 $d$ 维隐藏状态矩阵，后续逐层注入对应的动作专家层。

**符号说明**：

- $L$：VLM 层数（与动作专家层数对齐）
- $N$：token 数量
- $d$：隐藏维度

### 公式 2：[[动作分块|均匀动作量化]]

$$
\hat{a}_i \in [-1, 1] \xrightarrow{\text{量化}} q_i \in \{0, \ldots, 255\}
$$

量化边界：

$$
e_j = -1 + \frac{2j}{256}, \quad j = 0, \ldots, 256
$$

整体离散序列：$\mathbf{q} \in \{0, \ldots, 255\}^M$，其中 $M = T \times D$

**含义**：将连续动作值均匀量化为 256 个 bin，避免 VQ-VAE 的强压缩导致精细时空信息丢失。

### 公式 3：[[时序-空间2D掩码|掩码去噪训练损失]]

$$
\mathcal{L}_{\text{mask}} = \sum_{i \in \mathcal{M}} -\log p_\theta(a_i \mid \tilde{a}_{\setminus \mathcal{M}},\, \mathbf{H})
$$

**含义**：对所有被掩码位置的 token 做交叉熵预测，条件为未被掩码的动作 token $\tilde{a}_{\setminus \mathcal{M}}$ 以及 VLM 隐藏状态 $\mathbf{H}$。

**符号说明**：

- $\mathcal{M}$：被掩码 token 的索引集合
- $\tilde{a}_{\setminus \mathcal{M}}$：可见（未掩码）的动作 token
- $p_\theta$：动作专家的预测分布

### 公式 4：[[行为克隆|Step Unroll 联合训练目标]]

$$
\mathcal{L} = \frac{\mathcal{L}_{\text{mask}} + \lambda \, \mathcal{L}_{\text{unroll}}}{1 + \lambda}
$$

**含义**：掩码去噪损失与步骤展开（Step Unroll）损失的加权组合，$\mathcal{L}_{\text{unroll}}$ 监督推理时中间去噪步骤的一致性，起正则化作用。

**符号说明**：

- $\lambda$：步骤展开损失权重（消融实验最优值 $\lambda=0.5$）
- $\mathcal{L}_{\text{mask}}$：主掩码去噪损失
- $\mathcal{L}_{\text{unroll}}$：步骤展开正则损失

### 公式 5：[[离散扩散|推理迭代 ReMask 调度]]

$$
m_i = \left\lceil \gamma\!\left(\frac{i}{I}\right) \cdot |\Omega_i| \right\rceil
$$

**含义**：第 $i$ 次推理迭代中，从当前可见集合 $\Omega_i$ 中重新掩码 $m_i$ 个低置信度 token；$\gamma(\cdot)$ 为余弦调度，控制迭代过程中的重掩码比例，实现从全掩码到无掩码的渐进精化。

**符号说明**：

- $I$：推理总迭代次数
- $\Omega_i$：第 $i$ 轮的候选重掩码集合
- $\gamma(\cdot)$：余弦调度函数

---

## 关键图表

### Figure 3: 推理流程

![Figure 3: TS-Mask VLA 推理流程](https://arxiv.org/html/2607.09818v1/x3.png)

**说明**：从全掩码动作序列出发，每轮迭代中模型预测所有 masked token 的概率分布，保留高置信度预测，对低置信度 token 重新掩码（ReMask），迭代 $I$ 次后得到完整动作序列。

### Figure 4: 评测环境

![Figure 4: 仿真与真实评测环境](https://arxiv.org/html/2607.09818v1/x4.png)

**说明**：包含 LIBERO（四个任务套件）和 CALVIN（ABC→D 跨域设置），以及真实机器人（UR5e）三类任务。

### Figure 5: 真实世界实验结果

![Figure 5: 真实世界设置与结果](https://arxiv.org/html/2607.09818v1/x5.png)

**说明**：UR5e 机器人臂在放置苹果、拉取纸巾、翻转平底锅三类任务上均优于 OpenVLA-OFT 和 π₀ 基线。

### Table 1: LIBERO 基准对比（成功率 %）

| 方法 | 参数量 (B) | Spatial | Object | Goal | Long | **Avg.** |
|------|-----------|---------|--------|------|------|---------|
| FlowVLA | 8.5 | 93.2 | 95.0 | 91.6 | 72.6 | 88.1 |
| OpenVLA | 7 | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| CoT-VLA | 7 | 87.5 | 91.6 | 87.6 | 69.0 | 81.1 |
| ThinkAct | 7 | 88.3 | 91.4 | 87.1 | 70.9 | 84.4 |
| [[π₀]] | 3 | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| π₀-FAST | 3 | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| SmolVLA | 2.2 | 93.0 | 94.0 | 91.0 | 77.0 | 88.8 |
| GR00T N1 | 2 | 94.4 | 97.6 | 93.0 | 90.6 | 93.9 |
| Seer† | 0.57 | — | — | — | 78.7 | 78.7 |
| VLA-OS | 0.5 | 87.0 | 96.5 | 92.7 | 66.0 | 85.6 |
| Diffusion Policy† | — | 78.3 | 92.5 | 68.3 | 50.5 | 72.4 |
| VLA-Adapter-Pro* | 0.5 | 95.0 | 99.0 | 94.0 | 80.8 | 92.2 |
| **TS-Mask VLA** | **0.5** | **95.4** | **99.4** | **96.2** | **91.6** | **95.7** |

**关键发现**：TS-Mask VLA 以 0.5B 参数超越所有方法，尤其在长程任务（Long）上以 91.6% 超越 [[π₀]] 的 85.2%（+6.4pp），展现了 2D 掩码对长序列建模的优势。

### Table 2: CALVIN ABC→D 对比（成功率 % 与平均序列长度）

| 方法 | 参数量 (B) | 1 Task | 2 Tasks | 3 Tasks | 4 Tasks | 5 Tasks | Avg. Len |
|------|-----------|--------|---------|---------|---------|---------|---------|
| UniVLA | 7 | 95.5 | 85.8 | 75.4 | 66.9 | 56.5 | 3.80 |
| OpenVLA | 7 | 91.3 | 77.8 | 62.0 | 52.1 | 43.5 | 3.27 |
| [[OpenVLA-OFT]] | 7 | 96.3 | 89.1 | 82.4 | 75.8 | 66.5 | 4.10 |
| RoboDual | 7 | 94.4 | 82.7 | 72.1 | 62.4 | 54.4 | 3.66 |
| OpenHelix | 7 | 97.1 | 91.4 | 82.8 | 72.6 | 64.1 | 4.08 |
| ReconVLA | 7 | 95.6 | 87.6 | 76.9 | 69.3 | 64.1 | 3.95 |
| DeeR | 3 | 86.2 | 70.1 | 51.8 | 41.5 | 30.4 | 2.82 |
| RoboFlamingo | 3 | 82.4 | 61.9 | 46.6 | 33.1 | 23.5 | 2.48 |
| SuSIE | 1.3 | 87.0 | 69.0 | 49.0 | 38.0 | 26.0 | 2.69 |
| MoDE† | 0.44 | 96.2 | 88.9 | 81.1 | 71.8 | 63.5 | 4.01 |
| Seer† | 0.32 | 94.4 | 87.2 | 79.9 | 72.2 | 64.3 | 3.98 |
| **TS-Mask VLA** | **0.5** | **97.4** | **92.5** | **85.2** | **77.0** | **66.9** | **4.19** |

**关键发现**：CALVIN 上 TS-Mask VLA 平均序列长度 4.19，超越 7B 级别的 [[OpenVLA-OFT]]（4.10）和 OpenHelix（4.08）；4-Task 完成率 77.0% 对比 OpenVLA 的 52.1%（+24.9pp）。

### Table 3: 掩码策略消融（LIBERO 成功率 %）

| 掩码策略 | Spatial | Object | Goal | Long |
|----------|---------|--------|------|------|
| 1D 掩码 | 94.6 | 98.4 | 95.2 | 85.0 |
| **2D 掩码**（TS-Mask VLA） | **95.4** | **99.4** | **96.2** | **91.6** |

**关键发现**：2D 掩码在 Long 任务上提升最显著（+6.6pp），印证了时序-空间结构建模对长程任务的关键作用。

### Table 4: Step Unroll 强度消融（LIBERO-Spatial 成功率 %）

| 配置 | 成功率 |
|------|--------|
| 无 Unroll（λ=0） | 90.8% |
| **λ=0.5（最优）** | **95.4%** |
| λ=1.0 | 91.5% |

**关键发现**：适度的步骤展开正则（λ=0.5）提升 +4.6pp；过强（λ=1.0）反而损害性能，存在过正则化问题。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[LIBERO]] | 4 套件（Spatial/Object/Goal/Long） | 长程操作序列、多任务 | 训练 + 测试 |
| [[CALVIN]] | ABC→D 跨域设置 | 零样本跨域迁移、序列任务 | 测试 |
| 真实机器人 | 3 类任务，各 20 次 | 放苹果、拉纸巾、翻平底锅 | 真实世界验证 |

### 实现细节

- **Backbone**: [[Qwen]] 2.5-0.5B（[[LoRA]] 微调）
- **视觉编码器**: DINOv2 / SigLIP 风格
- **动作量化**: $V=256$ 均匀 bin，$T=8$（动作 chunk 长度），$D=7$（末端执行器 DoF）
- **掩码调度**: 余弦调度 $r = \cos(\pi t/2)$，$t \sim \text{Uniform}(0,1)$
- **Step Unroll 权重**: $\lambda=0.5$
- **硬件**: 单张 NVIDIA RTX 4090
- **真实机器人**: 6-DoF UR5e + ROBOTIQ-85 夹爪，两台 Intel RealSense D435i 相机（固定全局视角 + 手腕视角）

### 可视化结果

2D 掩码显著提升 Long 任务的跨帧一致性；真实世界实验中，TS-Mask VLA 在空间理解和多样场景泛化上均优于 [[OpenVLA-OFT]] 和 [[π₀]]。

---

## 批判性思考

### 优点

1. **参数效率突出**: 0.5B 参数超越多个 7B+ 方法，论证了结构化掩码策略的有效性
2. **方法设计优雅**: 2D 掩码策略直觉清晰，同时建模时序和空间两个轴向，与动作数据的内在结构对齐
3. **Bridge Attention 设计合理**: 层次化 VLM 特征注入 + 可学习门控，在训练稳定性和条件利用之间取得平衡

### 局限性

1. **仅在资源受限场景下评测**: 论文明确指出只在单卡 RTX 4090 条件下验证，未探索更多数据或更大模型的表现
2. **动作量化精度上限**: 256 bin 均匀量化对高精度操作可能不够精细，且量化误差不可消除
3. **推理迭代开销**: 迭代去噪推理（多步 ReMask）相比单步自回归在时延上有劣势，实时性未充分讨论

### 潜在改进方向

1. 探索非均匀/自适应量化（如 VQ-VAE）与 2D 掩码的结合
2. 减少推理迭代步数（蒸馏或一致性训练）以满足实时控制需求
3. 在更大规模训练数据（Open-X 等）上验证扩展性

### 可复现性评估

- [ ] 代码开源（论文中未提供 GitHub 链接）
- [ ] 预训练模型（未发布）
- [x] 训练细节完整（单卡、LoRA、动作 chunk=8、量化 bin=256、λ=0.5 均有明确说明）
- [x] 数据集可获取（LIBERO、CALVIN 均为公开基准）

---

## 关联笔记

### 基于

- [[π₀]]: 主要对比基线，flow-based VLA
- [[OpenVLA-OFT]]: 对比基线，7B 规模 VLA
- [[扩散策略]]: 底层方法，离散扩散的基础
- [[Qwen]]: 使用 Qwen2.5-0.5B 作为 VLM Backbone

### 对比

- [[FlowVLA]]: 8.5B 参数，88.1% LIBERO，远大于本文
- [[SmolVLA]]: 2.2B，同属小参数路线，88.8% LIBERO
- [[GR00T N1]]: 2B，93.9% LIBERO

### 方法相关

- [[离散扩散]]: 核心生成机制
- [[Bridge Attention]]: 关键条件注入模块
- [[时序-空间2D掩码]]: 核心掩码策略
- [[动作分块]]: 输出动作表示方式
- [[LoRA]]: VLM 微调技术

### 数据集相关

- [[LIBERO]]: 主要评测基准
- [[CALVIN]]: 跨域泛化评测

---

## 速查卡片

> [!summary] TS-Mask VLA
> - **核心**: 离散扩散 + Bridge Attention + 时序-空间 2D 掩码，0.5B 参数实现 SOTA 操作性能
> - **方法**: VLM 层次化隐状态注入 → 两阶段 2D 掩码去噪 → 迭代 ReMask 推理
> - **结果**: LIBERO 95.7%（超越 π₀ 的 94.2%）；CALVIN 平均序列长度 4.19（超越所有 7B 方法）
> - **代码**: 未开源

---

*笔记创建时间: 2026-07-15*
