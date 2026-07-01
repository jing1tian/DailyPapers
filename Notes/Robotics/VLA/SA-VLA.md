---
title: "SA-VLA: State-aware tokenizer for improving Vision-Language-Action Models' performance"
method_name: "SA-VLA"
authors: [Tengyue Jiang, Chunpu Xu, Jiayue Kang, Yao Mu]
year: 2026
venue: arXiv
tags: [vla, action-tokenization, proprioceptive-state, vector-quantization, robot-manipulation]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2606.30113
created: 2026-07-01
---

# 论文笔记：SA-VLA: State-aware tokenizer for improving Vision-Language-Action Models' performance

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Shanghai Jiao Tong University, East China University of Science and Technology, Hong Kong Polytechnic University, Xi'an University of Electronic Science and Technology |
| 日期 | June 2026 |
| 项目主页 | — |
| 对比基线 | [[FAST]], [[VQ-BeT]], [[OpenVLA]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.30113) / Code: — |

---

## 一句话总结

> SA-VLA 引入状态感知的动作 tokenizer，将机器人本体感知状态条件化到动作 token 的编解码过程，使同一 token 在不同关节构型下对应不同连续控制，仿真平均成功率从 0.29 提升至 0.56。

---

## 核心贡献

1. **状态感知 Tokenizer**: 提出将[[机器人状态|本体感知状态]]条件化到动作编解码过程，打破固定 codebook 映射的限制，使相同 token 在不同关节构型下产生不同连续动作
2. **两种实现机制**: 探索交叉注意力（Method A）和轻量级 Adapter（Method B）两种状态注入方式，Method B 通过预测状态调制因子 $w$ 实现更细粒度的动作空间覆盖
3. **零样本 Sim-to-Real 迁移**: 无需真实数据微调，在真实世界实验中平均成功率从 0.15 提升至 0.33，验证了方法的泛化能力

---

## 问题背景

### 要解决的问题

[[视觉语言动作模型|VLA]] 的动作 tokenizer 通常将每个离散 token 映射到固定的动作原型，忽略机器人当前的[[机器人状态|本体感知状态]]（关节角、末端执行器位置等）。这导致"同一 action token 在不同关节构型下需要不同连续控制"的问题，即 token 表达能力不足。

### 现有方法的局限

- **Binning（[[OpenVLA]]）**: 均匀离散化，完全忽略动作之间的关联性与状态依赖性
- **[[FAST]]**: 频域变换 + BPE 压缩，减少 token 数量但仍为固定映射
- **[[VQ-BeT]]**: [[VQ-VAE]] 离散化 + Transformer，codebook 条目与状态无关，有效编码空间受限

### 本文的动机

相同关节角 delta 在不同初始状态下对应完全不同的实际运动效果，因此动作 token 的解码应当以当前机器人状态为条件。通过状态调制，可以在相同 codebook 大小下覆盖更大的有效动作空间（不同状态下 codebook 条目被"拉伸"到不同区域）。

---

## 方法详解

### 模型架构

SA-VLA 采用**两阶段训练**架构：
- **Stage 1**: 独立训练状态感知动作 tokenizer（[[VQ-VAE]] 框架）
- **Stage 2**: 将 tokenizer 集成到[[视觉语言动作模型|VLA]] 中进行端到端微调

**输入**: [[视觉语言模型|VLM]] 输出隐变量 + 观测 $o_t$（RGB 图像）+ [[机器人状态]] $s_t$ + 语言指令 $L$  
**图像 Backbone**: [[SigLIP]]-SO400M-patch14-224（16×16 patch grid = 256 tokens/帧）  
**语言 Backbone**: 基础 LLM（保持原生 tokenizer）  
**状态编码**: 每个关节维度离散化为 256 个 bin  
**输出**: [[动作分块|动作块]] $a_{t:t+k}$，支持[[自回归语言模型|自回归]]和并行两种解码策略

### 核心模块

#### 模块1: 状态感知动作 Tokenizer（Method A — 交叉注意力）

**设计动机**: 利用[[交叉注意力]]将本体感知状态信息融入动作特征，使量化过程感知当前机器人构型。

**具体实现**:
- 编码器将动作序列 $a \in \mathbb{R}^{R \times d}$ 映射到中间特征 $Z \in \mathbb{R}^{R \times d}$
- Transformer 的两个层中，**keys 和 values 均来自状态输入** $s_t$，queries 来自动作特征
- 量化层从 codebook $\hat{Z}$ 中查找最近邻：$q_i = \arg\min_j \|z_i - \hat{z}_j\|_2$
- 解码器基于量化特征 $q(Z)$ 重建连续动作

#### 模块2: 状态感知动作 Tokenizer（Method B — 轻量级 Adapter，效果更优）

**设计动机**: 通过预测状态依赖的缩放因子 $w$，动态调整动作空间的"感受野"，使相同 codebook 条目在不同状态下对应不同的连续动作范围。

**具体实现**:
- 一个轻量级 MLP + Sigmoid 组成的 Adapter，输入状态 $s_t$，预测逐维缩放因子 $w \in \mathbb{R}^d$
- **标准化**: 将原始动作除以缩放因子：$a_{\text{trans}} = a \div w$
- **编码量化**: 对 $a_{\text{trans}}$ 进行 VQ-VAE 编码量化，得到 token 序列
- **解码还原**: 解码后的 $\hat{a}_{\text{trans}}$ 乘回缩放因子：$\hat{a} = \hat{a}_{\text{trans}} \times w$
- **核心效果**: 相同 codebook 条目在不同状态下被映射到不同的动作范围，等效于有更大有效 codebook 容量

**设计优势对比 Method A**:
- 无需在 Transformer 层引入[[交叉注意力]]，计算开销极小
- 缩放因子直接作用于动作空间，物理含义更清晰
- 允许 codebook 条目在大范围动作空间内有效分布

### VLA 集成（Section 3.3）

**Token 化策略**:
- 文本：原生 LLM tokenizer
- 状态：每关节维度均匀离散为 256 bins
- 图像：SigLIP patch tokens（256 tokens/帧）
- 动作：SA-VLA tokenizer，配合特殊边界 token

**两种解码策略**:
1. **自回归解码（AR）**: 依次预测每个动作 token，考虑历史 token 序列
2. **并行解码（PD）**: 所有动作 token 同时预测，速度更快

---

## 关键公式

### 公式1: [[VQ-VAE|向量量化]] 最近邻查找

$$
q_i = \arg\min_j \|z_i - \hat{z}_j\|_2
$$

**含义**: 对编码器输出的每个特征向量 $z_i$，在 codebook $\{\hat{z}_j\}$ 中查找欧氏距离最近的条目作为量化结果。

**符号说明**:
- $z_i \in \mathbb{R}^d$: 编码器对第 $i$ 个动作步的输出特征
- $\hat{z}_j \in \mathbb{R}^d$: codebook 中第 $j$ 个条目（可学习）
- $q_i$: 量化后的特征（离散 token）

### 公式2: [[VQ-VAE]] 训练损失

$$
\mathcal{L} = \|\hat{a} - a\|_2^2 + \lambda_1 \cdot \|sg(x) - q(x)\|_2^2 + \lambda_2 \cdot \|x - sg(q(x))\|_2^2
$$

**含义**: 三项损失之和：重建损失 + codebook 学习损失 + commitment 损失，后两项通过[[停止梯度|stop-gradient]] 操作解耦编码器和 codebook 的训练。

**符号说明**:
- $\hat{a}$: 解码器重建的连续动作
- $a$: 真实动作
- $sg(\cdot)$: stop-gradient 操作，阻止梯度回传
- $x$: 编码器输出特征
- $q(x)$: 量化后特征（codebook 条目）
- $\lambda_1, \lambda_2$: 损失权重系数

### 公式3: [[自回归语言模型|自回归]]解码目标

$$
\mathcal{L}_{AD} = -\sum_{i=1}^{h} \sum_{r=1}^{R} \log P(q_{i,r} | q_{<}, o_t, s_t, L)
$$

**含义**: 自回归预测每个动作 token，以历史 token、观测、状态和语言指令为条件。

**符号说明**:
- $h$: 动作 chunk 的 horizon 长度
- $R$: 每步动作的 token 数（codebook 级数）
- $q_{i,r}$: 第 $i$ 步第 $r$ 个 token
- $q_{<}$: 历史动作 token 序列
- $o_t, s_t, L$: 当前观测、状态、语言指令

### 公式4: 并行解码目标

$$
\mathcal{L}_{PD} = -\sum_{i=1}^{h} \sum_{r=1}^{R} \log P(q_{i,r} | o_t, s_t, L)
$$

**含义**: 所有动作 token 条件独立地从观测、状态和语言指令中并行预测，无历史 token 依赖，速度更快但表达能力略弱于自回归。

**符号说明**:
- 符号同公式3，区别在于条件不包含 $q_{<}$

### 公式5: Method B 动作标准化与还原

$$
a_{\text{trans}} = a \div w, \quad \hat{a} = \hat{a}_{\text{trans}} \times w
$$

**含义**: 用状态条件的缩放因子 $w$ 将动作归一化后编码，解码后再还原，实现动作空间的动态调整。

**符号说明**:
- $w \in \mathbb{R}^d$: MLP Adapter 输出的逐维缩放因子（Sigmoid 激活，恒正）
- $a_{\text{trans}}$: 经过状态归一化的动作
- $\hat{a}_{\text{trans}}$: 解码器输出的归一化动作
- $\hat{a}$: 最终还原的连续动作

---

## 关键图表

### Figure 1: Tokenizer 架构对比

![Figure 1 — Tokenizer Architecture](https://arxiv.org/html/2606.30113v1/tokenizer.png)

**说明**: 上方展示 **Method A（交叉注意力）**：在 Transformer 编码器层中，状态信息作为 Key 和 Value 参与动作特征的注意力计算。下方展示 **Method B（轻量级 Adapter）**：MLP Adapter 接收状态输入，预测缩放因子 $w$，对动作进行标准化后再编码量化，解码时乘回缩放因子还原。

### Figure 2: VLA 整体架构

![Figure 2 — VLA Architecture](https://arxiv.org/html/2606.30113v1/vla.png)

**说明**: SA-VLA 集成到基于 LLM 的 VLA 框架中，输入包含图像 token（SigLIP 编码）、状态 token（bin 离散化）和语言指令，输出动作 token 序列（支持自回归和并行两种解码模式）。

### Figure 3: RoboTwin 仿真任务

![Figure 3 — RoboTwin Simulation Tasks](https://arxiv.org/html/2606.30113v1/robotwin_task.png)

**说明**: RoboTwin 基准的部分任务可视化，包括 move_pillbottle_pad、place_burger_fries、click_bell 等 12 个双臂操作任务，涵盖精细抓取、放置和交互等不同难度。

### Figure 4: 真实世界实验场景

![Figure 4 — Real World Task Scenarios](https://arxiv.org/html/2606.30113v1/real_world_scenes.png)

**说明**: 三个真实世界任务场景：Click Bell（敲铃）、Place Container（放置容器）、Pick Bottles（抓取瓶子），每个任务进行 20 次试验验证零样本迁移性能。

### Figure 5: Tokenization 粒度可视化

![Figure 5 — Tokenization Granularity](https://arxiv.org/html/2606.30113v1/1.png)

**说明**: 对四个相似动作的余弦相似度比较。无状态方法的 token 余弦相似度差异仅 0.001（接近相同），而 SA-VLA 能将不同状态下的相似动作映射到区分度更高的 codebook 条目，证明状态感知带来的细粒度表达能力提升。

### Table 1: 仿真实验主要结果（RoboTwin，12 任务平均）

| Tokenizer | Beat Block | Move Card | Click Bell | Avg (全12任务) |
|-----------|-----------|-----------|-----------|----------------|
| Binning | 0.11 | 0.01 | 0.67 | 0.24 |
| FAST | 0.00 | 0.18 | 0.68 | 0.17 |
| VQ-BET | 0.10 | 0.27 | 0.64 | 0.29 |
| **Method A (AR)** | 0.25 | 0.47 | 0.83 | 0.55 |
| **Method B (AR)** | **0.16** | **0.60** | **0.90** | **0.56** |
| **Method B (PD)** | **0.29** | **0.63** | **0.89** | **0.56** |

**关键发现**: Method B 与 Method A 性能相当（均 0.55/0.56），但 Method B 参数更少、实现更简洁。并行解码（PD）与自回归解码（AR）性能基本持平，在部分任务上 PD 略优。

### Table 2: 消融实验——状态信息的影响

| 配置 | 解码方式 | 平均成功率 |
|------|----------|------------|
| w/o state | PD | 0.43 |
| w/o state | AR | 0.51 |
| Method A (state-aware) | AR | 0.55 |
| **Method B (state-aware)** | **AR** | **0.56** |

**关键发现**: 引入状态信息后，AR 模式从 0.51 提升至 0.55/0.56（+8-10%），PD 模式从 0.43 提升至 0.56（+30%），状态信息对并行解码的提升更显著。

### Table 3: 真实世界零样本迁移（每任务 20 次试验）

| Tokenizer | Click Bell | Place Container | Pick Bottles | 平均成功率 |
|-----------|-----------|-----------------|--------------|------------|
| Binning | 6/20 | 0/20 | 0/20 | 0.10 |
| FAST | 4/20 | 1/20 | 0/20 | 0.08 |
| VQ-BET | 7/20 | 2/20 | 0/20 | 0.15 |
| **Ours (AR)** | **10/20** | **7/20** | **3/20** | **0.33** |

**关键发现**: SA-VLA 在零样本 sim-to-real 迁移中将平均成功率从 0.15 提升至 0.33（+120%），在 Place Container 任务上从 10% 提升至 35%，证明状态感知显著提升了跨域泛化能力。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[RoboTwin]] | 19,200 条轨迹，12 任务 | 双臂操作仿真平台，自动生成数据 | Stage 1 + Stage 2 训练 |
| 真实世界（自采） | 3 任务各 20 次试验 | 无额外真实数据，纯零样本迁移 | 测试 |

### 实现细节

- **Backbone**: SigLIP-SO400M-patch14-224（图像编码），基础 LLM（语言）
- **Stage 1（Tokenizer 训练）**:
  - Batch size: 1024
  - Epochs: 200
  - Learning rate: $5 \times 10^{-5}$
  - Optimizer: [[AdamW]]（余弦退火调度）
- **Stage 2（VLA 微调）**:
  - Batch size: 64
  - Epochs: 10
  - Learning rate: $1 \times 10^{-4}$
  - Optimizer: [[AdamW]]
- **硬件**: RTX 4090 GPU（单卡）
- **状态离散化**: 每关节维度 256 bins

### 可视化结果

仿真实验中，SA-VLA 在精细操作任务（如 Move Card）上提升最显著（0.27→0.60），而在已经较简单的任务（如 Click Bell）上基线已有较高成功率，提升空间相对有限。真实世界中 Pick Bottles 任务最具挑战性，基线均为 0 成功，SA-VLA 达到 3/20。

---

## 批判性思考

### 优点
1. **思路简洁有效**: Method B 仅需一个轻量级 MLP Adapter 即可将状态信息注入 tokenizer，额外参数极少
2. **即插即用**: 两种机制均支持直接集成到现有 LLM-based VLA 框架，兼容自回归和并行解码
3. **零样本迁移强**: 在未见真实数据的情况下，sim-to-real 成功率翻倍，实用价值高

### 局限性
1. **数据规模有限**: 仅在 19,200 条轨迹（12 任务）上验证，未在更大规模数据集（如 Open-X）上测试
2. **架构局限**: 仅探索 VQ-VAE 框架，未考虑基于[[扩散模型]]的 tokenizer
3. **场景局限**: 仅在桌面操作双臂机器人上测试，未验证灵巧手、移动操作等更复杂场景
4. **缺少消融**: 缺少对 Adapter 结构（层数、隐层大小）的消融，以及 codebook 大小的敏感性分析

### 潜在改进方向
1. 与扩散策略结合，探索连续动作生成中的状态条件化
2. 在 Open-X Embodiment 等大规模数据集上验证跨机器人泛化
3. 探索层次化状态感知（末端执行器状态 vs 全身关节状态的不同贡献）

### 可复现性评估
- [ ] 代码开源（论文中未提供代码链接）
- [ ] 预训练模型（未提供）
- [x] 训练细节完整（超参数、数据集规模均有说明）
- [x] 数据集可获取（RoboTwin 为开放平台）

---

## 关联笔记

### 基于
- [[VQ-VAE]]: 核心量化框架，SA-VLA 基于 VQ-VAE 扩展状态感知能力
- [[视觉语言动作模型]]: SA-VLA 为 VLA 的 tokenizer 组件

### 对比
- [[FAST]]: 频域动作 tokenizer，SA-VLA 的主要基线之一（+39% 改进）
- [[VQ-BeT]]: 向量量化行为 Transformer，SA-VLA 的强基线（+27% 改进）
- [[OpenVLA]]: Binning 方案，最简单的离散化基线

### 方法相关
- [[交叉注意力]]: Method A 的核心机制
- [[停止梯度]]: VQ-VAE 训练中解耦编码器与 codebook 的关键技巧
- [[动作分块]]: SA-VLA 预测的动作表示形式
- [[机器人状态]]: 状态感知 tokenizer 的条件输入

### 硬件/数据相关
- [[RoboTwin]]: 主要评测 benchmark，12 个双臂操作任务

---

## 速查卡片

> [!summary] SA-VLA
> - **核心**: 状态感知动作 tokenizer，将关节状态条件化到 VQ-VAE 编解码，解决同一 token 对应不同运动的歧义问题
> - **方法**: Method B（MLP Adapter 预测缩放因子 $w$，归一化动作后编码再还原）效果最佳
> - **结果**: RoboTwin 平均成功率 0.29→0.56，零样本 sim-to-real 0.15→0.33
> - **代码**: 未开源

---

*笔记创建时间: 2026-07-01*
