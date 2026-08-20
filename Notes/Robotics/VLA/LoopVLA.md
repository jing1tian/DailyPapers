---
title: "LoopVLA: Learning Sufficiency in Recurrent Refinement for Vision-Language-Action Models"
method_name: "LoopVLA"
authors: [Boyang Shen, Kaixiang Yang, Hao Wang, Qiuyu Yu, Qiang Xie, Qiang Li, Zhiwei Wang]
year: 2026
venue: arXiv
tags: [vla, recurrent-refinement, early-exit, parameter-efficient, robot-manipulation, embodied-ai, transformer, action-prediction]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2605.09948v2
created: 2026-08-20
---

# 论文笔记：LoopVLA: Learning Sufficiency in Recurrent Refinement for Vision-Language-Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 未公开 |
| 日期 | May 2026（修订 August 2026）|
| 项目主页 | N/A |
| 对比基线 | [[Pi0\|π₀]], [[Qwen3OFT]], [[GR00T N1.5]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.09948) / Code: N/A |

---

## 一句话总结

> LoopVLA 通过共享 Transformer 块的循环精炼，让 VLA 模型学会"何时停止推理"，在参数减少 45% 的同时维持甚至超越最先进性能。

---

## 核心贡献

1. **循环精炼框架（Loop Block）**: 用共享权重的循环 [[Transformer]] 块替代固定深度的多层 Transformer，实现渐进式多模态表示精炼，参数量减少 45%。
2. **充分性估计双头设计（Sufficiency Head）**: 在每次迭代中同步预测动作和"是否已足够"的置信分数，基于[[剩余质量分配|Remaining Mass Allocation (RMA)]] 确保概率归一化。
3. **两阶段自监督训练**: Stage 1 监督所有中间预测 + 正则化；Stage 2 冻结模型、用 [[KL散度|KL Divergence]] 对齐置信分布与动作质量分布，无需外部标注。
4. **推理效率提升**: 自适应早退策略（LoopOFTα）实现高达 1.7× 推理吞吐，最优选择策略（LoopOFT\*）在 LIBERO 上达到 96.0% 成功率，超越更大模型。

---

## 问题背景

### 要解决的问题

现有 [[Vision-Language-Action Model|VLA]] 模型（如 [[Pi0|π₀]]、[[OpenVLA]]）依赖固定深度的 [[Transformer]] 主干网络，对所有输入使用相同的表示提取深度，无法根据任务难度动态调整计算量。

### 现有方法的局限

- **固定深度问题**: 深层表示对高层语义有利，但对精确控制所需的低层几何线索可能造成过度抽象，浪费计算。
- **后处理早退方案**: 现有"启发式早退"方法（heuristic early stop）为事后补丁，未能让模型内生学习"表示是否足够"。
- **参数量大**: 主流 VLA（π₀: 4.0B, π₀.₅: 4.1B）参数量巨大，推理吞吐低（3.70 Hz / 3.11 Hz）。

### 本文的动机

通过让共享 [[Transformer]] 块循环处理相同输入，模型可以在多次迭代中渐进精炼表示，并通过 [[充分性估计|Sufficiency Estimation]] 主动学习"何时表示已足够用于动作预测"，从而在精度与计算量之间实现内容自适应的权衡。

---

## 方法详解

### 模型架构

LoopVLA 采用 **循环 Transformer（Looped Transformer）** 架构：

- **输入**: 语言指令 $\ell$ + 视觉观测 $o_t$，以及 Action tokens 和 Sufficiency tokens
- **Backbone**: 预训练 [[Vision-Language Model|VLM]]（Qwen 系列）产生初始表示 $h_t$，再通过感知锚层（Perceptual Anchor Layers）提取稳定视觉特征
- **核心模块**: [[循环块|Loop Block]] — 共享权重的 Transformer，通过[[因果注意力|Causal Attention]] 让 Action tokens 关注视觉/语言 tokens；[[充分性头|Sufficiency Head]] 通过[[交叉注意力|Cross-Attention]] 聚合所有信息产生停止概率
- **输出**: [[Action Chunking|动作块]] $A^{(n)}_{t:t+c}$（每次迭代都输出，最终选择置信最高的那次）
- **总参数**: 1.2B（Qwen3OFT 基座 2.2B 的 ~55%）

### 核心模块

#### 模块 1: Loop Block（循环块）

**设计动机**: 利用 [[权重共享|Weight Sharing]] 在多次迭代中渐进提炼 [[多模态表示|Multimodal Representation]]，无需线性增加参数。

**具体实现**:
- 共享同一组 [[Transformer]] 权重 $M$，迭代 $N$ 次（默认 $N=8$，块数 $L=3$，记为 3⊗8）
- 使用[[因果注意力掩码|Causal Attention Masking]]，确保 Action tokens 只能看到视觉/语言上下文
- 感知锚层（Perceptual Anchor Layers）在每次循环前提供稳定的低层视觉特征注入

#### 模块 2: Sufficiency Head（充分性估计头）

**设计动机**: 让模型主动估计当前迭代的表示是否已"足够"做出高质量动作预测，而非依赖启发式规则。

**具体实现**:
- Sufficiency tokens 通过 [[循环位置编码|Loop-Index Positional Encoding]] 感知当前迭代步数 $n$
- 通过[[交叉注意力|Cross-Attention]] 层聚合所有 Action tokens 的全局信息：$z^{(n)} = \text{CrossAttn}(\tilde{h}_{\text{suf}}^{(n)}, h_{\text{action}}^{(n)})$
- MLP + Sigmoid 输出停止概率 $s^{(n)}$

#### 模块 3: Remaining Mass Allocation（剩余质量分配，RMA）

**设计动机**: 确保所有迭代步骤的停止概率之和为 1，使其成为合法概率分布，可直接用于最终动作选择。

**具体实现**: 递归维护剩余质量 $r^{(n)}$（初始为 1），每步分配 $p^{(n)} = s^{(n)} \cdot r^{(n)}$，然后更新剩余质量（详见公式 6-7）。

---

## 关键公式

### 公式 1: [[Vision-Language Model|VLM 编码]]

$$
h_t = \mathrm{VLM}(o_t, \ell)
$$

**含义**: 将视觉观测 $o_t$ 和语言指令 $\ell$ 编码为初始多模态表示 $h_t$。

**符号说明**:
- $o_t$: 时刻 $t$ 的视觉观测（图像/视频帧）
- $\ell$: 语言指令
- $h_t$: 初始多模态隐状态

---

### 公式 2: [[Action Chunking|动作预测]]

$$
A_{t:t+c} = \pi_{\theta}(h_t)
$$

**含义**: 标准 VLA 从最深层表示一次性预测未来 $c$ 步动作块。LoopVLA 在每次迭代 $n$ 处均预测一次，通过充分性估计选择最优迭代。

---

### 公式 3: [[循环块|Loop Block 迭代更新]]

$$
h^{(n)} = \mathrm{LoopBlock}(h^{(n-1)}; M)
$$

**含义**: 共享权重 $M$ 的 Loop Block 将上一迭代表示 $h^{(n-1)}$ 精炼为 $h^{(n)}$，$N$ 次迭代共享同一套参数。

**符号说明**:
- $h^{(n)}$: 第 $n$ 次迭代后的隐状态
- $M$: 共享 Transformer 块的参数（[[权重共享]] 核心）
- $N$: 最大迭代次数（默认 8）

---

### 公式 4: [[循环位置编码|充分性 Token 位置编码]]

$$
\tilde{h}_{\text{suf}}^{(n)} = h_{\text{suf}}^{(n)} + \mathrm{PE}(n)
$$

**含义**: 给 Sufficiency tokens 注入迭代步索引 $n$ 的位置编码，使其感知"当前是第几次精炼"，为[[充分性估计]] 提供序数信息。

---

### 公式 5: [[充分性估计|停止概率（Halting Score）]]

$$
s^{(n)} = \sigma\!\left(\mathrm{MLP}(z^{(n)})\right)
$$

**含义**: 将聚合的充分性特征 $z^{(n)}$ 映射为本迭代的停止概率分量 $s^{(n)} \in (0,1)$。

**符号说明**:
- $z^{(n)}$: Cross-Attention 聚合后的充分性特征向量
- $\sigma$: Sigmoid 激活函数

---

### 公式 6 & 7: [[剩余质量分配|Remaining Mass Allocation (RMA)]]

$$
p^{(n)} = s^{(n)} \cdot r^{(n)}
$$

$$
r^{(n+1)} = r^{(n)} \cdot \left(1 - s^{(n)}\right)
$$

**含义**: 确保各迭代步骤的分配概率 $\{p^{(n)}\}$ 之和为 1，形成合法的概率质量分布，便于最优迭代选择。

**符号说明**:
- $p^{(n)}$: 第 $n$ 次迭代被选中的概率质量
- $r^{(n)}$: 第 $n$ 次迭代开始时的剩余概率质量（$r^{(1)} = 1$）
- $s^{(n)}$: 第 $n$ 次迭代的停止概率

---

### 公式 8: 中间动作预测

$$
A^{(n)} = \pi_{\theta}(h^{(n)})
$$

**含义**: 在每次迭代 $n$ 处，用当前精炼后的表示预测动作块，供充分性估计评分使用。

---

### 公式 9: [[Action Loss|动作监督损失]]

$$
\mathcal{L}_{\text{action}} = \sum_{n=1}^{N} \ell\!\left(A^{(n)}, \hat{A}\right)
$$

**含义**: 监督所有 $N$ 次迭代的中间动作预测，强迫模型在每个精炼深度都保持动作质量。

**符号说明**:
- $\hat{A}$: Ground-truth 动作序列
- $\ell(\cdot)$: 任务损失函数（如 L2 或 [[Flow Matching]] 损失）

---

### 公式 10: 正则化损失（熵 + 多样性）

$$
\mathcal{L}_{\text{ent}} = \sum_{n=1}^{N} p^{(n)} \log p^{(n)}, \quad
\mathcal{L}_{\text{div}} = \sum_{i < j} \max\!\left(0,\, \cos(z^{(i)}, z^{(j)})\right)
$$

**含义**:
- **熵正则化** $\mathcal{L}_{\text{ent}}$（最大化熵）：防止置信分布过早坍缩到某一固定迭代，鼓励模型探索不同精炼深度。
- **多样性正则化** $\mathcal{L}_{\text{div}}$：惩罚不同迭代步骤充分性特征 $z^{(n)}$ 之间的余弦相似性，阻止所有迭代学到相同表示（平凡解）。

---

### 公式 11: Stage 1 总损失

$$
\mathcal{L}_{\text{stage1}} = \mathcal{L}_{\text{action}} + \lambda_1 \mathcal{L}_{\text{ent}} + \lambda_2 \mathcal{L}_{\text{div}}
$$

**含义**: 联合精炼学习阶段的完整优化目标。

**符号说明**:
- $\lambda_1 = 0.001$：熵正则化权重
- $\lambda_2 = 0.01$：多样性正则化权重

---

### 公式 12: [[充分性校准|目标分布（质量导向）]]

$$
q^{(n)} = \mathrm{softmax}\!\left(-\ell\!\left(A^{(n)}, \hat{A}\right) / \tau\right)
$$

**含义**: 用各迭代步骤的动作损失（取负）经温度 $\tau$ Softmax 后得到目标分布 $q^{(n)}$——损失越低的迭代，目标分配概率越大。

**符号说明**:
- $\tau = 0.5$：温度超参数，控制分布尖锐程度
- $q^{(n)}$：由动作质量导出的目标停止概率分布

---

### 公式 13: Stage 2 充分性校准损失

$$
\mathcal{L}_{\text{stage2}} = \mathrm{KL}(q \| p)
$$

**含义**: 冻结主模型，仅训练 [[充分性头|Sufficiency Head]]，使其预测分布 $p$ 对齐质量导出分布 $q$。用 [[KL散度|KL Divergence]] 衡量两个分布的差距。

---

### 公式 14: 最优选择推理策略（LoopOFT\*）

$$
n^* = \arg\max_n\, p^{(n)}, \quad A = A^{(n^*)}
$$

**含义**: 运行所有 $N$ 次迭代，选择置信概率最高的迭代步骤的动作。追求最优精度，不跳过迭代。

---

### 公式 15 & 16: 自适应早退策略（LoopOFTα）

$$
\sum_{k=1}^{n} p^{(k)} \geq \theta \quad \text{或} \quad \max_{k \leq n} p^{(k)} \geq r^{(n+1)}
$$

$$
n^* = \arg\max_{k \leq n} p^{(k)}, \quad A = A^{(n^*)}
$$

**含义**: 满足任一停止条件即提前退出迭代，在精度和计算效率间取得自适应平衡（$\theta = 0.68$）。

---

## 关键图表

### Figure 1: 范式对比

![Figure 1 — Comparison of Intermediate Representation Paradigms](https://arxiv.org/html/2605.09948v2/overview_7.png)

**说明**: 对比三种中间表示范式：(1) 后处理方法（post-hoc）、(2) 启发式早退方法、(3) LoopVLA 的共享循环方法。LoopVLA 在所有迭代步骤主动学习充分性，而非事后添加判断。

---

### Figure 2: Sufficiency Head 架构详图

![Figure 2 — Sufficiency Head Architecture](https://arxiv.org/html/2605.09948v2/detail_2.png)

**说明**: Sufficiency Head 的详细设计。Sufficiency tokens 注入[[循环位置编码]]后，通过[[交叉注意力|Cross-Attention]] 与 Action tokens 交互，MLP + Sigmoid 输出每迭代的停止分量 $s^{(n)}$，经 [[剩余质量分配|RMA]] 得到归一化概率 $p^{(n)}$。

---

### Figure 3: LoopVLA 系统总览

![Figure 3 — LoopVLA Overview](https://arxiv.org/html/2605.09948v2/overview_7.png)

**说明**: 完整系统流程。视觉（$o_t$）+ 语言（$\ell$）+ Action tokens + Sufficiency tokens 经感知锚层后进入共享 Loop Block，迭代 $N$ 次，每次输出动作预测和充分性分数，最终选择最优迭代的动作。

---

### Figure 4: LIBERO-Plus 分布偏移样例

![Figure 4 — LIBERO-Plus Distribution Shifts](https://arxiv.org/html/2605.09948v2/libero-plus.png)

**说明**: LIBERO-Plus 零样本泛化基准的 7 种分布偏移类型：相机视角、机器人状态、语言指令、光照、背景、传感器噪声、物体布局。LoopVLA 在分布偏移下平均达 65.8%，以 1.2B 参数接近 π₀.₅（65.0%，4.1B 参数）。

---

### Figure 5: VLA-Arena 任务套件样例

![Figure 5 — VLA-Arena Task Suites](https://arxiv.org/html/2605.09948v2/vla-arena.png)

**说明**: VLA-Arena 基准的 L0 任务套件，涵盖安全性（Safety）、干扰物（Distractors）、外推（Extrapolation）和长时序（Long Horizon）四个维度。LoopVLA 在长时序任务上以 76.0% 显著超越 Qwen3OFT（59.0%）。

---

### Figure 6: 各任务套件的循环迭代分布

（图片 URL 未在 HTML 中暴露，见 arXiv 附录 A.3）

**说明**: 统计不同 LIBERO 任务套件中被选中迭代步骤 $n^*$ 的分布。简单任务（Object）集中在前几次迭代；复杂任务（Spatial、Goal）偏向更深精炼；长时序任务呈双峰分布，说明早期迭代已能处理简单子步骤，复杂决策需要更多精炼。

---

### Table 1: LIBERO 基准主结果

| 方法 | Spatial | Object | Goal | Long | Avg | Params | FLOPs | 吞吐 (Hz) |
|------|---------|--------|------|------|-----|--------|-------|----------|
| Diffusion Policy | 78.3 | 92.5 | 68.3 | 50.0 | 72.4 | — | — | — |
| OpenVLA | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 | 7.2B | 6.58T | 3.26 |
| π₀+FAST | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 | 3.5B | 606.68T | 0.05 |
| GR00T-N1.5 | 92.0 | 92.0 | 86.0 | 76.0 | 86.5 | 2.4B | 0.47T | 7.63 |
| UniVLA | 96.5 | 96.8 | 95.6 | 92.0 | 95.2 | 7.2B | 108.71T | 1.41 |
| π₀ | 96.8 | 98.8 | 95.8 | 85.2 | 94.1 | 4.0B | 1.79T | 3.70 |
| π₀.₅ | 95.4 | 98.4 | 97.2 | 89.6 | 95.1 | 4.1B | 2.41T | 3.11 |
| F1-VLA | 98.2 | 97.8 | 95.4 | 91.3 | 95.7 | 4.2B | 5.93T | 1.68 |
| Qwen3FM | 94.0 | 92.3 | 91.3 | 65.7 | 85.8 | 2.3B | 0.53T | 0.97 |
| LoopFMα (3⊗8) | 91.0 | 99.0 | 95.3 | 79.0 | 91.1 | 1.3B | 0.38T | 2.04 |
| Qwen3OFT | 95.0 | 97.0 | 97.1 | 90.5 | 94.9 | 2.2B | 0.53T | 10.49 |
| LoopOFT₀ (3⊗8) | 93.4 | 98.2 | 96.0 | 90.6 | 94.6 | 1.2B | 0.31T | **18.41** |
| LoopOFTα (3⊗8) | 94.0 | 99.6 | 96.8 | 91.0 | 95.3 | 1.2B | 0.42T | 12.15 |
| **LoopOFT\* (3⊗8)** | **95.0** | **100** | **97.4** | **91.0** | **96.0** | **1.2B** | 0.53T | 10.93 |

**关键发现**: LoopOFT\* 以 1.2B 参数（Qwen3OFT 的 55%）超越所有基线，达到 96.0% 平均成功率；LoopOFT₀ 吞吐量 18.41 Hz 是 Qwen3OFT 的 1.76×。

---

### Table 2: LIBERO-Plus 零样本泛化结果

| 方法 | Camera | Robot | Language | Light | BG | Noise | Layout | Total | Params |
|------|--------|-------|----------|-------|-----|-------|--------|-------|--------|
| OpenVLA | 0.8 | 3.5 | 23.0 | 8.1 | 34.8 | 15.2 | 28.5 | 15.6 | 7.2B |
| UniVLA | 1.8 | 46.2 | 69.6 | 69.0 | 81.0 | 21.2 | 31.9 | 45.8 | 7.2B |
| π₀ | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | 53.6 | 4.0B |
| π₀+FAST | 65.1 | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 61.6 | 3.5B |
| π₀.₅ | 53.0 | 50.3 | 65.7 | 83.1 | 77.3 | 53.2 | 72.7 | 65.0 | 4.1B |
| VLA-Adapter | 76.6 | 36.4 | 73.8 | 71.0 | 70.2 | 37.4 | 57.2 | 60.4 | 1.2B |
| Qwen3OFT | 45.6 | 55.0 | 73.5 | 86.1 | 90.2 | 69.3 | 72.4 | 68.8 | 2.2B |
| **LoopOFTα (3⊗8)** | 58.3 | 41.7 | 66.7 | 88.9 | 88.3 | 61.4 | 71.8 | **65.8** | **1.2B** |

**关键发现**: LoopOFTα 以 1.2B 参数达到 65.8%，接近参数量 3.4× 的 π₀.₅（65.0%）；在光照和背景泛化上表现尤为出色。

---

### Table 3: VLA-Arena 结果

| 方法 | Safety | Distractor | Extrapolation | Long Horizon | Avg |
|------|--------|-----------|---------------|--------------|-----|
| SmolVLA | 33.0 | 48.0 | 23.0 | 74.0 | 37.0 |
| Qwen3OFT | 54.0 | 71.0 | 13.3 | 59.0 | 47.0 |
| **LoopVLA** | **55.6** | 60.0 | 12.0 | **76.0** | **48.7** |

**关键发现**: LoopVLA 在长时序任务上以 76.0% 大幅超越 Qwen3OFT（59.0%），说明循环精炼对需要复杂规划的长程任务特别有益。

---

### Table 4: 循环配置消融（层数 ⊗ 迭代次数）

| 配置 | Spatial | Object | Goal | Long | Avg | Params |
|------|---------|--------|------|------|-----|--------|
| 8⊗3 | 95.0 | 100 | 95.3 | 80.7 | 92.8 | 1.48B |
| 6⊗4 | 96.0 | 98.7 | 95.7 | 85.0 | 93.9 | 1.37B |
| 4⊗6 | 96.4 | 96.2 | 97.0 | 90.5 | 95.0 | 1.27B |
| **3⊗8** | **94.7** | **98.6** | **96.6** | **90.2** | **95.0** | **1.22B** |
| 2⊗12 | 91.2 | 98.6 | 95.0 | 83.4 | 92.1 | 1.17B |

**关键发现**: 均衡配置（3⊗8、4⊗6）优于极端配置（层数过多或迭代次数过少），说明适当的"深度-迭代"平衡至关重要。

---

### Table 5: 推理层选择策略消融

| 策略 | Spatial | Object | Goal | Long | Avg |
|------|---------|--------|------|------|-----|
| (3⊗8)₁（固定第1次） | 91.0 | 98.7 | 95.4 | 90.0 | 93.8 |
| (3⊗8)₃（固定第3次） | 93.2 | 99.4 | 96.0 | 90.6 | 94.9 |
| (3⊗8)₅（固定第5次） | 94.6 | 96.7 | 96.2 | 91.0 | 94.7 |
| (3⊗8)₆（固定第6次） | 93.4 | 99.4 | 95.6 | 90.6 | 94.8 |
| (3⊗8)₈（固定第8次） | 92.0 | 99.0 | 94.6 | 91.0 | 94.2 |
| **(3⊗8)\*（自适应选择）** | **95.0** | **100** | **97.4** | **91.0** | **96.0** |

**关键发现**: 自适应选择（96.0%）持续优于任何固定层策略（最高 94.9%），证实不同输入确实需要不同精炼深度。

---

### Table 6: Sufficiency Head 设计消融

| 方法 | Spatial | Object | Goal | Long | Avg |
|------|---------|--------|------|------|-----|
| 直接 MLP（无 Cross-Attn） | 94.0 | 99.2 | 97.2 | 87.0 | 94.4 |
| **Sufficiency Head（完整设计）** | **95.0** | **100.0** | **97.4** | **91.0** | **96.0** |

**关键发现**: 完整 Sufficiency Head 设计在长时序任务上提升最大（87.0% → 91.0%），表明 Cross-Attention 全局聚合对复杂任务的充分性估计尤为重要。

---

### Table 7: 训练超参数

| 超参数 | 值 |
|--------|---|
| 优化器 | AdamW（余弦调度） |
| 全局 Batch Size | 64 |
| Action Chunk Size | 8 |
| Action Tokens 数量 | 8 |
| Sufficiency Tokens 数量 | 3 |
| 熵正则化权重 $\lambda_1$ | 0.001 |
| 多样性正则化权重 $\lambda_2$ | 0.01 |
| Warmup Steps | 1K |
| Stage 2 温度 $\tau$ | 0.5 |
| 早退阈值 $\theta$ | 0.68 |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| LIBERO | 4 套任务（Spatial/Object/Goal/Long） | 仿真，语言条件机器人操控 | 主要训练+测试 |
| LIBERO-Plus | 7 种分布偏移 | 零样本泛化测试 | 泛化评估 |
| VLA-Arena | 4 维度（Safety/Distractor/Extrapolation/Long） | 仿真多样挑战 | 泛化评估 |

### 实现细节

- **Backbone**: Qwen3 系列（OFT 变体用 [[OFT|Orthogonal Fine-Tuning]]，FM 变体用 Flow Matching）
- **优化器**: AdamW，余弦学习率调度，Warmup 1K 步
- **Batch Size**: 64
- **Loop 配置**: 默认 3 层感知锚层 ⊗ 8 次循环迭代（3⊗8）
- **训练阶段**: Stage 1（联合精炼）→ Stage 2（充分性校准，冻结主体仅训练 Sufficiency Head）

### 可视化结果

- **迭代深度分布**（Figure 6）：Object 任务多在第 1-3 次迭代停止；Spatial/Goal 偏向第 5-8 次；Long-Horizon 出现双峰（简单子任务早停，复杂决策深精炼）
- **LoopFM vs LoopOFT**：OFT 基座性能更优，两套变体均验证了循环精炼框架的有效性

---

## 批判性思考

### 优点

1. **理论动机清晰**: 循环精炼 + 充分性估计的结合，为"动态计算深度"提供了内生学习机制，优于启发式后处理方案。
2. **高效参数利用**: 1.2B 参数超越 4.0B 的 π₀，参数效率提升极为显著，工程实践价值高。
3. **两阶段训练设计合理**: Stage 1 先确保每个精炼深度的动作质量，Stage 2 再对齐充分性估计与质量分布，避免两个目标相互干扰。
4. **无监督充分性标签**: 质量导出分布 $q^{(n)}$ 完全来自已有的动作损失，无需额外标注，方法简洁。

### 局限性

1. **严重分布偏移时能力受限**: LIBERO-Plus 的 Camera（58.3%）和 Robot（41.7%）偏移成功率仍较低，说明循环精炼对几何/运动学分布偏移的泛化能力有限。
2. **缺乏真实机器人实验**: 所有评估均在仿真中进行，真实部署效果未知。
3. **Scaling 属性未探索**: 论文仅测试 1.2B-1.5B 规模，更大模型（7B+）下循环精炼是否仍然有效未知。
4. **感知锚层细节不足**: 论文未充分说明感知锚层的具体设计对循环精炼质量的影响。

### 潜在改进方向

1. **在线强化学习结合**: 用真实机器人交互信号替代/补充 Stage 2 的质量导出分布，增强对真实场景的适应性。
2. **多粒度循环**: 对不同 Token 类型（视觉/语言/动作）分别设计独立的充分性估计，实现更细粒度的动态计算。
3. **跨任务迁移**: 预训练充分性估计头在不同机器人平台/任务上的迁移学习。

### 可复现性评估

- [ ] 代码开源（尚未开源）
- [ ] 预训练模型（未发布）
- [x] 训练细节完整（超参数完整记录在 Table 7）
- [x] 数据集可获取（LIBERO/VLA-Arena 均公开）

---

## 关联笔记

### 基于

- [[Pi0|π₀]]: 主要对比基线，LoopVLA 以更少参数超越其在 LIBERO 上的性能
- [[OpenVLA-OFT|Qwen3OFT]]: 同 Backbone 基线，LoopVLA 在其基础上引入循环精炼
- [[GR00T N1.5]]: 对比基线之一，LoopVLA 在长时序任务上大幅超越

### 对比

- [[Diffusion Policy]]: LoopVLA 在 LIBERO 平均成功率（96.0%）远超 Diffusion Policy（72.4%）
- [[UniVLA]]: 以 1.2B 参数比肩 UniVLA（7.2B, 95.2%）的性能

### 方法相关

- [[Adaptive Depth Reasoning]]: 相似的自适应计算深度思想，在推理 LLM 领域有类似工作
- [[Action Chunking]]: LoopVLA 的动作输出形式
- [[OFT]]: LoopOFT 变体使用的微调方法
- [[Flow Matching]]: LoopFM 变体使用的动作预测框架
- [[KL散度|KL Divergence]]: Stage 2 充分性校准的核心损失

### 硬件/数据相关

- [[LIBERO]]: 主要评估基准
- [[VLA-Arena]]: 多维度挑战评估基准

---

## 速查卡片

> [!summary] LoopVLA
> - **核心**: 通过共享循环 Transformer 块 + 充分性估计，让 VLA 学会"何时停止精炼"
> - **方法**: Loop Block（共享权重循环）+ Sufficiency Head（Cross-Attn 充分性估计）+ 两阶段训练（动作监督 → KL 校准）
> - **结果**: 1.2B 参数在 LIBERO 达 96.0%（超越 4B 级别基线），吞吐提升 1.7×
> - **代码**: 未开源

---

*笔记创建时间: 2026-08-20*
