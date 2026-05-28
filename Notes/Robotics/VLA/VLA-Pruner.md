---
title: "Bridging the Semantic-Action Gap in Visual Token Pruning for Efficient VLA Inference"
method_name: "VLA-Pruner"
authors: [Ziyan Liu, Yeqiu Chen, Yiming Zhang, Hongyi Cai, Tao Lin, Runquan Gui, Shuo Yang, Zheng Liu, Bo Zhao]
year: 2025
venue: arXiv
tags: [visual-token-pruning, vla, inference-acceleration, token-compression, training-free, attention-mechanism, temporal-smoothing]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2511.16449v5
created: 2026-05-28
---

# 论文笔记：Bridging the Semantic-Action Gap in Visual Token Pruning for Efficient VLA Inference

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Shanghai Jiao Tong University, USTC, Harbin Institute of Technology (Shenzhen), BAAI |
| 日期 | November 2025 (v5: May 2026) |
| 项目主页 | — |
| 对比基线 | [[FastV]], [[SparseVLM]], [[DivPrune]], VLA-Cache |
| 链接 | [arXiv](https://arxiv.org/abs/2511.16449) / [Code](https://github.com/MINT-SJTU/VLA-Pruner) |

---

## 一句话总结

> VLA-Pruner 通过联合建模语义注意力与动作解码注意力，以无训练方式剪除冗余视觉 token，在保持操作性能的同时实现最高 1.99× 的推理加速。

---

## 核心贡献

1. **语义-动作注意力差异分析**: 揭示 VLA 推理中 prefill 阶段与 action-decode 阶段的 top-attended 视觉 token 仅约 50% 重叠，直接套用 VLM 剪枝方法会破坏动作关键 token。
2. **VLA-Pruner 框架**: 一种无训练的双重重要性估计方法，结合语义相关性（prefill 注意力）与动作相关性（经时序平滑的历史 decode 注意力），并通过 Combine-then-Filter 策略去除冗余。
3. **跨架构验证**: 在 OpenVLA、OpenVLA-OFT、π0 三种 VLA 架构以及 LIBERO、SIMPLER benchmark 和真实机器人任务上均优于现有基线，最高达 1.99× 加速。

---

## 问题背景

### 要解决的问题

[[视觉语言动作模型]] (VLA) 推理延迟主要来源于 [[Transformer]] 对大量视觉 token 的处理（每帧 256×n 个 token），导致实时机器人控制受限。

### 现有方法的局限

已有 VLM token 剪枝方法（[[FastV]]、[[SparseVLM]]、[[DivPrune]]）仅利用语言-视觉 prefill 阶段的注意力分数来评估 token 重要性。然而 VLA 还需要一个额外的 action-decode 阶段，该阶段的注意力模式与 prefill 阶段显著不同，这些方法因此会删除对动作生成至关重要的视觉 token，导致操作性能严重下降（Figure 1）。

### 本文的动机

VLA 的 action-decode 阶段会对视觉 token 产生独立的注意力分布，且相邻时间步之间的注意力具有高度时序一致性（连续时间步 Top-k overlap > 80%）。因此可以用历史 decode 注意力的指数移动平均（[[指数移动平均|EMA]]）来近似当前时间步的动作相关重要性，从而在 prefill 时刻做出准确的 token 选择决策。

---

## 方法详解

### 模型架构

VLA-Pruner 是一个**无训练**的推理时 token 剪枝框架，作用于任意基于 [[Transformer]] 的 VLA 模型的第 3 层之后：

- **输入**: 语言指令 $l$ + 多帧视觉观测（每帧 $M$ 个视觉 token）+ 文本 token（$N$ 个）
- **Backbone**: 沿用原始 VLA 模型（OpenVLA、π0 等）
- **核心模块**: 双重重要性估计（Semantic-Action Importance Estimation）+ Combine-then-Filter 策略
- **输出**: 保留 $\tilde{M}$ 个视觉 token 后的动作预测 $a_t$
- **作用层**: Transformer 第 3 层（低层注意力更可靠）

### 核心模块

#### 模块1: 语义-动作双重重要性估计

**设计动机**: 利用 [[Self-Attention]] 在两个不同阶段的注意力分布来分别捕捉语义相关性和动作相关性。

**具体实现**:

**语义重要性**（来自 prefill 阶段）:
- 对所有 $M+N$ 个 token（视觉+文本）到第 $m$ 个视觉 token 的注意力求均值
- 使用最后一层 prefill 注意力

**动作重要性**（来自 action-decode 阶段的历史估计）:
- 对所有 $\hat{N}$ 个动作 token 到第 $m$ 个视觉 token 的注意力求均值
- 当前时间步的 action 注意力在 prefill 时不可得，因此使用 [[指数移动平均|EMA]] 从近 $w=3$ 步的历史数据中估计

#### 模块2: Combine-then-Filter 策略

**设计动机**: 简单的加权融合（Score-fusion）对权重超参数敏感，而 Combine-then-Filter 先取并集再去冗余，对超参数更鲁棒。

**具体实现**:
1. **Selection**: 分别从语义和动作重要性中选出 top-$\tilde{M}$ 个候选 token 集合 $\mathcal{C}_{vl}$ 和 $\mathcal{C}_{act}$
2. **Pooling**: 取并集 $\mathcal{C}_{dual} = \mathcal{C}_{vl} \cup \mathcal{C}_{act}$（候选集大小为 $|\mathcal{C}_{dual}| \leq 2\tilde{M}$）
3. **Filtering**: 用 [[最大最小多样性问题|MMDP]]（Max-Min Diversity Problem）基于余弦距离从并集中筛选出 $\tilde{M}$ 个互相最不相似（最多样）的 token，去除语义冗余

---

## 关键公式

### 公式1: [[Self-Attention|语义重要性分数]]

$$
\mathcal{S}_{vl}[m] = \frac{1}{M+N} \sum_{i=1}^{M+N} \mathcal{A}_{vl}[i, m]
$$

**含义**: 对 prefill 阶段所有 token（包括视觉和文本）关注到第 $m$ 个视觉位置的注意力权重求均值，衡量该视觉 token 的语义重要性。

**符号说明**:
- $M$: 视觉 token 数量
- $N$: 文本（语言+历史动作）token 数量
- $\mathcal{A}_{vl}[i, m]$: prefill 阶段第 $i$ 个 token 对第 $m$ 个视觉 token 的注意力权重

---

### 公式2: [[Self-Attention|动作-视觉注意力分数]]

$$
\mathcal{S}_{act}[m] = \frac{1}{\hat{N}} \sum_{i=1}^{\hat{N}} \mathcal{A}_{act}[i, m]
$$

**含义**: 对 action-decode 阶段所有动作 token 关注到第 $m$ 个视觉 token 的注意力权重求均值，衡量该视觉 token 对动作生成的重要性。

**符号说明**:
- $\hat{N}$: 动作 token 数量
- $\mathcal{A}_{act}[i, m]$: 第 $i$ 个动作 token 对第 $m$ 个视觉 token 的注意力权重
- 取后半层的均值以获得更稳定的估计

---

### 公式3: [[指数移动平均|时序平滑估计（EMA）]]

$$
\hat{\mathcal{S}}_{act}^{t} = \frac{\sum_{i=1}^{w} \gamma^{i} \mathcal{S}_{act}^{t-i}}{\sum_{i=1}^{w} \gamma^{i}}
$$

**含义**: 用过去 $w$ 个时间步的动作注意力的指数加权平均来估计当前时间步的动作相关视觉 token 重要性，越近的历史权重越大。

**符号说明**:
- $w = 3$: 历史窗口大小（实验表明 $w \in [1, 5]$ 均稳定）
- $\gamma = 0.8$: 衰减系数（越近的时间步权重越大）
- $\mathcal{S}_{act}^{t-i}$: $t-i$ 时间步的动作注意力分数
- 冷启动：前 $w$ 步仅用语义重要性（warm-start period）

---

### 公式4: [[视觉 Token 剪枝|优化目标]]

$$
\min_{f} \mathcal{L}(\mathcal{P}, \tilde{\mathcal{P}}) \quad \text{s.t.} \quad |f(\mathbf{E}_v)| = \tilde{M}
$$

**含义**: 寻找 token 选择函数 $f$，使剪枝后 VLA 的输出分布 $\tilde{\mathcal{P}}$ 与原始分布 $\mathcal{P}$ 的差异最小，同时将视觉 token 数压缩到 $\tilde{M}$。

**符号说明**:
- $f$: token 选择函数
- $\mathbf{E}_v$: 视觉 token 嵌入集合（大小为 $M$）
- $\tilde{M}$: 目标保留 token 数（$\tilde{M} < M$）
- $\mathcal{L}$: 两个分布之间的差异度量

---

### 公式5: [[最大最小多样性问题|Filtering 余弦距离]]

$$
d(v_i, v_j) = 1 - \frac{v_i \cdot v_j}{\|v_i\| \|v_j\|}
$$

**含义**: 用 1 减去余弦相似度作为两个视觉 token 特征之间的距离度量，MMDP 贪心选择使最小距离最大化的子集，以去除冗余视觉 token。

**符号说明**:
- $v_i, v_j$: 视觉 token 的特征向量
- $d(\cdot, \cdot) \in [0, 2]$: 余弦距离，值越大越不相似

---

## 关键图表

### Figure 1: 各方法在不同剪枝比例下的性能对比

![Figure 1](https://arxiv.org/html/2511.16449v5/x1.png)

**说明**: 在多种剪枝比例下，VLA-Pruner（蓝线）相比 FastV、SparseVLM、DivPrune、VLA-Cache 等基线保持了最高的相对准确率。直接将 VLM 剪枝方法（FastV 等）应用于 VLA 推理会导致性能大幅下降，在高压缩率下尤为明显。

---

### Figure 2: VLA 推理阶段的注意力差异分析

![Figure 2a - Average Overlap](https://arxiv.org/html/2511.16449v5/x2.png)

![Figure 2b - Overlap Ratio Trend](https://arxiv.org/html/2511.16449v5/x3.png)

![Figure 2c - Prefill Attention Map](https://arxiv.org/html/2511.16449v5/x4.png)

![Figure 2d - Decode Attention Map](https://arxiv.org/html/2511.16449v5/x5.png)

**说明**: (a-b) 不同阶段 Top-k attended 视觉 patch 的重叠率。Prefill 与 action-decode 注意力 Top-k 重叠仅约 50%，而连续时间步之间的 action-decode 注意力重叠 > 80%。(c-d) Prefill 阶段注意力集中在语义物体上（杯子），而 action-decode 阶段则关注末端执行器附近的局部区域，两者模式截然不同，支撑了本文的核心动机。

---

### Figure 3: VLA-Pruner 整体框架（budget k=3）

![Figure 3 - VLA-Pruner Overview](https://arxiv.org/html/2511.16449v5/x6.png)

**说明**: VLA-Pruner 的整体框架。在 Transformer 第 3 层后，分别提取语义重要性（prefill 注意力）和由 [[指数移动平均|EMA]] 估计的动作重要性，各选出候选集合后取并集，再通过 MMDP 去冗余，最终保留 $\tilde{M}$ 个视觉 token。动作 token 预测仍在完整 token 序列上进行。

---

### Figure 4: OpenVLA 在 LIBERO 各子任务下的性能曲线

![Figure 4a](https://arxiv.org/html/2511.16449v5/x7.png)

![Figure 4b](https://arxiv.org/html/2511.16449v5/x8.png)

![Figure 4c](https://arxiv.org/html/2511.16449v5/x9.png)

![Figure 4d](https://arxiv.org/html/2511.16449v5/x10.png)

**说明**: 在 LIBERO-Spatial、LIBERO-Object、LIBERO-Goal、LIBERO-Long 四个子任务上，随剪枝比例增加，VLA-Pruner 始终显著优于所有基线。在 50% token 保留率下，VLA-Pruner 甚至略超过原始 Vanilla 模型（相对准确率 >100%）。

---

### Figure 5: 真实机器人验证（75% 剪枝率）

![Figure 5 - Real Robot Performance](https://arxiv.org/html/2511.16449v5/x11.png)

**说明**: 在 6-DoF xArm6 机械臂上对 Can Stack、Cup Pour、Cube Place、Bottle Place 四个任务进行测试，75% 剪枝率下 VLA-Pruner 在 OpenVLA-OFT 上取得最高相对准确率，验证了方法在真实场景中的有效性。

---

### Figure 6: 消融分析与可视化

![Figure 6a - Ablation](https://arxiv.org/html/2511.16449v5/x12.png)

![Figure 6b - Sensitivity](https://arxiv.org/html/2511.16449v5/x13.png)

![Figure 6c - Visualization](https://arxiv.org/html/2511.16449v5/x14.png)

**说明**: (a) 消融实验：Prefill-only（FastV）和 Action-only 分别不足，Score-fusion 对权重敏感，Combine-then-Filter 表现最优。(b) 超参数 $w$（窗口大小，1-5）和 $\gamma$（衰减率）敏感性分析，方法鲁棒性强。(c) 定性可视化：VLA-Pruner 保留了机械臂末端执行器和目标物体附近的关键 token。

---

### Table 1: LIBERO 基准完整结果

| 方法 | OpenVLA Spatial | Object | Goal | Long | 相对准确率↑ | FLOPs(T)↓ | 延迟(ms)↓ | OFT Spatial | OFT Object | OFT Goal | OFT Long | 相对准确率↑ | FLOPs(T)↓ | 延迟(ms)↓ |
|------|----------------|--------|------|------|------------|-----------|-----------|-------------|------------|----------|----------|------------|-----------|-----------|
| **Vanilla (100%)** | 87.6 | 84.6 | 78.6 | 52.2 | 100.0% | 1.906 | 236.41 | 95.8 | 98.7 | 96.3 | 90.7 | 100.0% | 3.903 | 135.78 |
| **保留 50% Token** | | | | | | | | | | | | | | |
| FastV | 86.2 | 81.6 | 77.2 | 50.6 | 97.43% | 1.136 | 172.32 | 94.6 | 96.8 | 92.7 | 87.0 | 97.26% | 2.219 | 88.23 |
| SparseVLM | 85.6 | 80.5 | 75.0 | 48.6 | 95.44% | 1.155 | 175.77 | 94.1 | 93.7 | 91.2 | 85.1 | 95.37% | 2.289 | 90.22 |
| DivPrune | 82.6 | 78.8 | 71.8 | 47.6 | 92.64% | 1.105 | 173.88 | 90.8 | 91.1 | 89.9 | 83.1 | 92.61% | 2.126 | 88.01 |
| VLA-Cache | 87.1 | 80.7 | 78.6 | 51.8 | 98.48% | 1.384 | 192.20 | 95.4 | 96.0 | 96.7 | 90.2 | 99.09% | 2.730 | 101.01 |
| **VLA-Pruner** | **88.2** | **85.8** | **79.4** | **56.4** | **102.45%** | 1.139 | 178.38 | **97.3** | **98.6** | **96.8** | **92.6** | **101.05%** | 2.234 | 92.86 |
| **保留 25% Token** | | | | | | | | | | | | | | |
| FastV | 81.6 | 69.6 | 71.6 | 43.8 | 87.62% | 0.757 | 141.25 | 87.8 | 81.8 | 87.6 | 74.6 | 87.48% | 1.401 | 72.73 |
| SparseVLM | 83.4 | 72.8 | 67.6 | 45.6 | 88.67% | 0.772 | 144.12 | 89.8 | 84.8 | 82.7 | 79.1 | 88.63% | 1.459 | 74.71 |
| DivPrune | 77.4 | 61.3 | 65.4 | 41.4 | 80.96% | 0.743 | 142.12 | 83.9 | 72.6 | 79.3 | 72.5 | 80.76% | 1.389 | 72.30 |
| VLA-Cache | 78.1 | 73.2 | 70.2 | 45.5 | 88.08% | 0.961 | 164.17 | 85.6 | 85.3 | 84.1 | 80.3 | 88.11% | 1.938 | 85.02 |
| **VLA-Pruner** | **85.4** | **82.5** | **78.4** | **51.8** | **98.48%** | 0.758 | 144.81 | **93.5** | **96.2** | **95.2** | **90.2** | **98.37%** | 1.420 | 75.64 |
| **保留 12.5% Token** | | | | | | | | | | | | | | |
| FastV | 62.0 | 58.5 | 55.8 | 18.8 | 63.08% | 0.568 | 125.76 | 60.6 | 59.2 | 60.6 | 28.6 | 61.64% | 1.082 | 66.11 |
| SparseVLM | 65.8 | 55.3 | 55.4 | 19.0 | 63.20% | 0.581 | 128.77 | 73.1 | 65.0 | 67.5 | 32.8 | 62.11% | 1.136 | 67.65 |
| DivPrune | 55.4 | 54.8 | 51.4 | 17.6 | 58.06% | 0.569 | 125.36 | 61.4 | 61.8 | 64.9 | 30.8 | 56.83% | 1.059 | 66.46 |
| VLA-Cache | 52.5 | 50.1 | 52.0 | 15.1 | 54.79% | 0.710 | 145.93 | 57.3 | 58.7 | 64.8 | 26.2 | 53.88% | 1.373 | 76.15 |
| **VLA-Pruner** | **80.2** | **78.4** | **69.0** | **42.8** | **88.90%** | 0.571 | 129.01 | **88.1** | **87.6** | **84.9** | **68.8** | **88.27%** | 1.096 | 68.95 |

**关键发现**: 在 12.5% token 保留率下，VLA-Pruner 相对准确率达 88.9%，而最佳基线仅 63.2%，领先幅度高达 **34.39%**。即使保留极少 token，VLA-Pruner 仍能维持接近 Vanilla 的操作质量。

---

### Table 2: SIMPLER 跨环境泛化（75% 剪枝率）

| 方法 | Move Near | Pick Coke Can | Open/Close Drawer | 整体 |
|------|-----------|---------------|-------------------|------|
| OpenVLA (Vanilla) | 54.0% (100%) | 52.8% (100%) | 49.5% (100%) | 52.1% (100%) |
| FastV | 38.7% (71.7%) | 41.9% (79.4%) | 33.7% (68.1%) | 38.1% (73.1%) |
| VLA-Cache | 41.2% (76.3%) | 40.6% (76.9%) | 38.8% (78.4%) | 40.2% (77.2%) |
| **VLA-Pruner** | **52.4% (97.0%)** | **50.1% (94.9%)** | **48.8% (98.6%)** | **50.4% (96.8%)** |

**关键发现**: 在 SIMPLER 跨环境泛化测试中，VLA-Pruner 保持整体 96.8% 相对准确率，远超 FastV（73.1%）和 VLA-Cache（77.2%），表明方法具备良好的跨场景鲁棒性。

---

## 实验

### 数据集与 Benchmark

| 数据集/Benchmark | 规模 | 特点 | 用途 |
|--------|------|------|------|
| LIBERO-Spatial | 500 episodes/任务 | 空间位置理解 | 训练+测试 |
| LIBERO-Object | 500 episodes/任务 | 物体识别 | 训练+测试 |
| LIBERO-Goal | 500 episodes/任务 | 目标条件 | 训练+测试 |
| LIBERO-Long | 500 episodes/任务 | 长时序任务 | 训练+测试 |
| SIMPLER | 多任务 | 跨环境迁移泛化 | 测试 |
| 真实机器人 | 4 任务 | xArm6 真实操作 | 测试 |

### 实现细节

- **剪枝层**: Transformer 第 3 层之后执行 token 剪枝
- **语义注意力**: 使用最后一层 prefill 注意力
- **动作注意力**: 对后半层注意力取均值
- **EMA 参数**: 窗口 $w=3$，衰减率 $\gamma=0.8$
- **方法特性**: 无训练（training-free），适用于自回归和扩散头策略
- **测试架构**: OpenVLA（7B LLaMA2 backbone）、OpenVLA-OFT、π0（3B PaliGemma）

### 效率分析

| 保留比例 | 相对加速比（OpenVLA） | FLOPs 降低 |
|---------|---------------------|-----------|
| 50% | 1.33× | ~40% |
| 25% | ~1.63× | ~60% |
| 12.5% | ~1.83× | ~70% |
| （π0 实测最高） | **1.99×** | — |

VLA-Pruner 的额外 GPU 内存占用与 Vanilla 模型几乎相同（可忽略不计），且相比同等压缩率的 token 剪枝基线还少约 20% FLOPs。

### 可视化结果

定性分析（Figure 6c）显示 VLA-Pruner 保留的视觉 token 主要集中在机械臂末端执行器和目标物体附近，语义上合理；而 FastV 保留的 token 分布更分散，缺少动作执行所需的关键局部信息。

---

## 批判性思考

### 优点
1. **洞察深刻**: 首次系统量化 VLA prefill 和 action-decode 两阶段注意力的差异，揭示了现有 VLM 剪枝方法失效的根本原因。
2. **无训练，即插即用**: 不需要微调原始模型，对任何基于 [[Transformer]] 的 VLA 架构通用。
3. **高压缩率下领先显著**: 在 12.5% 极端保留率下相对准确率领先最佳基线 34.39%，工程实用价值高。

### 局限性
1. **冷启动问题**: 前 $w=3$ 步无历史 action 注意力，只能依赖语义重要性，在任务初始阶段可能存在短暂的性能下降。
2. **超参数依赖**: $w$ 和 $\gamma$ 需要一定的调整，尽管实验表明鲁棒性较好，但跨机器人平台的适用性尚未充分验证。
3. **MMDP 近似计算**: Combine-then-Filter 的 MMDP 步骤使用贪心近似，引入额外延迟，在极端实时场景下可能成为瓶颈。

### 潜在改进方向
1. **自适应 budget**: 根据任务阶段（接近/抓取/放置）动态调整 token 保留数量，而非固定比例。
2. **多模态扩展**: 将时序平滑的思路扩展到多摄像头或深度图 token 的联合剪枝。

### 可复现性评估
- [x] 代码开源（[GitHub](https://github.com/MINT-SJTU/VLA-Pruner)）
- [ ] 预训练模型（未明确提供）
- [x] 训练细节完整（超参数均已报告）
- [x] 数据集可获取（LIBERO、SIMPLER 均公开）

---

## 关联笔记

### 基于
- [[视觉语言动作模型]]: VLA 架构基础，VLA-Pruner 在其推理流程中插入剪枝模块
- [[Transformer]]: 注意力机制和视觉 token 序列处理基础
- [[指数移动平均|EMA]]: 时序平滑估计动作注意力的核心工具

### 对比
- [[FastV]]: 仅用语义注意力剪枝，忽略动作阶段，本文主要对比基线
- [[SparseVLM]]: 文本-视觉交叉注意力剪枝，同样忽略动作阶段
- [[DivPrune]]: 多样性驱动的视觉 token 选择

### 方法相关
- [[Self-Attention]]: 注意力分数计算的基础
- [[最大最小多样性问题]]: Combine-then-Filter 的去冗余步骤
- [[视觉 Token 剪枝]]: 本文核心优化对象

### 硬件/数据相关
- [[LIBERO]]: 主要评测 benchmark（4 个子任务）
- [[SIMPLER]]: 跨环境泛化测试集

---

## 速查卡片

> [!summary] VLA-Pruner
> - **核心**: 联合语义和动作两阶段注意力来识别 VLA 推理的关键视觉 token
> - **方法**: 双重重要性估计（prefill 注意力 + EMA 平滑历史 decode 注意力）+ Combine-then-Filter（并集去冗余）
> - **结果**: 12.5% token 保留下达 88.9% 相对准确率（基线最高 63.2%），最高 1.99× 加速
> - **代码**: [https://github.com/MINT-SJTU/VLA-Pruner](https://github.com/MINT-SJTU/VLA-Pruner)

---

*笔记创建时间: 2026-05-28*
