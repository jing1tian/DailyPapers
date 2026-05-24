---
title: "Identifiable Token Correspondence for World Models"
method_name: "ITC"
authors: [Youngin Kim, Ray Sun, Inho Kim, Bumsoo Park, Hyun Oh Song]
year: 2025
venue: arXiv
tags: [world-model, optimal-transport, token-correspondence, visual-rl, transformer]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2605.16457
created: 2026-05-23
---

# 论文笔记：Identifiable Token Correspondence for World Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Seoul National University |
| 日期 | May 2025 |
| 项目主页 | — |
| 对比基线 | [[IRIS]]、[[DreamerV3]]、[[STORM]] |
| 链接 | [arXiv](https://arxiv.org/abs/2605.16457) / [Code](https://github.com/snu-mllab/Identifiable-Token-Correspondence) |

---

## 一句话总结

> ITC 通过最优传输将 token-based 世界模型的下一帧预测建模为结构化分配问题，使每个 token 选择性地从上一帧复制或重新生成，从根本上消除长程滚动中的对象重影和消失问题。

---

## 核心贡献

1. **ITC 解码机制**: 将下一帧 token 预测建模为[[最优传输|Optimal Transport]]分配问题，在不修改 Transformer 架构或训练流程的前提下插入 OT 求解器。
2. **3D RoPE 位置编码**: 引入时空联合的[[旋转位置编码|3D RoPE]]，同时编码 token 的空间 (x, y) 坐标和时间维度，增强跨帧的位置感知能力。
3. **全面 SOTA 提升**: 在 Craftax-classic、MinAtar、Atari 100K 四个基准上达到 SOTA，且额外计算开销仅 2.8%。

---

## 问题背景

### 要解决的问题

基于 token 的[[Transformer]]世界模型在长程想象（rollout）中存在严重的时序不一致问题：相邻帧中持续存在的对象（如游戏角色、建筑物）会被模型重复生成，导致**对象重影（duplication）**和**对象消失（disappearance）**。

### 现有方法的局限

[[IRIS]]、[[DreamerV3]]、[[STORM]] 等模型把下一帧预测纯粹视为 token 生成任务，完全忽略了"相邻帧中大量 token 对应同一持续存在的实体"这一先验知识。模型被迫在每步从头生成所有 token，既低效又容易产生幻觉。

### 本文的动机

如同光流（Optical Flow）的直觉：相邻帧之间共享大量结构，大多数 token 只是空间位移而非全新生成。ITC 利用这一先验，让模型显式学习哪些 token 可以直接从前一帧复制，哪些需要重新采样，从而减少自由度、抑制幻觉。

---

## 方法详解

### 整体架构

ITC 在[[Transformer]]输出层与最终 token 决策之间插入一个[[最优传输]]求解层：

- **输入**: 上一帧 token $\mathbf{u} \in \mathbb{R}^{L \times d}$（前一帧）+ Transformer 预测 $\mathbf{p} \in \mathbb{R}^{L \times d}$（候选下一帧）
- **OT 求解器**: 计算亲和矩阵 → [[Sinkhorn 算法]]得到软分配 → [[二值化（Binarization）]]得到硬分配
- **输出**: 最终下一帧 token $\hat{\mathbf{u}}' \in \mathbb{R}^{L \times d}$，每个 token 来自复制或新采样
- **总参数**: 与基线模型相同（ITC 不添加可学习参数）

### 核心模块

#### 模块1: 二部图亲和矩阵

**设计动机**: 利用[[最优传输]]框架，把 2L 个源 token（L 个前帧 token + L 个候选 token）与 L 个目标 token 之间的匹配代价形式化。

**前帧亲和矩阵** $A^{(\text{prev})} \in \mathbb{R}^{L \times L}$：

$$
A^{(\text{prev})}_{ij} = \langle p_j, u_i \rangle - c_d \cdot D\bigl((x_i, y_i),\ (x_j, y_j)\bigr)
$$

- 点积 $\langle p_j, u_i \rangle$ 衡量预测 token 与前帧 token 的语义相似度
- 距离惩罚项 $c_d \cdot D(\cdot)$ 阻止 token 发生不合理的大位移

**生成 token 亲和矩阵** $A^{(\text{gen})} \in \mathbb{R}^{L \times L}$（对角形式）：

$$
A^{(\text{gen})}_{kj} = \begin{cases} \|p_j\|_\infty - c_w & \text{if } k = j \\ -\infty & \text{otherwise} \end{cases}
$$

- 对角项代表"wildcard"：当 Transformer 对预测 token 有高置信度（大 $\ell_\infty$ 范数）时，鼓励生成新 token
- $c_w$ 为 wildcard 惩罚常数，平衡复制与生成的倾向

#### 模块2: Sinkhorn 最优传输求解

**设计动机**: 用[[Sinkhorn 算法]]高效求解带约束的最优传输问题，得到软分配矩阵。

将两类亲和矩阵拼接为统一的 $2L \times L$ 矩阵后，求解以下约束优化问题：

$$
\min_{P^{(\text{prev})},\ P^{(\text{gen})}} \left\langle -\begin{bmatrix} A^{(\text{prev})} \\ A^{(\text{gen})} \end{bmatrix},\ \begin{bmatrix} P^{(\text{prev})} \\ P^{(\text{gen})} \end{bmatrix} \right\rangle
$$

约束条件：
- $P^{(\text{prev})} \mathbf{1}_L \leq \mathbf{1}_L$（每个前帧 token 最多被使用一次）
- $P^{(\text{gen})} \mathbf{1}_L \leq \mathbf{1}_L$（每个 wildcard 最多使用一次）
- $\left(P^{(\text{prev})} + P^{(\text{gen})}\right)^T \mathbf{1}_L = \mathbf{1}_L$（每个目标 token 恰好有一个来源）

#### 模块3: 二值化（Binarization）

**设计动机**: 将连续的软分配 $P$ 转换为严格的 0-1 硬分配 $\Pi$，使最终 token 选择确定化。

使用贪心列优先 argmax（Algorithm 2）：逐列找到得分最高的源 token，冲突时按优先级重分配，直到所有分配稳定。

#### 模块4: 3D RoPE 位置编码

**设计动机**: 标准 1D [[旋转位置编码|RoPE]] 无法感知 token 的空间位置，3D RoPE 同时编码 $(x, y, t)$ 三维坐标，使 Transformer 能利用空间邻近性先验。

- 对 $x, y$ 坐标使用空间旋转矩阵，对时间步 $t$ 使用时间旋转矩阵
- 采用**块因果注意力掩码（Block Causal Attention Mask）**：同帧内所有 token 互相可见，跨帧只能看到历史帧

---

## 关键公式

### 公式1: [[最优传输|前帧亲和矩阵]]

$$
A^{(\text{prev})}_{ij} = \langle p_j,\ u_i \rangle - c_d \cdot D\bigl((x_i, y_i),\ (x_j, y_j)\bigr)
$$

**含义**: 衡量 Transformer 预测的第 $j$ 个 token 与前帧第 $i$ 个 token 的对应关系，结合语义相似度和空间位移惩罚。

**符号说明**:
- $p_j$: Transformer 预测的第 $j$ 个下一帧候选 token（嵌入向量）
- $u_i$: 前一帧第 $i$ 个 token
- $c_d$: 距离代价系数（默认 0.6）
- $D(\cdot)$: 2D 欧氏距离函数

### 公式2: [[最优传输|生成 Token 亲和矩阵]]

$$
A^{(\text{gen})}_{kj} = \begin{cases} \|p_j\|_\infty - c_w & \text{if } k = j \\ -\infty & \text{otherwise} \end{cases}
$$

**含义**: 对角矩阵，衡量"直接采用 Transformer 生成的新 token"的收益，置信度越高越倾向于生成。

**符号说明**:
- $\|p_j\|_\infty$: 预测 token 的 $\ell_\infty$ 范数，作为生成置信度代理
- $c_w$: Wildcard 惩罚常数（默认 0.3），越大越倾向于复制前帧 token

### 公式3: [[最优传输|最优传输目标函数]]

$$
\min_{P^{(\text{prev})},\ P^{(\text{gen})}} \left\langle -\begin{bmatrix} A^{(\text{prev})} \\ A^{(\text{gen})} \end{bmatrix},\ \begin{bmatrix} P^{(\text{prev})} \\ P^{(\text{gen})} \end{bmatrix} \right\rangle
$$

约束：$P^{(\text{prev})} \mathbf{1}_L \leq \mathbf{1}_L,\quad P^{(\text{gen})} \mathbf{1}_L \leq \mathbf{1}_L,\quad (P^{(\text{prev})} + P^{(\text{gen})})^T \mathbf{1}_L = \mathbf{1}_L$

**含义**: 在"每个目标 token 恰好分配一个来源"的约束下，最大化总亲和度（最小化负亲和度），即找到代价最小的 token 匹配方案。

**符号说明**:
- $P^{(\text{prev})},\ P^{(\text{gen})}$: 连续软分配矩阵（由 Sinkhorn 输出）
- $\mathbf{1}_L$: 全 1 向量，维度 $L$

### 公式4: [[最近邻查找|Tokenizer 最近邻 Token 查找]]

$$
q_i = \arg\min_{1 \leq k \leq K} \|p_i - c_k\|_2^2
$$

**含义**: 将 Transformer 输出的连续嵌入向量 $p_i$ 映射到最近的离散 codebook 条目 $c_k$，得到离散 token 索引。

**符号说明**:
- $K$: Codebook 大小（默认 4096）
- $c_k$: 第 $k$ 个 codebook 向量（$7 \times 7 \times 3$ patch 的平坦嵌入）

### 公式5: [[ITC|最终 Token 解码规则]]

$$
u'_j = \begin{cases} u_i & \text{if } \Pi^{(\text{prev})}_{ij} = 1 \quad \text{（从前帧复制）} \\ \text{sample}(p_j) & \text{if } \Pi^{(\text{gen})}_{jj} = 1 \quad \text{（新采样）} \end{cases}
$$

**含义**: 根据二值化后的分配矩阵，每个目标 token 要么直接复制前帧对应 token，要么从 Transformer 预测分布中采样新 token。

**符号说明**:
- $\Pi^{(\text{prev})},\ \Pi^{(\text{gen})}$: 二值化后的硬分配矩阵 $\in \{0, 1\}^{L \times L}$
- $\text{sample}(p_j)$: 从预测 codebook 分布中离散采样

### 公式6: [[PPO|PPO 策略损失]]

$$
\mathcal{L}_{\text{PPO}}(\Phi) = \frac{1}{T} \sum_t \left\{ -\min\bigl(p_t(\Phi)A_t,\ \text{clip}(p_t(\Phi))A_t\bigr) + \lambda_{\text{TD}}\bigl(V_\Phi(o_t) - q_t\bigr)^2 - \lambda_{\text{ent}} H\bigl(\pi_\Phi(\cdot | o_t)\bigr) \right\}
$$

**含义**: 在世界模型想象出的轨迹中训练策略网络，包含 PPO 裁剪损失、TD 值函数损失和熵正则项。

**符号说明**:
- $p_t(\Phi)$: 当前策略与旧策略的概率比
- $A_t$: 优势估计
- $\lambda_{\text{TD}} = 2.0$, $\lambda_{\text{ent}} = 0.01$: 权重系数
- $H(\cdot)$: 策略熵

---

## 关键图表

### Figure 1: 问题动机

![Figure 1](https://arxiv.org/html/2605.16457v2/x1.png)

**说明**: 可视化环境（Craftax-classic 和 Atari）相邻帧的 token 结构。相邻帧中大量 token 对应同一持续实体，说明直接从上一帧复制 token 是合理且高效的。

### Figure 2: ITC 方法概览

![Figure 2](https://arxiv.org/html/2605.16457v2/x2.png)

**说明**: ITC 核心流程图。Transformer 预测 $\tilde{s}_{t+1}$（绿色）与前帧 token $s_t$（蓝色）共同输入 OT 求解器，通过亲和矩阵计算传输计划，最终输出 $\hat{s}_{t+1}$——每个 token 选择性地来自复制或新生成。

### Figure 3: Craftax-classic 性能曲线

![Figure 3](https://arxiv.org/html/2605.16457v2/x3.png)

**说明**: ITC 在 Craftax-classic 上的 Return 和 Score 收敛曲线（10 个种子，阴影为标准差）。ITC 在 0.5M 步就超过了基线 1M 步的性能，展现出显著更快的收敛速度。

### Figure 4: 想象轨迹对比（Craftax-classic）

![Figure 4](https://arxiv.org/html/2605.16457v2/x4.png)

**说明**: Ground-truth 轨迹 vs. 基线 vs. ITC 的想象轨迹对比。红框标注动力学预测错误，蓝框标注重影错误。ITC 成功消除基线中存在的对象重影（蓝框）并修正了部分动力学误差（红框）。

### Figure 5: DreamerV3 vs. ITC 想象对比

![Figure 5](https://arxiv.org/html/2605.16457v2/x5.png)

**说明**: DreamerV3 想象轨迹与 Ground-truth 对比。红框标注树木消失现象，蓝框标注重影。展示 token-free 世界模型同样存在此类幻觉问题。

### Figure 6: 注意力掩码对比

![Figure 6](https://arxiv.org/html/2605.16457v2/x6.png)

**说明**: 标准因果注意力掩码 vs. 块因果注意力掩码的 token 依赖关系对比。块因果掩码允许同帧内的 token 相互可见，提供空间上下文；仅对跨帧方向施加因果约束。

### Table 1: Craftax-classic 主要结果（1M 步）

| 方法 | Return (%) @ 0.5M | Score (%) @ 0.5M | Return (%) @ 1M | Score (%) @ 1M |
|------|:-----------------:|:----------------:|:---------------:|:--------------:|
| Human expert | — | — | 65.0 ± 10.5 | 50.5 ± 6.8 |
| DreamerV3 | — | — | 53.2 ± 8.0 | 14.5 ± 1.6 |
| IRIS | — | — | 25.0 ± 3.2 | 6.66 |
| Δ-IRIS | — | — | 35.0 ± 3.2 | 9.30 |
| DART | — | — | 55.45 ± 3.39 | — |
| Dedieu et al. (2025) | — | — | 67.42 ± 0.55 | 27.91 ± 0.63 |
| Dedieu et al. (reproduced)† | 54.32 ± 0.60 | 13.06 ± 0.39 | 68.55 ± 0.72 | 27.24 ± 0.86 |
| **ITC (ours)** | **63.10 ± 1.24** | **20.12 ± 0.80** | **72.46 ± 0.45** | **35.60 ± 0.92** |

**关键发现**: ITC 在 1M 步的 Return 提升 4 个百分点，Score 提升 8 个百分点（相对提升约 30%）；0.5M 步的 Score 已超过基线 1M 步水平。

### Table 2: 组件消融实验（Craftax-classic，1M 步）

| 配置 | Return (%) | Score (%) |
|------|:-----------:|:---------:|
| 基线 Dedieu et al. (2025)† | 68.55 ± 0.72 | 27.24 ± 0.86 |
| + 3D RoPE | 71.85 ± 0.63 | 33.94 ± 1.10 |
| + ITC（完整模型）| **72.46 ± 0.45** | **35.60 ± 0.92** |

**关键发现**: 3D RoPE 贡献了大部分提升（Return +3.3pp，Score +6.7pp）；ITC 在此基础上进一步显著提升 Score（+1.7pp）。

### Table 3: 逐步预测准确率（Craftax-classic）

| 方法 | 总体准确率 (%) | 有生物帧 (%) | 无生物帧 (%) |
|------|:-------------:|:------------:|:------------:|
| Dedieu et al. (2025)† | 46.94 | 33.83 | 61.03 |
| **ITC (ours)** | **51.13** | **38.37** | **65.46** |

**关键发现**: ITC 在包含随机移动生物的帧上提升最大（+4.5pp），说明 ITC 最擅长处理动态实体的 token 对应关系。

### Table 4: 计算开销分析（RTX 3090）

| 方法 | WM 训练 (min/iter) | 想象生成 (min/iter) | 总时间 (hrs) |
|------|:-----------------:|:------------------:|:-----------:|
| Dedieu et al. (2025)† | 8.08 | 4.48 | 46.3 |
| **ITC (ours)** | **8.19 (+1.4%)** | **4.78 (+6.7%)** | **48.2 (+2.8%)** |

**关键发现**: 总体仅增加 2.8% 训练时间，性价比极高。

### Table 5: 超参数敏感性分析

**Table 5a: 距离代价系数 $c_d$（固定 $c_w = 0.3$）**

| $c_d$ | Return (%) | Score (%) |
|:---:|:-----------:|:---------:|
| 0.0 | 72.08 | 32.89 |
| 0.3 | 71.10 | 31.41 |
| **0.6** | **72.46** | **35.60** |
| 0.8 | 71.49 | 33.54 |

**Table 5b: Wildcard 惩罚 $c_w$（固定 $c_d = 0.6$）**

| $c_w$ | Return (%) | Score (%) |
|:---:|:-----------:|:---------:|
| **0.3** | **72.46** | **35.60** |
| 0.6 | 71.60 | 35.41 |

**关键发现**: ITC 对两个超参数均不敏感，证明方法鲁棒性强，且在所有四个基准上使用相同超参数。

### Table 6: Craftax 结果（1M 步）

| 方法 | Return (%) | Score (%) |
|------|:-----------:|:---------:|
| Dedieu et al. (2025) | 5.44 ± 0.25 | 1.53 ± 0.10 |
| Simulus | 6.59 | — |
| **ITC (ours)** | **7.09 ± 0.20** | **2.40 ± 0.04** |

### Table 7: MinAtar 结果（1M 步）

| 方法 | Asterix | Breakout | Freeway | SpaceInvaders |
|------|:-------:|:--------:|:-------:|:-------------:|
| AD | 21.05 ± 0.65 | 27.78 ± 0.16 | 57.68 ± 0.07 | 140.36 ± 1.70 |
| Dedieu et al. (2025) | 44.81 ± 3.54 | 93.92 ± 1.44 | 71.12 ± 0.13 | 186.16 ± 1.25 |
| **ITC (ours)** | **50.04 ± 2.98** | **99.53 ± 2.31** | **71.34 ± 0.07** | **188.85 ± 0.62** |

**关键发现**: ITC 在全部 4 个游戏上均超越基线，说明方法泛化性好，与具体游戏场景无关。

### Table 8: Atari 100K 结果（5 个种子）

| 方法 | IQM (↑) | Optimality Gap (↓) |
|------|:-------:|:------------------:|
| DreamerV3 | 0.487 | 0.510 |
| STORM | 0.561 | 0.472 |
| Diamond | 0.641 | 0.480 |
| Simulus | 0.990 | 0.412 |
| **ITC (ours)** | **1.092** | **0.376** |

**关键发现**: ITC 在使用 VQ-VAE tokenizer 的 Simulus 基础上进一步显著提升 IQM（+10.3%），证明 ITC 与 tokenizer 类型无关。

---

## 实验

### 数据集 / 环境

| 环境 | 观测空间 | 特点 | 用途 |
|------|----------|------|------|
| Craftax-classic | RGB 视觉 | 开放世界、程序化生成、复杂任务层次 | 主要基准 |
| Craftax | RGB 视觉 | 比 classic 更复杂（更大屏幕、更多敌人和关卡）| 泛化测试 |
| MinAtar | 10×10 符号观测 | 简化版 Atari，4 个游戏 | 符号环境泛化 |
| Atari 100K | RGB 视觉 | 26 个多样化游戏，100K 交互 | 大规模 sample efficiency |

### 实现细节

- **Tokenizer**: Patch-lookup（Craftax）/ VQ-VAE（Atari 100K）
- **架构**: 3 个 Transformer 块，8 个注意力头，嵌入维度 128，MLP 隐层 512
- **每帧 Token 数 $L$**: 81
- **Codebook 大小 $K$**: 4096，patch shape $7 \times 7 \times 3$
- **OT 超参**: $c_d = 0.6$，$c_w = 0.3$，Sinkhorn 正则化 $\varepsilon = 1 \times 10^{-5}$，迭代 10 次
- **策略优化器**: Adam，lr = 0.00045
- **总环境交互**: 1M 步（Craftax）/ 100K 步（Atari 100K）
- **硬件**: 单块 Nvidia RTX 3090，总时间 48.2 小时

---

## 批判性思考

### 优点

1. **即插即用**: 不修改 Transformer 架构和训练流程，可直接叠加在现有世界模型上
2. **超低开销**: 仅 2.8% 额外计算成本，收益极为显著
3. **跨 Tokenizer 泛化**: 在 Patch-lookup 和 VQ-VAE 两类 tokenizer 上均有效
4. **超参数鲁棒**: 同一组超参数在四个完全不同的基准上均表现最优

### 局限性

1. **空间坐标假设**: 方法依赖 token 具有明确的 2D 坐标，不适用于无空间结构的潜在空间（如 Dreamer-style 连续潜变量）
2. **离散 Token 限制**: 当前设计基于离散 tokenization，扩展到连续潜空间的路径不明确
3. **距离度量简化**: 仅使用简单欧氏距离惩罚，无法处理瞬移（teleportation）等非连续运动（虽然论文声称 Atari 的结果说明鲁棒性）
4. **高分辨率扩展**: 当每帧 token 数 $L$ 非常大时，OT 求解的复杂度 $O(L^2)$ 可能成为瓶颈

### 潜在改进方向

1. 探索软分配（保留连续 $P$ 而非硬二值化）是否可以更好地处理遮挡和部分对应关系
2. 将 ITC 扩展到连续潜变量世界模型（如 DreamerV3）
3. 引入更高级的距离度量（如学习到的运动模型）替代静态欧氏距离

### 可复现性评估

- [x] 代码开源（GitHub）
- [ ] 预训练模型（暂未提供）
- [x] 训练细节完整（附录有完整超参数表）
- [x] 数据集可获取（Craftax、MinAtar、Atari 均公开）

---

## 关联笔记

### 基于

- [[IRIS]]: 最早将 Transformer 世界模型应用于 RL 的工作，ITC 基于其 token-based 框架
- [[DreamerV3]]: 主要对比基线，代表 latent 世界模型方向
- [[STORM]]: 另一 Transformer 世界模型基线

### 对比

- [[DART]]: 同期 Craftax-classic 基线，ITC 超越其 55.45% Return
- [[Simulus]]: Atari 100K 的直接基线，ITC 在其 VQ-VAE tokenizer 基础上进一步提升

### 方法相关

- [[最优传输]]: ITC 的核心数学工具
- [[Sinkhorn 算法]]: 求解正则化 OT 的高效迭代算法
- [[旋转位置编码]]: ITC 中引入的 3D RoPE，提供时空位置感知
- [[Transformer]]: 基础架构
- [[PPO]]: 策略优化方法

### 数据集/环境相关

- [[Craftax]]: 主要评测环境（procedurally generated open-world）
- [[Atari 100K]]: 大规模泛化评测

---

## 速查卡片

> [!summary] Identifiable Token Correspondence (ITC)
> - **核心**: 将世界模型的下一帧 token 预测建模为最优传输分配问题，token 选择性复制或新生成
> - **方法**: Sinkhorn OT 求解器 + 二值化硬分配 + 3D RoPE，即插即用，不改变架构/训练
> - **结果**: Craftax-classic 72.5% Return（vs. 68.6%），Atari 100K IQM 1.092（vs. 0.990），额外开销仅 2.8%
> - **代码**: [github.com/snu-mllab/Identifiable-Token-Correspondence](https://github.com/snu-mllab/Identifiable-Token-Correspondence)

---

*笔记创建时间: 2026-05-23*
