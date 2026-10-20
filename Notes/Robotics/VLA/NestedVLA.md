---
title: "NestedVLA: Learning to Consolidate and Generate Skills for Vision-Language-Action Models"
method_name: "NestedVLA"
authors: [Guodong Du, Wanyu Lin]
year: 2026
venue: arXiv
tags: [vla, skill-consolidation, hypernetwork, model-merging, parameter-efficient, multi-task, robot-manipulation]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2610.23271v1
created: 2026-10-20
---

# 论文笔记：NestedVLA: Learning to Consolidate and Generate Skills for Vision-Language-Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Department of Computing, The Hong Kong Polytechnic University |
| 日期 | October 2026 |
| 项目主页 | — |
| 对比基线 | [[模型合并\|Model Merging]] (TIES, WUDI, Free-Merging 等) |
| 链接 | [arXiv](https://arxiv.org/abs/2610.23271) / [Code](https://github.com/WanyuGroup/NeurIPS2026-NestedVLA) |

---

## 一句话总结

> NestedVLA 用一个嵌套[[Hypernetwork|超网络]]为每个任务**生成**专属的 Action Expert 参数，避免跨任务模型合并中的参数冲突，在仅 1.5× 存储开销下平均超越最强合并基线 41.1 个百分点。

---

## 核心贡献

1. **Post-hoc 技能整合框架**: 将独立在不同任务/平台微调的 VLA 策略（[[任务向量]]）合并到单一系统，无需重新联合训练，规避任务干扰问题。
2. **嵌套超网络（Nested Hypernetwork）**: 设计 $\mathcal{G}_\phi(m)$，以任务 ID 为条件，生成任务专属的 [[Action Expert]] 参数增量，代替静态[[模型合并]]操作。
3. **Drop-and-Rescale 压缩**: 提出稀疏保留 + 全局缩放的轻量压缩算子，将每个任务的 VLM 残差压缩存储，实现约 1.5× 总存储倍率。

---

## 问题背景

### 要解决的问题

[[VLA]] 模型（如 [[Pi05|π₀.₅]]）通常在各任务/平台分别独立微调，产生多个"技能检查点"。如何将这些独立技能**合并**到一个单一模型、同时保留各任务性能？

### 现有方法的局限

- **联合多任务训练**: 任务间梯度干扰（Task Interference）导致性能下降，且无法在不重新训练的前提下增量添加新任务。
- **静态[[模型合并]]（Averaging / TIES / WUDI / Free-Merging）**: 将参数向量直接叠加，[[Action Expert]] 模块对合并最敏感，在 MetaWorld 等任务上成功率从 72% 暴跌至 3-10%；CALVIN 任务完全失效（0%）。
- **动态合并（TALL-Mask / EMR-Merging / Twin-Merging）**: 稍有改善但仍远不够，平均成功率约 22-26%，且 CALVIN 仍接近 0%。

### 本文的动机

- 诊断发现：VLA 中 Action Expert 权重的任务间 cosine 相似度极低，方向干扰严重（Interference Factor 高）；VLM 主体虽然静态合并也不可行，但 VLM 的改变可通过压缩残差独立保存。
- 解决思路：**生成而非合并**——用超网络按任务需求生成 Action Expert，再叠加各任务压缩 VLM 残差，从根本上绕开参数冲突。

---

## 方法详解

### 模型架构

**NestedVLA** 采用**超网络条件生成 + 压缩残差**架构，分三阶段执行：

- **阶段 1（Skill Acquisition）**: 各任务独立在冻结 [[Pi05|π₀.₅]] 基座上微调，得到任务 $m$ 的完整参数 $\Theta_m$
- **阶段 2（Skill Consolidation）**: 训练超网络 $\mathcal{G}_\phi$，学习从任务 ID $m$ 重构 Action Expert 增量；同时对 VLM 增量做 [[Drop-and-Rescale]] 压缩
- **阶段 3（Skill Generation / 推理）**: 以任务 ID 为输入调用超网络，实时生成 Action Expert；加载对应压缩 VLM 残差，恢复完整参数

**模型规模**: 超网络约 430M 可训练参数

### 核心模块

#### 模块 1：嵌套超网络 $\mathcal{G}_\phi$（Nested Hypernetwork）

**设计动机**: [[Action Expert]] 模块参数因任务高度差异化，无法直接合并；[[Hypernetwork|超网络]]可学习一个任务到参数空间的隐式映射，以较少额外参数拟合多任务参数分布。

**具体实现**:
- **Per-task Patch Embedding**: 将 Action Expert 权重张量切割为 patch，每个任务学习独立的 patch embedding 作为超网络的任务条件输入
- **Task Encoder**: 聚合 patch embeddings，产生任务级全局表示
- **共享 FC Trunk + Per-task [[LoRA]] Path**: 主干全连接网络由所有任务共享，LoRA 支路提供任务特定的低秩调制
- **多头输出（Multi-head Output）**: 每个权重矩阵对应独立输出头，生成对应的参数增量

#### 模块 2：Drop-and-Rescale 压缩（VLM 残差存储）

**设计动机**: VLM 主体静态合并完全失效（CALVIN 降至 0%），因此每个任务必须独立保存 VLM 增量；但原始 VLM 增量存储开销巨大，需要压缩。

**具体实现**:
- 对每个权重模块 $p$，按绝对值取 Top-K（$K=10\%$）的参数保留，其余置零
- 全局重缩放系数 $\alpha=1.1$ 补偿截断引入的幅度损失
- 实验对比表明此 Drop-and-Rescale 优于 1-bit 量化和 SVD 残差方法（Table 7）

---

## 关键公式

### 公式 1：[[任务向量|任务向量（Task Vector）]]

$$
\tau_m = \Theta_m - \Theta_0
$$

**含义**: 任务 $m$ 的技能向量为微调后参数与基础参数之差，代表该任务在参数空间中的"方向"。

**符号说明**:
- $\tau_m$: 任务 $m$ 的任务向量（task vector）
- $\Theta_m$: 任务 $m$ 的微调后完整参数
- $\Theta_0$: 共享的预训练基础模型参数

---

### 公式 2：[[Drop-and-Rescale|Drop-and-Rescale 压缩]]

$$
z_m^{(p)} = \alpha \cdot \tau_m^{(p)} \odot M_m^{(p)}
$$

**含义**: 对任务 $m$ 的 VLM 权重模块 $p$，用二值掩码 $M_m^{(p)}$ 保留幅值最大的 Top-K 位置，再全局乘以重缩放系数 $\alpha$。

**符号说明**:
- $z_m^{(p)}$: 压缩后的 VLM 残差（稀疏表示）
- $\alpha$: 全局重缩放系数（实验中取 $1.1$）
- $\tau_m^{(p)}$: 任务 $m$ 在模块 $p$ 的原始任务向量
- $M_m^{(p)}$: Top-K 二值掩码（$K=10\%$）
- $\odot$: 逐元素乘积（Hadamard 积）

---

### 公式 3：[[Hypernetwork|超网络生成训练损失]]

$$
\mathcal{L}(\phi) = \sum_{m=1}^{M} w_m \cdot \frac{1}{|M_m^{A}|} \left\| \left( \mathcal{G}_\phi(m) - z_m^{A} \right) \odot M_m^{A} \right\|_2^2
$$

**含义**: 对所有任务，仅在 Action Expert 的 Top-K 保留位置（$M_m^A$ 掩码内）计算超网络生成参数与目标压缩参数之间的加权均方误差。

**符号说明**:
- $M$: 任务总数（本文 $M=3$）
- $w_m$: 任务 $m$ 的权重系数
- $\mathcal{G}_\phi(m)$: 超网络以任务 $m$ 为条件生成的 Action Expert 增量
- $z_m^{A}$: 任务 $m$ 的 Action Expert 压缩目标（Top-K 稀疏）
- $M_m^A$: 任务 $m$ 的 Action Expert Top-K 掩码
- $|M_m^A|$: 掩码内非零元素数（归一化分母）

---

### 公式 4：推理时参数合成

$$
\hat{\Theta}_m^{A} = \Theta_0^{A} + \alpha \cdot \mathcal{G}_\phi(m) \odot M_m^{A}
$$

$$
\hat{\Theta}_m^{V} = \Theta_0^{V} + z_m^{V}
$$

**含义**: 推理时对任务 $m$，超网络生成并掩码、缩放后的增量加回基础 Action Expert；压缩后的 VLM 残差直接加回基础 VLM。

**符号说明**:
- $\hat{\Theta}_m^{A}$: 任务 $m$ 的最终 Action Expert 参数
- $\hat{\Theta}_m^{V}$: 任务 $m$ 的最终 VLM 参数
- $\Theta_0^{A}, \Theta_0^{V}$: 基础 Action Expert / VLM 参数（冻结）

---

### 公式 5：总存储量分析

$$
\text{storage}_{\text{NestedVLA}} = 1 + M \cdot \left(K_{\text{VLM}} + \frac{1}{32}\right) \cdot \frac{P_{\text{VLM}}}{P_{\text{base}}} + M \cdot \frac{1}{32} \cdot \frac{P_{\text{act}}}{P_{\text{base}}} + \frac{P_{\mathcal{G}_\phi}}{P_{\text{base}}}
$$

**含义**: 总存储 = 基础模型 + 各任务 VLM 稀疏值与掩码 + 各任务 Action Expert 掩码 + 超网络参数，实验中约为 1.48×。

**符号说明**:
- $M$: 任务数（本文为 3）
- $K_{\text{VLM}}$: VLM 残差的 Keep Ratio（实验中 0.1）
- $\frac{1}{32}$: 存储 1-bit 掩码的额外开销
- $P_{\text{VLM}}, P_{\text{act}}, P_{\mathcal{G}_\phi}$: VLM、Action Expert、超网络参数量
- $P_{\text{base}}$: 基础模型总参数量

---

### 公式 6：权重空间干扰度量（Appendix E）

$$
\mathrm{IF} = 1 - \frac{\left\|\sum_{m} \delta_m\right\|_2}{\sum_m \|\delta_m\|_2}
$$

**含义**: 干扰因子（Interference Factor）衡量多个任务向量叠加后的幅值损失，IF 越高说明参数方向冲突越严重，模型合并越困难。

**符号说明**:
- $\delta_m$: 任务 $m$ 相对于预训练参考点的参数增量
- $\left\|\sum_m \delta_m\right\|_2$: 叠加后向量的 $\ell_2$ 范数
- $\sum_m \|\delta_m\|_2$: 各任务向量 $\ell_2$ 范数之和（理想无干扰的上界）

---

## 关键图表

### Figure 1：NestedVLA 动机与概览

![Figure 1](https://arxiv.org/html/2610.23271v1/figure1_nestedvla_v1.png)

**说明**: 左侧为传统[[模型合并]]方案——将多个独立微调的 VLA 检查点合并为一个共享模型，由于参数方向冲突导致严重性能损失。右侧为 NestedVLA——保留压缩的任务专属 VLM 残差，以[[Hypernetwork|超网络]]动态生成 [[Action Expert]] 增量，从根本上规避参数冲突。

---

### Figure 2：NestedVLA 三阶段流程

![Figure 2](https://arxiv.org/html/2610.23271v1/main_nested.png)

**说明**: 完整的 NestedVLA Pipeline。**阶段 1（Skill Acquisition）**: 各任务独立微调，得到任务向量 $\tau_m$；**阶段 2（Skill Consolidation）**: 训练[[Hypernetwork|超网络]] $\mathcal{G}_\phi$，同时对 VLM 残差做 [[Drop-and-Rescale]] 压缩；**阶段 3（Skill Generation）**: 推理时以任务 ID 调用超网络生成 Action Expert，加载压缩 VLM 残差。

---

### Figure 3：嵌套超网络架构

![Figure 3](https://arxiv.org/html/2610.23271v1/hypernetwork.png)

**说明**: 超网络 $\mathcal{G}_\phi$ 的内部架构。每个任务有独立的 Per-task Patch Embedding，经 Task Encoder 得到全局任务表示；共享 FC Trunk 处理主干特征，Per-task [[LoRA]] 支路提供任务特定调制；Multi-head 输出生成各权重模块的参数增量。

---

### Table 1：主实验结果（成功率 %）

| 方法 | MetaWorld | ManiSkill | CALVIN | 平均 |
|------|-----------|-----------|--------|------|
| Per-task FlowSDE（上界） | 72.3 | 88.1 | 81.3 | 80.5 |
| Averaging | 3.1 | 41.3 | 0.0 | 14.8 |
| TIES | 2.0 | 37.2 | 0.0 | 13.1 |
| WUDI | 2.7 | 32.8 | 0.0 | 11.8 |
| TSV-Merging | 3.9 | 36.3 | 0.0 | 13.4 |
| TALL-Mask | 8.1 | 53.8 | 6.0 | 22.6 |
| EMR-Merging | 8.2 | 54.4 | 6.3 | 23.0 |
| Twin-Merging | 10.4 | 56.9 | 8.7 | 25.3 |
| Free-Merging | 9.4 | 58.3 | 9.1 | 25.6 |
| **NestedVLA（1.5× 存储）** | **37.9** | **87.2** | **75.0** | **66.7** |

**关键发现**: NestedVLA 以 1.5× 存储开销平均超越最强合并基线 Free-Merging **41.1 pp**；在 ManiSkill 上与 per-task 上界仅差 0.9 pp；CALVIN 超越最佳基线 65.9 pp（9.1% → 75.0%）。

---

### Table 2：超网络架构消融（In-mask 重建损失，越低越好）

| 配置 | MetaWorld | ManiSkill | CALVIN |
|------|-----------|-----------|--------|
| 完整模型（默认） | 1.08 | 1.17 | 1.14 |
| 移除多头（Multi-head）路径 | 1.33 | 1.71 | 1.36 |
| 移除 [[LoRA]] 路径 | **0.91** | **0.92** | **0.89** |

**关键发现**: 移除 Multi-head 输出头对重建损失影响最大（+0.25~+0.54）；移除 LoRA 路径反而略微降低损失，说明在该配置下 LoRA 对重建损失贡献有限，但其在实际任务成功率上的作用需进一步验证（作者保留 LoRA 路径）。

---

### Table 3：理想重建上界分析（Table 3）

| Keep Ratio (K) | MetaWorld | ManiSkill | CALVIN |
|----------------|-----------|-----------|--------|
| K=10%（当前） | 48.4 | 87.5 | 78.1 |
| K=20% | 60.2 | 87.5 | 85.4 |
| NestedVLA 实际（K=10%） | 37.9 | 87.2 | 75.0 |

**关键发现**: MetaWorld 的性能差距（48.4 vs 37.9）主要来自稀疏预算不足（K=10% 的 Drop-and-Rescale 压缩限制），而非超网络设计本身；提高 K 至 20% 可将 MetaWorld 上界提升至 60.2%。

---

### Table 7：压缩算子对比

| 压缩方法 | 重建损失 / 任务成功率 |
|----------|----------------------|
| 1-bit 量化 | 严重失效 |
| SVD 残差 | 与 Drop-and-Rescale 相近，但存储成本更高 |
| **Drop-and-Rescale（本文）** | **匹配或优于 SVD，存储更低** |

**关键发现**: Drop-and-Rescale 在相同存储预算下优于 SVD 残差，且大幅优于 1-bit 量化。

---

## 实验

### 数据集 / 基准

| 基准 | 特点 | 任务数 | 用途 |
|------|------|--------|------|
| MetaWorld | 密集操作任务（较多任务干扰） | 3 | 训练 + 测试 |
| ManiSkill | 多样化操作，含视觉挑战 | 3 | 训练 + 测试 |
| CALVIN | 长序列语言条件操作，挑战最强 | 3 | 训练 + 测试 |

### 实现细节

- **基础 VLA**: [[Pi05|π₀.₅]]（FlowSDE 训练变体），冻结预训练参数
- **超网络可训练参数**: ~430M
- **训练步数**: 6,000 steps
- **优化器**: AdamW，学习率 $3 \times 10^{-4}$
- **Keep Ratio**: $K = 10\%$
- **重缩放系数**: $\alpha = 1.1$
- **实验环境**: 仿真环境（无真实机器人实验）

### 重要分析

**VLM 静态合并失效验证（Appendix F）**: 仅合并 VLM 部分（Action Expert 保留 per-task），TIES 和 WUDI 仍导致 CALVIN 降至 0%，验证 VLM 残差必须独立保存。

**损失侧消融（Appendix G）**: 去除 Out-of-mask 惩罚项（$w_{out}=0$）可提升约 +2.4 pp 平均成功率；绝对值损失空间和幅值加权各自单独使用有损，组合使用可近似恢复性能。最终默认采用标准化损失 + $w_{out}=0$。

---

## 批判性思考

### 优点
1. **从根本上解决参数冲突**: 超网络生成取代静态合并，彻底规避方向干扰问题，在 ManiSkill 几乎达到 per-task 上界。
2. **存储效率合理**: 1.5× 存储实现 66.7% vs 25.6%（最强基线），每个任务增量约 0.15× 基础模型大小。
3. **清晰的诊断链**: 论文通过权重空间诊断（IF）和 VLM 静态合并消融，清晰地建立了方法设计的因果动机。

### 局限性
1. **仅整合 3 个技能**: 任务数量有限（M=3），超网络对更多任务的可扩展性未验证；合并更多任务时超网络参数量可能显著增长。
2. **VLM 仍需 per-task 存储**: 每个任务的 VLM 残差仍需独立保存，无法共享（future work 提及"v2 版本"），限制了存储效率的进一步提升。
3. **无法增量添加新任务**: 加入新任务需要重新训练超网络，不支持持续学习场景。
4. **仅仿真验证**: 所有实验在仿真中进行，实际机器人的噪声、延迟和感知误差等未测试。
5. **MetaWorld 差距仍较大**: 37.9% vs 72.3% per-task 上界，作者虽归因于稀疏预算，但实际系统距离实用仍有距离。

### 潜在改进方向
1. 扩展超网络以生成 VLM 参数增量（作者提到的"v2"方向），进一步降低 VLM 的 per-task 存储
2. 设计增量超网络训练机制，支持在不重训的情况下添加新技能
3. 提高 Action Expert 的 Keep Ratio（$K>10\%$）或采用自适应压缩以改善 MetaWorld 性能
4. 在真实机器人平台上验证性能

### 可复现性评估
- [x] 代码开源（GitHub: WanyuGroup/NeurIPS2026-NestedVLA）
- [ ] 预训练模型（未明确提及）
- [x] 训练细节完整（Table 4 & 5 提供了详细超参数）
- [x] 数据集可获取（MetaWorld / ManiSkill / CALVIN 均为公开 benchmark）

---

## 关联笔记

### 基于
- [[Pi05|π₀.₅]]: 基础 VLA 模型，NestedVLA 在其预训练参数上独立微调并整合
- [[任务向量]]: 技能表示的核心数学工具
- [[Hypernetwork|超网络]]: 超网络生成参数的基础概念

### 对比
- [[模型合并]]: NestedVLA 直接对比并超越的所有静态/动态合并基线
- Free-Merging: 最强合并基线（25.6%），NestedVLA 平均超越 41.1 pp

### 方法相关
- [[Drop-and-Rescale]]: VLM 残差压缩的核心操作
- [[LoRA]]: 超网络内部的任务特定低秩调制路径
- [[Action Expert]]: NestedVLA 超网络生成的目标模块

### 硬件/数据相关
- MetaWorld: 密集操作基准（仿真）
- ManiSkill: 多样化操作基准（仿真）
- CALVIN: 长序列语言条件操作基准（仿真）

---

## 速查卡片

> [!summary] NestedVLA
> - **核心**: 用嵌套超网络为每个任务生成专属 Action Expert 参数，规避跨任务模型合并的参数冲突
> - **方法**: Nested Hypernetwork $\mathcal{G}_\phi(m)$ + Drop-and-Rescale VLM 残差压缩，共 1.5× 存储
> - **结果**: 平均 66.7% 成功率，超越最强合并基线 41.1 pp；ManiSkill 几乎达到 per-task 上界
> - **代码**: [GitHub](https://github.com/WanyuGroup/NeurIPS2026-NestedVLA)

---

*笔记创建时间: 2026-10-20*
