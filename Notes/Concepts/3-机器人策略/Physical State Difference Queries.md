---
type: concept
aliases: [物理状态差异查询, Difference Queries, State Transition Queries]
---

# Physical State Difference Queries

## 定义

VLA-Δ 中提出的可学习 Query 集合，通过将其追加到视觉-语言上下文末尾并对骨干网络施加约束，将观测中的物理状态变化信息压缩为紧凑的转变表示 $Z_t$。

## 数学形式

$$
Z_t = \text{Readout}_Q(F_\theta([C_t; Q])) \in \mathbb{R}^{N_q \times d}
$$

其中差异文本 Token 仅可 attend 到 Queries 和前面的文本，形成信息瓶颈。

## 核心要点

1. **信息瓶颈设计**：所有观测与指令信息必须经过 Queries 才能到达下游差异文本生成，强迫模型学习物理状态的紧凑表示
2. **分组结构**：$N_q$ 个 Queries 被分为 7 个功能组（G1-G7），分别对应不同物理状态类别（进度、位置、接触等）
3. **多路监督**：预训练时同时接受结构化槽位、稠密光流和差异文本三路监督信号
4. **推理零开销**：微调和推理时只需提取 $Z_t$，无需解码中间目标

## 代表工作

- [[VLA-Delta]]: 首次提出 Physical State Difference Queries，用于 VLA 物理状态转变建模

## 相关概念

- [[Physical State Transition]]
- [[Structured Transition Slots]]
- [[Dense Optical Flow]]
- [[Cross-Attention]]
- [[GR00T N1]]
