---
type: concept
aliases: [因果注意力, Causal Self-Attention, Masked Causal Attention]
---

# Causal Attention

## 定义

因果注意力（Causal Attention）是一种受限的自注意力机制，每个 token 只能关注序列中当前位置及之前的 token，通过注意力掩码（causal mask）强制单向信息流，是自回归生成模型的核心组件。

## 核心要点

1. **单向性**: 位置 $i$ 的 token 只能看到 $j \leq i$ 的 token，确保生成时不会"看到未来"
2. **实现**: 在注意力矩阵上施加上三角掩码（将未来位置设为 $-\infty$，softmax 后趋近于 0）
3. **视频生成中的扩展**:
   - **Segment-wise Causal Attention**: 片段内部全注意力，片段间用因果掩码（Hy-Nomas、Genie 2 等）
   - **Block Causal Attention**: 以固定大小的 block 为单位施加因果约束

## 数学形式

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V
$$

其中掩码 $M_{ij} = \begin{cases} 0 & j \leq i \\ -\infty & j > i \end{cases}$

## 代表工作

- [[HyNomas]]: Segment-wise Causal Attention + Action-Aware Causal Attention，区分描述型和动作型 token 的跨片段可见性
- GPT 系列: 标准因果自注意力的经典应用
- [[CausVid]]: 视频生成中的因果注意力应用

## 相关概念

- [[Attention Mechanism]]
- [[Bidirectional Attention]]
- [[Block Causal Attention]]
