---
type: concept
aliases: [自注意力, Self Attention, 自注意力机制, Scaled Dot-Product Attention]
---

# Self-Attention

## 定义

Transformer 的核心计算单元，序列中每个元素通过与序列内所有其他元素计算相似度（注意力权重）后加权聚合信息，实现长距离依赖建模。

## 数学形式

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

其中查询 $Q$、键 $K$、值 $V$ 均来自同一输入序列：$Q = XW_Q,\ K = XW_K,\ V = XW_V$。

## 核心要点

1. **全局感受野**: 每个 token 可直接与序列中任意位置的 token 交互，不受局部窗口限制。
2. **多模态融合**: 在 JOPAT 等模型中，将动作、视觉、轨迹三类 token 拼接后做全局双向 self-attention，实现跨模态信息融合。
3. **计算复杂度**: $O(n^2 d)$，序列过长时计算代价显著增大。
4. **Register Token**: 可学习的全局 token，通过 self-attention 与序列中所有其他 token 交互，起到信息汇聚中心的作用。

## 代表工作

- [[JOPAT]]: 在 DiT 中对三模态联合序列做全局双向 self-attention
- [[Diffusion Transformer (DiT)]]: 基于 self-attention 的扩散模型主干

## 相关概念

- [[Diffusion Transformer (DiT)]]
- [[AdaLN]]
- [[World Action Model]]
