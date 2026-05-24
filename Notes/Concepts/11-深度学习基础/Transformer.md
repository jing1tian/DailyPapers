---
type: concept
aliases: [Transformer, 注意力机制架构, Attention-based Architecture]
---

# Transformer

## 定义
基于自注意力机制的神经网络架构，由 Vaswani et al.（2017）提出，通过并行计算序列中所有位置对之间的注意力权重，替代 RNN 的顺序处理，成为现代深度学习的基础架构。

## 数学形式

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

## 核心要点
1. **自注意力**（Self-Attention）：序列内每个位置都可以直接关注其他所有位置。
2. **多头注意力**（Multi-Head Attention）：并行运行多组注意力，捕捉不同维度的关系。
3. **前馈网络**（FFN）：每个位置独立经过两层线性变换 + 激活函数。
4. **位置编码**：通过加性或旋转位置编码注入序列位置信息。
5. 编码器-解码器结构用于 seq2seq；仅编码器（BERT）用于理解；仅解码器（GPT）用于生成。

## 代表工作
- Vaswani et al.（2017）："Attention is All You Need"
- [[OrbiSim]]: OrbiSim-Dynamics 使用基于 Transformer 的耦合模块建模多实体物理交互

## 相关概念
- [[多头自注意力]]
- [[交叉注意力]]
- [[自注意力机制]]
- [[层归一化]]
- [[AdaLN]]
