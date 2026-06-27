---
type: concept
aliases: [QK-Norm, Query-Key Normalization, QK 归一化]
---

# QK-Norm

## 定义

QK-Norm（Query-Key Normalization）是一种在注意力机制计算 attention score 之前对 Query 和 Key 向量分别做归一化（如 RMSNorm）的技术，用于稳定大规模 Transformer 训练中的注意力数值范围，缓解注意力 logits 爆炸问题。

## 数学形式

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{\widehat{Q}\,\widehat{K}^\top}{\sqrt{d_h}}\right)V, \qquad \widehat{Q} = \frac{Q}{\lVert Q \rVert}, \;\; \widehat{K} = \frac{K}{\lVert K \rVert}
$$

## 核心要点

1. **数值稳定性**: 防止 Query/Key 向量范数随训练增长导致 softmax 前 logits 过大，从而稳定大 batch size / 长训练下的注意力分布
2. **常见于大规模训练**: 在大规模视觉/语言/动作模型的注意力池化（attention pooling）模块中广泛使用，尤其是条件信息聚合场景
3. **与标准注意力的区别**: 仅在计算 score 前对 Q、K 做归一化，不改变 V 的计算方式

## 代表工作

- [[ABC]]: 在 ABC-VLA 的注意力池化（attention pooling）模块中使用 QK-Norm 提升大 batch size 训练下的数值稳定性

## 相关概念

- [[Cross-Attention]]
- [[Diffusion Transformer (DiT)|DiT]]
