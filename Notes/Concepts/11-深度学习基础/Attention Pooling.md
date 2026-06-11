---
type: concept
aliases: [注意力池化, Attention-based Pooling]
---

# Attention Pooling

## 定义
一种利用注意力机制对序列进行自适应聚合的池化方法，通过可学习的查询向量动态加权计算输入序列的加权和，而非固定的平均或最大池化。

## 数学形式

$$
\text{AttnPool}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

其中 $Q$ 为可学习查询，$K, V$ 来自输入序列。

## 核心要点
1. 相比平均池化，能够聚焦于序列中信息量更大的位置
2. 支持变长序列到固定维度表示的映射
3. 在时序压缩场景中（如技能发现）可学习边界感知的聚合策略

## 代表工作
- [[HiMem-WAM]]: 分层技能发现中用于将低层潜变量序列压缩为高层技能表示

## 相关概念
- [[Transformer]]
- [[Hierarchical Latent Action]]
- [[Action Chunking]]
