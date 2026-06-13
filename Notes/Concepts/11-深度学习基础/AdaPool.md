---
type: concept
aliases: [Adaptive Pooling, 自适应池化]
---

# AdaPool

## 定义

AdaPool（Adaptive Pooling）是一种自适应 token 聚合方法，通过学习每个 token 的重要性权重来加权聚合特征序列，相比均值池化或最大池化能更好地保留语义显著信息。

## 数学形式

$$
\text{AdaPool}(\mathbf{Z}) = \sum_{i=1}^{N} w_i \cdot z_i, \quad w_i = \text{softmax}(f_w(z_i))
$$

其中 $f_w$ 为可学习的线性变换，$N$ 为 token 数量。

## 核心要点

1. 权重由网络自动学习，不依赖手工设计的聚合规则
2. 相比全局平均池化，对局部显著区域有更强的响应
3. 计算开销低，仅需一次线性变换 + softmax
4. 常用于将变长序列压缩为固定维度表示，再接 MLP 做分类/回归

## 代表工作

- [[WEAVER]]: 用于奖励头和 Critic 网络中，将 $H \times W$ 个潜在 token 聚合为单一状态向量，结合语言指令 $\ell$ 预测任务成功概率

## 相关概念

- [[Transformer]]
- [[Attention Pooling]]
- [[流匹配]]
