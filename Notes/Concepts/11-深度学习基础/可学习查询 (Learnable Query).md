---
type: concept
aliases: [Learnable Query, Learnable Queries, 可学习查询向量, Perceiver Query]
---

# 可学习查询 (Learnable Query)

## 定义

一组随机初始化并参与端到端训练的向量，通过交叉注意力机制从高维上下文特征（如图像 token 序列）中提取固定数量的紧凑表示，常见于 Perceiver、Q-Former、Flamingo 等架构。

## 数学形式

$$
\text{Output} = \text{CrossAttn}(Q_{\text{learnable}},\; K=V=\text{Context})
$$

其中 $Q_{\text{learnable}} \in \mathbb{R}^{N \times d}$（$N$ 为查询数，$d$ 为特征维度）是随机初始化并参与梯度更新的参数矩阵。

## 核心要点

1. 将变长上下文压缩为固定数量的向量，解耦下游模块与上下文长度。
2. 查询向量隐式学习"关注什么"，无需手工设计特征选择规则。
3. 查询数是超参数：数量越多信息越丰富，但计算开销越大；实验表明 50 个查询与 256 个效果相当（见 PDRVLA Table 15）。

## 代表工作

- [[PDRVLA]]: 50 个可学习查询提取物理 token（Physical Dynamics Module）
- Perceiver / Perceiver IO: 最早系统化使用该设计
- Q-Former（BLIP-2）: 视觉语言对齐中的代表性应用

## 相关概念

- [[Gated Cross-Attention]]
- [[交叉注意力]]
- [[Physical Dynamics Module]]
