---
type: concept
aliases: [双向注意力, Bidirectional Attention, 双向交叉注意力]
---

# Bidirectional Attention

## 定义

双向注意力允许两组 token 序列互相完整关注对方（$A \to B$ 且 $B \to A$），相较于单向因果注意力，双向注意力使两路表示能够充分交互融合。

## 数学形式

对于两组 token $X_A$ 和 $X_B$，双向注意力掩码为：

$$
M = \begin{pmatrix}1 & 1 \\ 1 & 1\end{pmatrix}
$$

即 $A$ 能看到 $B$，$B$ 也能看到 $A$，形成完整的交叉感知。

## 核心要点

1. 与因果（单向）注意力相比，双向注意力没有序列方向约束
2. 常见于 BERT 类编码器、跨模态融合模块
3. 在多流模型中用于让不同模态/功能流互相感知

## 代表工作

- [[LoopWAM]]: 视频流（$V$）与动作流（$A$）之间使用双向注意力，使预测的未来视频状态反哺动作生成

## 相关概念

- [[Diffusion Transformer]]
- [[Mixture-of-Transformers]]
- [[World Action Model]]
