---
type: concept
aliases: [Sinkhorn Algorithm, Hungarian Algorithm Approximation]
---

# Sinkhorn-Knopp

## 定义
Sinkhorn-Knopp 算法：求解最优传输（Optimal Transport）问题的高效迭代算法，通过交替行/列归一化求双随机矩阵。

## 数学形式
$$P^{(t+1)} = \text{diag}(u)^{-1} K \text{diag}(v)^{-1}$$
其中 $K_{ij} = \exp(-C_{ij}/\varepsilon)$，$u, v$ 为迭代更新的缩放向量。

## 核心要点
1. 每次迭代只需矩阵乘法，时间复杂度低
2. $\varepsilon$ 控制熵正则化强度（小 $\varepsilon$ 趋近精确 OT）
3. 可 GPU 并行化，适合大规模深度学习场景

## 代表工作
- [[EmbodiedVAE]]: 用 Sinkhorn-Knopp 求解 content/motion 解耦的最优传输

## 相关概念
- [[EmbodiedVAE]]
- [[OT]]
