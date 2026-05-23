---
type: concept
aliases: [Sinkhorn Algorithm, Sinkhorn-Knopp, 正则化最优传输]
---

# Sinkhorn 算法

## 定义

Sinkhorn 算法是求解**熵正则化最优传输问题**的高效迭代算法，通过在传输计划上加入熵惩罚项，将原 LP 问题转化为可用矩阵缩放（iterative matrix scaling）高效求解的形式。

## 数学形式

带熵正则化的最优传输问题：

$$
\min_{P \geq 0} \langle C, P \rangle - \varepsilon H(P) \quad \text{s.t.} \quad P \mathbf{1} = \mathbf{a},\ P^T \mathbf{1} = \mathbf{b}
$$

其中 $H(P) = -\sum_{ij} P_{ij} \log P_{ij}$ 为熵项，$\varepsilon > 0$ 为正则化系数。

**Sinkhorn 迭代**（交替行列归一化）：

$$
u^{(t+1)} = \frac{a}{K v^{(t)}}, \quad v^{(t+1)} = \frac{b}{K^T u^{(t+1)}}
$$

其中 $K_{ij} = \exp(-C_{ij}/\varepsilon)$，最优传输计划为 $P^* = \text{diag}(u) K \text{diag}(v)$。

## 核心要点

1. **时间复杂度**: $O(n^2/\varepsilon^2)$，远优于精确 LP 求解的 $O(n^3 \log n)$
2. **可微分**: 由于是光滑近似，Sinkhorn 输出对输入可微，可端到端训练
3. **调节参数**: $\varepsilon$ 越小，解越接近精确 OT，但迭代收敛越慢
4. **GPU 友好**: 矩阵乘法形式，易于批量并行计算

## 代表工作

- [[ITC]]: 用 Sinkhorn 求解跨帧 token 对应的软分配矩阵（$\varepsilon = 10^{-5}$，10 次迭代）

## 相关概念

- [[最优传输]]
- [[ITC]]
