---
type: concept
aliases: [Markley Quaternion Averaging, 四元数加权均值, Quaternion Mean]
---

# Markley 四元数均值

## 定义

对多个旋转四元数进行**加权平均**的数学方法：构造加权外积矩阵，取其**最大特征向量**作为融合后的四元数，保证结果仍为单位四元数（合法旋转）。

## 数学形式

$$
M = \sum_{n \in \mathcal{E}_t} w_n q_n q_n^{\top}
$$

$$
\hat{q} \in \operatorname*{arg\,max}_{\|q\|_2=1} q^{\top} M q
$$

即 $\hat{q}$ 为对称矩阵 $M$ 的最大特征向量。

## 核心要点

1. **为什么不能直接线性插值**：四元数空间非线性，线性平均会偏离单位球面，导致非法旋转
2. **最大特征向量解**：等价于最小化各 $q_n$ 与 $\hat{q}$ 之间的加权球面测地距离
3. **与 SLERP 的区别**：SLERP 只处理两个四元数，Markley 方法可处理任意多个
4. **平移部分**：位置（平移）分量用普通加权均值，只有旋转用 Markley 方法

## 代表工作

- [[LAW]]: 将 Markley 方法用于 top-K 候选动作块中四元数部分的加权融合

## 相关概念

- [[Action Chunking]]
- [[Value Function]]
- [[子任务局部价值学习]]
