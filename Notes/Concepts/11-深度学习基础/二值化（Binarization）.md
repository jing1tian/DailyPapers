---
type: concept
aliases: [Binarization, 硬分配, Hard Assignment, 离散化]
---

# 二值化（Binarization）

## 定义

二值化是将连续值矩阵（如软分配概率矩阵）转换为 0-1 硬分配矩阵的过程，使得每个目标元素恰好对应一个唯一的来源。

## 数学形式

给定软分配矩阵 $P \in [0,1]^{n \times m}$，贪心列优先 argmax 二值化：

$$
\Pi_{ij} = \begin{cases} 1 & \text{if } i = \arg\max_k P_{kj} \text{ and no conflict} \\ 0 & \text{otherwise} \end{cases}
$$

满足约束：每列恰好有一个 1，每行最多有一个 1（部分传输）。

## 核心要点

1. **贪心策略**: 按列逐一选择得分最高的源，冲突时按优先级重分配
2. **迭代收敛**: 冲突解决通过迭代进行，直到所有分配稳定（见 ITC Algorithm 2）
3. **对比 argmax**: 简单全局 argmax 可能导致多个列争抢同一行；ITC 的迭代过程解决冲突
4. **与 Gumbel-Softmax 区别**: Gumbel-Softmax 通过温度趋近 0 实现连续近似，二值化则是直接硬化

## 代表工作

- [[ITC]]: 将 Sinkhorn 输出的连续传输计划二值化为严格 0-1 的硬分配，决定每个 token 复制还是生成

## 相关概念

- [[最优传输]]
- [[Sinkhorn 算法]]
- [[ITC]]
