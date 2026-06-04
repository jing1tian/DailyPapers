---
type: concept
aliases: [Projecting Conflicting Gradients]
---

# PCGrad

## 定义

PCGrad（Projecting Conflicting Gradients）是一种多任务学习优化方法，对每对任务梯度执行双向投影以消除冲突分量，使各任务的更新方向相互协调。

## 数学形式

对于任务 $i$ 和任务 $j$，若 $g_i \cdot g_j < 0$：

$$
g_i^{\text{PC}} = g_i - \frac{g_i \cdot g_j}{\|g_j\|^2} g_j
$$

最终梯度为所有任务修正后梯度的均值。

## 核心要点

1. 双向投影：对冲突的每对任务 $(i, j)$ 都进行双向修正
2. 随机顺序处理任务对，减少顺序依赖
3. 实现简单，可直接封装在优化器中
4. Yu et al., NeurIPS 2020 提出

## 代表工作

- [[GeoSem-WAM]]: 作者建议引入 PCGrad 解决多任务梯度冲突

## 相关概念

- [[Gradient Conflict]]
- [[Gradient Surgery]]
- [[Multi-Task Learning]]
