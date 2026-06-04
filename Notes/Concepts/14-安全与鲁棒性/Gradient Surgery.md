---
type: concept
aliases: [梯度外科手术]
---

# Gradient Surgery（梯度外科手术）

## 定义

Gradient Surgery 是一种多任务梯度协调方法，当两个任务的梯度存在冲突（余弦相似度为负）时，将其中一个任务的梯度投影到另一个任务梯度的法平面上，消除冲突分量。

## 数学形式

对于任务 $i$ 的梯度 $g_i$ 和任务 $j$ 的梯度 $g_j$，若 $g_i \cdot g_j < 0$，则对 $g_i$ 进行投影：

$$
g_i \leftarrow g_i - \frac{g_i \cdot g_j}{\|g_j\|^2} g_j
$$

## 核心要点

1. 仅在梯度冲突时触发（余弦相似度 < 0），不冲突时保持原梯度
2. PCGrad 是 Gradient Surgery 的一个具体实现，同时对所有任务对执行双向投影
3. 不改变模型架构，只修改优化步骤，即插即用

## 代表工作

- [[GeoSem-WAM]]: 作者建议未来工作中引入 Gradient Surgery 缓解多任务梯度冲突

## 相关概念

- [[Gradient Conflict]]
- [[PCGrad]]
- [[Multi-Task Learning]]
