---
type: concept
aliases: [梯度冲突, 梯度干扰]
---

# Gradient Conflict（梯度冲突）

## 定义

在多任务学习中，当不同任务的损失梯度在共享参数上方向相反（余弦相似度为负）时发生梯度冲突，导致联合优化无法同时改善所有任务的性能。

## 数学形式

当任务 $i$ 和任务 $j$ 的梯度满足：

$$
\cos(\nabla_\theta \mathcal{L}_i, \nabla_\theta \mathcal{L}_j) < 0
$$

即发生梯度冲突，简单加权求和会损害至少一个任务的性能。

## 核心要点

1. 多任务学习的核心挑战之一，尤其在异质损失（L1、CE、MSE 混合）时更突出
2. 表现为训练过程中某任务性能先升后降
3. 解决方法：Gradient Surgery（投影冲突梯度）、PCGrad、MGDA、动态权重调整

## 代表工作

- [[GeoSem-WAM]]: 四路联合损失的潜在梯度冲突问题（作者将其列为局限性）

## 相关概念

- [[Multi-Task Learning]]
- [[Gradient Surgery]]
- [[PCGrad]]
