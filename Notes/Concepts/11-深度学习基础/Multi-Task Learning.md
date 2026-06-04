---
type: concept
aliases: [多任务学习, MTL]
---

# Multi-Task Learning（多任务学习）

## 定义

多任务学习通过在共享表示上同时优化多个相关任务的损失，利用任务间的互补信息提升每个任务的泛化性能。

## 数学形式

$$
\mathcal{L}_{\text{MTL}} = \sum_{i=1}^{T} \lambda_i \cdot \mathcal{L}_i
$$

其中 $\lambda_i$ 为第 $i$ 个任务的权重，$\mathcal{L}_i$ 为对应任务损失。

## 核心要点

1. 共享表示：多个任务共享底层特征提取器，任务特定头各自独立
2. 负迁移（Negative Transfer）：任务间目标冲突可能导致某些任务性能下降
3. 梯度冲突：不同任务梯度方向相反时，简单加权求和可能损害性能
4. 解决方案：Gradient Surgery、PCGrad、MGDA 等梯度协调方法

## 代表工作

- [[GeoSem-WAM]]: 四路联合损失（RGB + 几何 + 语义 + 动作）的多任务学习

## 相关概念

- [[Gradient Conflict]]
- [[Gradient Surgery]]
- [[PCGrad]]
