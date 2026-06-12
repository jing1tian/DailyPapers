---
type: concept
aliases: [多任务学习, MTL, Joint Training, 联合训练]
---

# Multi-task Learning

## 定义

同时优化多个相关任务的损失函数，通过共享表征或梯度信号实现知识迁移与正则化。

## 数学形式

$$
\mathcal{L}_{\text{total}} = \sum_{i=1}^{T} \lambda_i \mathcal{L}_i
$$

其中 $\lambda_i$ 为各任务的损失权重，$\mathcal{L}_i$ 为第 $i$ 个任务的损失函数。

## 核心要点

1. **知识迁移**: 相关任务间的共享特征可相互增益，减少过拟合
2. **正则化效果**: 辅助任务充当隐式正则化，约束主任务的表征学习方向
3. **权重平衡**: 各任务权重 $\lambda_i$ 的选取对性能影响显著，需仔细调优

## 代表工作

- [[AGRA]]: 联合优化 WAM 视频重建损失和 AGRA 表征对齐损失 $\mathcal{L} = \mathcal{L}_{\text{WAM}} + \lambda_{\text{agra}} \cdot \mathcal{L}_{\text{AGRA}}$

## 相关概念

- [[Representation Alignment]]
- [[Cosine Similarity]]
