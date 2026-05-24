---
type: concept
aliases: [跨任务迁移, Zero-Shot Transfer]
---

# Cross-Task Transfer（跨任务迁移）

## 定义

将从一个源任务学习到的表征、策略或引导算子直接应用于目标任务，无需在目标任务上重新收集数据或重新训练。

## 核心要点

1. COAST 发现：失败子空间在不同任务间高度共享，成功子空间保持任务特异性
2. 这种非对称性使跨任务 Conceptor 迁移成为可能——用源任务失败子空间引导目标任务
3. 失败子空间包容度（containment）预测迁移增益（$r = 0.30$-$0.49$，$p < 0.005$）
4. 部分跨任务迁移甚至超过自适应引导（如 Cheese+Butter 任务 Top-1 迁移 +0.80 vs Self +0.33）

## 代表工作

- [[COAST]]: 发现 VLA 失败几何的跨任务共享性，实现跨任务 Conceptor 迁移引导

## 相关概念

- [[Conceptor]]
- [[Subspace Similarity]]
- [[Activation Steering]]
