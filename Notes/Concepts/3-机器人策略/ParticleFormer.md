---
type: concept
aliases: [ParticleFormer]
---

# ParticleFormer

## 定义

ParticleFormer 是一种基于图神经网络（GNN）的粒子动力学模型，用于从点云数据学习物体的物理动力学，适用于机器人操纵中的 model-based planning。

## 核心要点

1. 将点云表示为粒子图，用 GNN 建模粒子间相互作用
2. 支持刚体和可变形物体的动力学预测
3. 用于机器人操纵中的 model-based planning（MPPI 等）
4. 是 [[PointWorld]] 的对比基线

## 相关概念

- [[PointWorld]]
- [[MPPI]]
- [[ManiSkill]]
