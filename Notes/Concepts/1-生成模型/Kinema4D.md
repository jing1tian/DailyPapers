---
type: concept
aliases: [Kinema4D World Model]
---

# Kinema4D

## 定义

一种使用密集几何渲染（mesh / 点图）作为条件的机器人动作条件视频世界模型，参数量达 14B。

## 核心要点

1. 使用机器人 mesh 或 4D 点图作为显式几何条件，包含丰富的外观信息
2. **局限性**: 过度依赖机器人特定外观（纹理、形状），在不同本体间泛化能力弱；参数量高达 14B，计算成本高
3. 需要高质量机器人特定 URDF 外观资产，难以扩展至人类手部等非机器人场景
4. 被 OSCAR 以 2B 参数在多项指标上超越

## 代表工作

- [[OSCAR]]: 对比基线，OSCAR 用更轻量的 2D 骨架条件在多数指标上超越 Kinema4D

## 相关概念

- [[Diffusion Transformer]]
- [[Forward Kinematics]]
