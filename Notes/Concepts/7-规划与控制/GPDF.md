---
type: concept
aliases: [Generalized Positive Definite Functions, Gaussian Process Distance Field]
---

# GPDF

## 定义
在 SplatCtrl 中指 Gaussian Process Distance Field，利用 3D Gaussian 场景表示估计机器人到障碍物的距离场，用于反应式运动规划中的碰撞约束。

## 核心要点
1. 从 3DGS 场景表示中直接提取距离场信息
2. 不需要显式的 SDF 网格，利用 Gaussian 密度场近似距离
3. 结合二次规划（QP）做实时约束运动规划
4. 支持动态障碍物（Gaussian 位置实时更新）

## 代表工作
- [[SplatCtrl]]：GPDF + QP 实现 3DGS 场景下的反应式机器人控制

## 相关概念
- [[3D Gaussian Splatting]]
- [[SDF]]
