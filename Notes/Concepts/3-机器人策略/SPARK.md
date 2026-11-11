---
type: concept
aliases: [Spatial Representation Distillation for VLA]
---

# SPARK

## 定义
通过知识蒸馏将 3D 空间先验注入 2D RGB VLA 的方法，用 routing 机制支持多任务场景下的空间表示选择。

## 核心要点
1. 3D 空间 teacher（depth/点云）蒸馏到 2D RGB VLA backbone
2. Knowledge routing：不同任务选择不同空间表示子空间
3. 推理时不需要额外深度传感器（蒸馏已内化空间先验）
4. 在 [[LIBERO]] 上验证，有真实实验

## 代表工作
- [[SPARK]]: 原始论文（Queen Mary Univ/Yonsei Univ，2611.10443）

## 相关概念
- [[VLA]]
- [[LIBERO]]
- [[SpatialVLA]]
