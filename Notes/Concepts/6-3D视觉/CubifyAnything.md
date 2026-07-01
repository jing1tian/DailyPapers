---
type: concept
aliases: [CubifyAnything]
---

# CubifyAnything

## 定义
物体级 3D 包围盒估计方法，从 RGB 图像预测场景中物体的 3D bounding box（旋转对齐的长方体）。

## 核心要点
1. 端到端预测 6-DOF bounding box
2. 与 SAM 等分割模型结合可做 object-centric 场景分解
3. 用于 3DGS 的 object-level 初始化

## 代表工作
- [[CubifyGS]]: 用 CubifyAnything 做 object 分割和包围盒初始化

## 相关概念
- [[SAM]]
- [[FoundationPose]]
