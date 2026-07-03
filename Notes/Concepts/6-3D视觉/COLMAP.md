---
type: concept
aliases: [COLMAP, Structure from Motion]
---

# COLMAP

## 定义

COLMAP 是一种广泛使用的 Structure-from-Motion（SfM）和 MVS（多视图立体）重建工具，从无序图像集合重建三维场景点云和相机位姿。

## 核心要点

1. 支持 SfM（相机位姿估计 + 稀疏点云）和 MVS（稠密点云）
2. 开源工具，在三维重建领域是事实上的标准工具
3. 在 [[NeRF]]、3DGS 等方法中广泛用于初始化相机位姿
4. [[PVWM-Path]] 用 COLMAP 做室外场景重建

## 相关概念

- [[PhysGaussian]]
- [[PVWM]]
