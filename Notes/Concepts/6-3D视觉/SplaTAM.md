---
type: concept
aliases: [SplaTAM, Splat Track and Map]
---

# SplaTAM

## 定义
基于 3D Gaussian Splatting 的 SLAM 系统，在 3DGS 表示上同时做相机跟踪和地图构建。

## 核心要点
1. 直接优化 Gaussian 参数做 dense SLAM
2. 无需 mesh 或点云中间表示
3. 实时渲染用于视觉反馈和位姿估计

## 代表工作
- [[CubifyGS]]: SplaTAM 为参比的 passive 重建方案

## 相关概念
- [[MonoGS]]
- [[SLAM]]
