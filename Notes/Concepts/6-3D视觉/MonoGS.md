---
type: concept
aliases: [Monocular Gaussian Splatting SLAM]
---

# MonoGS

## 定义
基于 3D Gaussian Splatting 的单目视觉 SLAM 系统，同时优化相机位姿和 Gaussian 表示。

## 核心要点
1. 3DGS 代替传统点云/网格作为场景表示
2. 光度误差 + 几何约束联合优化
3. 支持 monocular RGB 输入（不需要深度传感器）

## 代表工作
- [[CubifyGS]]: 以 MonoGS 为参比方案

## 相关概念
- [[SplaTAM]]
- [[SLAM]]
