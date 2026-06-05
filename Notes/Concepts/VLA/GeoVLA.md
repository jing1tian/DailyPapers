---
type: concept
aliases: [Geometry-aware VLA, 几何感知 VLA]
---

# GeoVLA

## 定义

一种通过显式几何特征注入增强 VLA 模型 3D 空间感知能力的方法，在 LIBERO 基准上取得 97.7% 平均成功率，是 3DThinkVLA 的主要对比基线之一。

## 核心要点

1. 显式注入 3D 几何表示（如点云或深度特征）到 VLM 骨干或动作头
2. 聚焦低层次几何感知，但缺乏高层次空间推理迁移机制
3. LIBERO 性能（97.7%）被 3DThinkVLA（98.7%）超越

## 代表工作

- [[3DThinkVLA]]: 作为对比基线，3DThinkVLA 的几何适配器设计参考并超越了 GeoVLA

## 相关概念

- [[VLA]]
- [[VGGT]]
- [[SpatialVLA]]
