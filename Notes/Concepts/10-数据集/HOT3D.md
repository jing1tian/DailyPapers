---
type: concept
aliases: [HOT3D, Hand and Object Tracking 3D]
---

# HOT3D

## 定义

HOT3D（Hand and Object Tracking in 3D）是一个自我中心视角的手部与物体三维交互数据集，包含精确的 3D 手部和物体轨迹标注，用于评估手部姿态估计和物体追踪方法。

## 核心要点

1. **自我中心视角**: 采集自穿戴式相机，提供第一人称视角的手部操作视频
2. **精确 3D 标注**: 包含 [[MANO]] 参数化手部标注和物体 6-DOF 位姿
3. **多样场景**: 覆盖多种日常物体抓取和操作任务
4. **标准 benchmark**: 用于 [[MPJPE]] 等标准指标的评估

## 代表工作

- [[P3WAM]]: 使用 HOT3D 评估 Perceive（手部重建）模式，P3-WAM 相机空间 MPJPE 远优于 [[HaMeR]] 和 [[Dyn-HaMR]] baseline

## 相关概念

- [[EgoDex]]
- [[ARCTIC]]
- [[EgoLive]]
- [[MANO]]
- [[MPJPE]]
