---
type: concept
aliases: [MegaSaM, Mega-SAM]
---

# MegaSaM

## 定义

MegaSaM 是用于从视频序列中估计相机姿态（6-DoF pose）的工具，在 WBench 导航评分中被用于从生成视频里提取相机轨迹并与 ground-truth 对齐。

## 核心要点

1. **相机姿态估计**: 从单目视频序列重建每帧相机的位置和朝向
2. **导航评测核心**: 在 [[WBench]] 中支撑 I.1 Navigation Score 的计算（nATE）
3. **视角无关**: 支持第一人称和第三人称视频的姿态估计

## 代表工作

- [[WBench]]: 使用 MegaSaM 计算 Normalized Absolute Trajectory Error

## 相关概念

- [[归一化绝对轨迹误差]]
- [[WBench]]
