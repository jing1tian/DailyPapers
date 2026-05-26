---
type: concept
aliases: [DexYCB Dataset]
---

# DexYCB

## 定义
用于手-物体交互的 RGB-D 视频数据集，包含 1000 个序列（10 名受试者 × 10 种 YCB 物体 × 多种抓取方式），提供 6DoF 物体姿态、手部姿态和 MANO 参数的精确标注。

## 核心要点
1. 物体来自 YCB 数据集（日常物品，如杯子、剪刀、工具）
2. 用多目 RGB-D 相机采集 + 光学追踪系统标注
3. 提供逐帧的物体 6DoF 姿态和 MANO 手部模型参数
4. 常用于评估手持物体姿态估计和追踪方法

## 代表工作
- [[ComPose]]: 在 DexYCB 上评估手-物交互中的物体姿态追踪

## 相关概念
- [[OakInk]]
- [[FoundationPose]]
