---
type: concept
aliases: [TransFuser]
---

# TransFuser

## 定义
基于 Transformer 的自动驾驶感知-预测融合模型，融合 RGB 图像和 LiDAR BEV 特征，用于 CARLA/NAVSIM 上的 end-to-end 驾驶任务。

## 核心要点
1. 多模态融合：图像 Transformer + LiDAR Transformer 的跨模态注意力
2. 无需显式 waypoint 规划，直接从融合特征预测控制
3. 2021 CVPR 工作，Auto-JEPA 等后续工作的基础 baseline

## 代表工作
- [[Auto-JEPA]]: 在 NAVSIM 上与 TransFuser 对比

## 相关概念
- [[NAVSIM]]
- [[CARLA]]
- [[DiffusionDrive]]
