---
type: concept
aliases: [GigaBrain-WBC, GigaBrain-WBC-0.5, Behavior World Model for WBC]
---

# GigaBrain-WBC

## 定义
在 SONIC 全身运动跟踪框架基础上引入 Behavior World Model（BWM）做 OOD 指令实时过滤，同时加入 Automatic Spatial Terrain Annotation 处理地形与物体交互，提升人形机器人全身控制在复杂场景下的鲁棒性。

## 核心要点
1. Behavior World Model：在每个控制步前预测动作可行性，充当 OOD gate
2. Automatic Spatial Terrain Annotation：自动标注地形类别用于物理交互
3. 基于 MotionMillion 数据集训练，覆盖多地形和物体交互场景

## 代表工作
- Cheng et al., 2026 — (arXiv 2608.18234)

## 相关概念
- [[SONIC]]
- [[MuJoCo]]
- [[MotionMillion]]
- [[HoloMotion]]
