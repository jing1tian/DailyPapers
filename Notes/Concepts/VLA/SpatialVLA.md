---
type: concept
aliases: [SpatialVLA, Spatial VLA]
---

# SpatialVLA

## 定义

一类融合空间感知能力的 [[VLA]] 模型，在语言和视觉之外显式建模 3D 空间关系，提升机器人在复杂空间推理任务中的操作能力。

## 核心要点

1. 在标准 VLA 架构之上引入空间表征模块
2. 在 LIBERO 基准上平均成功率约 78.1%（Spatial: 88.2%, Object: 89.9%, Goal: 78.6%, Long: 55.5%）
3. 对空间感知类任务（LIBERO-Spatial）表现较好，但长时序任务仍有明显差距
4. 在 DyGRO-VLA 论文中作为重要对比基线

## 代表工作

- [[DyGRO-VLA]]: 超越 SpatialVLA（97.1% vs 78.1%），尤其在 Long 任务大幅领先

## 相关概念

- [[VLA]]
- [[LIBERO]]
- [[Vision-Language-Action Model]]
