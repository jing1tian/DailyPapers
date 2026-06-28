---
type: concept
aliases: [SoM]
---

# Shape of Motion

## 定义
单目动态场景重建方法：联合优化一个 canonical-space 的 [[3D Gaussian Splatting]] 表示和逐帧形变场，用长程 2D 轨迹（long-range track）监督形变，解决纯靠相邻帧光流容易漂移、长时序运动难以保持一致性的问题。是"先建 3D 再变形"这条单目 4D 重建路线的代表性工作。

## 核心要点
1. 用预训练的长程点追踪器（如 [[CoTracker3]]）提供跨多帧的稠密轨迹监督，而非只用相邻帧约束
2. canonical 高斯 + 逐帧形变的分解方式让模型可以在大幅非刚性运动下仍保持身份一致性
3. 局限：先验（轨迹监督）主要在训练阶段起作用，初始化之后的优化更多依赖视频证据本身，对遮挡和远超训练分布的运动鲁棒性有限

## 代表工作
- [[Lift4D]]: 将 Shape of Motion 列为对比 baseline，指出其"先验只用一次就退场"的问题，并提出用重建结果持续指导形变场优化的改进方案

## 相关概念
- [[4DGS]]
- [[3D Gaussian Splatting]]
- [[CoTracker3]]
