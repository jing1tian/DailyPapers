---
type: concept
aliases: [4D Gaussian Splatting, Dynamic 3D Gaussian Splatting]
---

# 4DGS

## 定义
[[3D Gaussian Splatting]] 向时间维度的扩展：为每个高斯核额外建模随时间变化的位置/形变（学习到的形变场或时空神经网络），从而用显式高斯表示渲染动态场景的新视角视频，兼顾 3DGS 的渲染速度和对非刚性运动的建模能力。

## 核心要点
1. 在 canonical-space 3DGS 基础上叠加形变场（deformation field），把每帧的高斯参数表示为 canonical 高斯 + 时变形变
2. 相比逐帧独立重建 3DGS，4DGS 通过共享 canonical 表示大幅减少参数量，并保证时序一致性
3. 训练信号通常来自多视角视频或长程 2D 轨迹（如光流/track）监督
4. 单目（monocular）场景下的 4DGS 是更难的子问题，因为缺乏多视角几何约束，容易在遮挡区域和大幅运动处退化

## 代表工作
- [[Lift4D]]: 用持续指导的形变场优化替代"只在初始化阶段用先验"的传统 4DGS 流程，解决单目 in-the-wild 4D 重建问题

## 相关概念
- [[3D Gaussian Splatting]]
- [[Shape of Motion]]
- [[CoTracker3]]
