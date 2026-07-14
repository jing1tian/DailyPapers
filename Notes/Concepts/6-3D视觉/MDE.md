---
type: concept
aliases: [Monocular Depth Estimation, 单目深度估计]
---

# MDE

## 定义
从单张 RGB 图像估计每个像素的深度值，是计算机视觉中经典的几何感知任务，也是机器人感知和 3DGS 重建的关键前处理步骤。

## 核心要点
1. 绝对深度估计 vs. 相对深度估计（尺度模糊）
2. 代表方法：Depth Anything、ZoeDepth、Marigold（diffusion-based）
3. 在机器人中用于无 LiDAR 场景的 3D 感知
4. SplatCtrl 用 MDE 补全 3DGS 的相机盲区

## 代表工作
- [[SplatCtrl]]：用 MDE 辅助 3DGS 机器人感知
- [[DepthSplat]]：深度估计辅助前馈 3DGS

## 相关概念
- [[3D Gaussian Splatting]]
- [[NeRF]]
