---
type: concept
aliases: [Depth Splat]
---

# DepthSplat

## 定义
基于单目/立体深度估计的前馈 3D Gaussian Splatting 方法，无需逐场景优化，直接从多视图图像预测 Gaussian 参数。

## 核心要点
1. 前馈推理，无需 per-scene 优化
2. 利用深度估计作为 Gaussian 初始化
3. 在 RealEstate10K、KITTI-360 等数据集评测
4. 相比 PixelSplat 使用了更强的深度先验

## 代表工作
- [[StereoSplat+]]：在 DepthSplat 基础上引入 diffusion 补全

## 相关概念
- [[3D Gaussian Splatting]]
- [[PixelSplat]]
- [[MDE]]
