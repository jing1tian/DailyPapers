---
type: concept
aliases: [HY-Pano2, 混元全景生成, Panorama Generation Module]
---

# HY-Pano 2.0

## 定义

HY-World 2.0 的全景生成模块，基于 MMDiT 架构实现从文本提示或单视角透视图到高保真 360° 等矩形全景图的合成。

## 核心要点

1. 使用 [[Multi-Modal Diffusion Transformer|MMDiT]] 作为骨干网络
2. **隐式透视-等矩形映射**：不依赖显式相机元数据，从数据中学习空间对应关系
3. **环形填充（Circular Padding）**：在 Latent Space 中将全景图两端拼接，确保扩散过程的边缘连续性
4. **像素混合（Pixel Blending）**：在 Pixel Space 对拼接边界平滑处理，消除可见接缝
5. 训练数据：真实全景数据 + Unreal Engine 合成资产，严格过滤拼接伪影

## 代表工作

- [[HYWorld2]]: HY-World 2.0 四阶段流水线的第一阶段

## 相关概念

- [[Multi-Modal Diffusion Transformer]]
- [[World Model]]
- [[3D Gaussian Splatting]]
