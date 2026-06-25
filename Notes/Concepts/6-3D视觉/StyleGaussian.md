---
type: concept
aliases: []
---

# StyleGaussian

## 定义
基于 3D Gaussian Splatting 的快速 3D 场景风格迁移方法，将参考风格图像的外观特征迁移到已重建的 3DGS 场景上，同时保持原场景的几何结构和多视角一致性。

## 核心要点
1. 风格迁移只动颜色/外观特征，几何（高斯位置、形状）基本保持不变
2. 相比基于 NeRF 的风格迁移（如 [[StylizedNeRF]]），利用 3DGS 的显式表征实现更快的风格化速度
3. 常被后续"几何感知风格迁移"工作点名为"只做外观、不动几何结构"的局限性代表

## 相关概念
- [[3D Gaussian Splatting]]
- [[StylizedNeRF]]
