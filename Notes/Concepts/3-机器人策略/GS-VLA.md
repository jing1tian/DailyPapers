---
type: concept
aliases: [GS-VLA, Gaussian Splatting VLA Adapter]
---

# GS-VLA

## 定义
一种 plug-and-play 视角归一化框架，用 pixel-aligned 3D Gaussian Splatting 将任意相机视角的观测渲染为标准视角，直接喂给冻结的 VLA 策略，无需重训 VLA 参数。

## 核心要点
1. Pixel-aligned 3D-Gaussian Architecture：保留像素级对齐的 Gaussian 表示，确保细粒度空间信息
2. 只训练视角变换模块，VLA 权重完全冻结
3. 在 LIBERO 上测试了联合平移+旋转和外参噪声扰动鲁棒性

## 代表工作
- Park & Kim, 2026, Dankook University — [[GS-VLA]] (arXiv 2608.19066)

## 相关概念
- [[OpenVLA]]
- [[AnyCamVLA]]
- [[3D Gaussian Splatting]]
