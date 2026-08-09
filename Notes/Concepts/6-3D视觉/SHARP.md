---
type: concept
aliases: [SHARP, pixel-aligned Gaussian splatting]
---

# SHARP

## 定义
一种单图前馈 3D Gaussian Splatting 方法，从固定像素网格位置直接预测 Gaussian 属性，是 pixel-aligned 表示范式的代表工作。

## 核心要点
1. 从输入图像的像素网格位置预测每个 Gaussian 的属性（位置、颜色、不透明度等）
2. 前馈推理无需每场景优化，速度快
3. 在近视角渲染质量好，但大视角变化下 Gaussian 与场景几何面的对齐较弱，出现散乱 primitive

## 代表工作
- [[InfiniSplat]]: 提出 surface-aligned 替代方案，批判 pixel-aligned 在大基线 NVS 下的局限

## 相关概念
- [[PixelSplat]]
- [[TokenGS]]
- [[InfiniSplat]]
- [[3D Gaussian Splatting]]
