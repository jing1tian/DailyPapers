---
type: concept
aliases: [Large View Synthesis Model]
---

# LVSM

## 定义
LVSM：大规模视角合成模型，直接从稀疏输入视图生成新视角图像，无需显式 3D 表示（如 NeRF 或 3DGS）。

## 核心要点
1. Transformer-based 直接回归新视角像素
2. 无需 test-time 优化，单次前向推理
3. 与 [[InfiniSplat]] 代表不同路线：implicit vs explicit 3D

## 代表工作
- [[InfiniSplat]]: 与 LVSM 对比，在大基线新视角合成上的隐式 3DGS 方案

## 相关概念
- [[InfiniSplat]]
- [[3DGS]]
