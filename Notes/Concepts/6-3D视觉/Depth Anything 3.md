---
type: concept
aliases: [Depth Anything 3, DepthAnything3, DepthAnythingV3]
---

# Depth Anything 3

## 定义

Depth Anything 3 是深度估计领域的先进基础模型，可对任意场景图像生成高精度单目深度图，在 WBench 等基准中被用于几何一致性评估（深度重投影位移）。

## 核心要点

1. **单目深度估计**: 从单张 RGB 图像预测稠密深度图
2. **强泛化性**: 在多种场景（室内、室外、合成）均有稳定性能
3. **评测工具**: 在 [[WBench]] 中作为 C.5 Geometric Consistency 的计算基础

## 代表工作

- [[WBench]]: 使用 Depth Anything 3 进行几何一致性评估

## 相关概念

- [[DreamSim]]
- [[WBench]]
