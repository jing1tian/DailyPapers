---
type: concept
aliases: [DuQuant, Dual Rotation Quantization]
---

# DuQuant

## 定义

基于旋转变换的 LLM 后训练量化方法，通过两阶段旋转（块旋转 + 三角变换）消除激活异常值，实现 W4A4 精度下的 LLM 量化。

## 核心要点

1. **双重旋转**: 先用块旋转（Block Rotation）将相邻通道的能量混合，再用三角旋转（Triangular Rotation）进一步消除残余异常值
2. **LLM 专用**: 设计针对 Transformer LLM 的激活分布，未考虑 DiT 动作头的跨步动态漂移特性
3. **局限**: 应用于 VLA 的 DiT 动作头时，W4A4 下 GR00T-N1.5 仅达 70.0%（vs Ω-QVLA 的 87.8%）

## 代表工作

- [[Omega-QVLA]]: 对比基线，超越 DuQuant 的方法

## 相关概念

- [[后训练量化（PTQ）]]
- [[Hadamard 矩阵]]
- [[SVD]]
