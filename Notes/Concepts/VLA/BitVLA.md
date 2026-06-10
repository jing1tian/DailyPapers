---
type: concept
aliases: [BitVLA, Bit VLA]
---

# BitVLA

## 定义

BitVLA 是一种使用三元权重（$\{-1, 0, 1\}$）量化的视觉-语言-动作模型，基于 BitNet-SigLIP 骨干网络，以极低内存占用（~1.3 GiB）实现 VLA 推理。

## 核心要点

1. **三元量化**: 权重仅取 $\{-1, 0, 1\}$ 三个值，大幅降低内存占用
2. **架构**: BitSigLIP-L 视觉编码器 + BitNet-2B 语言模型 + MLP 回归动作头（2.4B 参数）
3. **无迭代去噪**: 动作头为 MLP 回归，无需流匹配或扩散的迭代步骤（$T = \text{None}$）
4. **内存效率**: 在 [[vla.cpp]] 中仅需 1312 MiB VRAM，LIBERO-Object 达到 100% 成功率

## 代表工作

- [[vla.cpp]]: 为 BitVLA 定制 W2A8 IMMA GEMM kernel，实现 4.5× 推理加速

## 相关概念

- [[vla.cpp]]: 支持 BitVLA 推理的运行时
- [[VLA]]: 视觉-语言-动作模型范式
- [[SigLIP]]: 视觉编码器基础架构
