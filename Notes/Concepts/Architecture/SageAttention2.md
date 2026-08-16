---
type: concept
aliases: [SageAttention2, Sage Attention]
---

# SageAttention2

## 定义
一种高效注意力计算实现，通过量化和优化 CUDA 内核显著降低注意力计算的显存占用和延迟，专为大型生成模型（尤其是视频 DiT）的实时推理设计。

## 核心要点
1. 在 SageAttention 基础上进一步优化，支持更低比特量化（如 INT8/INT4 KV 缓存）
2. 相比标准 Flash Attention，在视频生成场景下可将推理速度提升 2-4×
3. 主要用于视频世界模型、视频 DiT 的部署加速（如 ABot-World-0 的实时推理栈）

## 代表工作
- [[ABot-World-0]]: 在全栈推理 co-design 中使用 SageAttention2 实现 RTX 5090 单卡 16FPS 世界模型

## 相关概念
- [[DiT]]
- [[Flash Attention]]
