---
type: concept
aliases: [Gated DeltaNet-2, Gated Delta Rule 2]
---

# GatedDeltaNet2

## 定义
线性注意力架构，将记忆的"擦除"（erase）和"写入"（write）操作显式解耦，解决 Delta Rule 中两者耦合导致的记忆干扰问题。

## 数学形式

Gated Delta Rule-2 的状态更新：
$$\bm{S}_t = \bm{S}_{t-1} \odot (1 - \bm{g}_t^e \bm{k}_t^\top) + \bm{g}_t^w \bm{v}_t \bm{k}_t^\top$$

其中 $\bm{g}_t^e$（erase gate）和 $\bm{g}_t^w$（write gate）独立控制，不再共享。

## 核心要点
1. 原始 Delta Rule 把 erase 和 write 耦合在同一个权重中，更新新值时会干扰已有关联
2. GatedDeltaNet-2 解耦两个 gate，使记忆编辑更精确
3. 保留 chunkwise 并行计算结构（兼容 [[KDA]] 框架），推理效率与原始线性注意力相当
4. 在 RULER、NIAH 等长序列检索任务上超越 KDA 和原始 DeltaNet

## 代表工作
- [[GatedDeltaNet2]]（Hatamizadeh et al., 2026, NVIDIA）: 提出方法并在多个 NLP benchmark 上验证

## 相关概念
- [[KDA]]: Kimi Delta Attention，GatedDeltaNet-2 改进的基础
- [[状态空间模型]]: SSM 是另一类线性复杂度序列模型
- [[Mamba]]: 基于 SSM 的高效 sequence model
