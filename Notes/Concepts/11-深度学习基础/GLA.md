---
type: concept
aliases: [Gated Linear Attention, 门控线性注意力]
---

# GLA (Gated Linear Attention)

## 定义
一种线性注意力变体，在每个 token 的状态更新中引入数据依赖的门控(gate)来控制历史信息的保留与遗忘，在保持线性时间/空间复杂度的同时获得比朴素线性注意力更强的长程记忆能力。

## 数学形式
$$S_t = \text{diag}(\alpha_t) S_{t-1} + k_t v_t^\top, \quad o_t = q_t^\top S_t$$

其中 $\alpha_t \in (0,1)^d$ 是数据依赖的门控向量，$S_t$ 是随时间线性递推的隐藏状态(等价于一个可遗忘的 KV 矩阵)。

## 核心要点
1. 用门控向量取代标准线性注意力中固定不变的状态累积，允许模型按 token 自适应地决定保留多少历史信息
2. 递推形式可并行化为分块(chunk-wise)矩阵运算，训练效率接近标准 Transformer 的线性注意力实现
3. 常被用作长序列/长时程任务里全局记忆模块的骨架，配合滑窗注意力做多尺度时序建模(局部用滑窗、全局用 GLA)

## 代表工作
- [[Kairos]]: 用 Hybrid Linear Temporal Attention 把滑窗局部注意力、扩张滑窗中程注意力和 GLA 全局记忆三层组合，限制长时程误差累积

## 相关概念
- [[FlashAttention]]
- [[GatedDeltaNet2]]
