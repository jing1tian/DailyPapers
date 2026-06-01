---
type: concept
aliases: [Flash Attention, 闪速注意力]
---

# FlashAttention

## 定义
一种 IO 感知的精确注意力算法，通过分块计算（tiling）和重计算（recomputation）避免将完整的 N×N attention 矩阵写入 HBM，将 attention 的内存复杂度从 O(N²) 降至 O(N)，同时保持数值等价性。

## 数学形式

标准 attention：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

FlashAttention 用 tile 分块计算 softmax 的 online normalization：
$$m_i = \max_j(q_i k_j^T / \sqrt{d}), \quad \ell_i = \sum_j e^{q_i k_j^T/\sqrt{d} - m_i}$$

## 核心要点
1. 将 Q、K、V 分块加载到 SRAM，避免反复从 HBM 读写大矩阵
2. 使用 online softmax 算法，单 pass 完成数值稳定的 softmax + attention
3. 反向传播时重新计算 attention，不存储中间矩阵，节省显存
4. FlashAttention-2 优化了并行策略；FlashAttention-3 针对 Hopper GPU 的异步特性优化

## 代表工作
- [[FlashAttention]]: Dao et al., 2022 — 原始版本
- [[FlashAttention3]]: Hopper GPU 专项优化版本

## 相关概念
- [[DiT]]
- [[RoPE]]
- [[FlexAttention]]
