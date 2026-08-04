---
type: concept
aliases: [Key-Value Cache, KV缓存, 键值缓存]
---

# KV Cache（键值缓存）

## 定义

Transformer 推理加速技术：将注意力机制中已计算过的 Key（K）和 Value（V）矩阵缓存到内存中，在自回归生成时直接复用，避免对历史 token 的重复计算，将每步推理复杂度从 $O(n^2)$ 降至 $O(n)$。

## 数学形式

标准自注意力：

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

KV Cache 在自回归第 $t$ 步时，将 $K_{1:t-1}$、$V_{1:t-1}$ 缓存，仅计算新 token $t$ 的 $K_t$、$V_t$ 并拼接：

$$
K_{1:t} = [K_{1:t-1};\, K_t], \quad V_{1:t} = [V_{1:t-1};\, V_t]
$$

## 核心要点

1. **适用场景**: 仅对因果（自回归）Transformer 有效，前向编码器不受益
2. **内存换时间**: 缓存占用显存随序列长度线性增长，换取推理速度的大幅提升
3. **跨步复用**: [[Fibonacci Recurrent Inference]] 利用斐波那契对齐使 VLA 历史帧的 KV 缓存可跨推理步骤精确复用
4. **量化兼容**: KV Cache 可与 INT8/FP8 量化结合进一步减少显存占用（如 MLA 等变体）

## 代表工作

- [[FibVLA]]: 利用斐波那契递归推理跨步复用历史帧 KV Cache，降低 VLA 推理延迟 27%
- [[MLA]]: Multi-head Latent Attention，通过低秩分解压缩 KV Cache 显存

## 相关概念

- [[Fibonacci Recurrent Inference]]: 跨推理步骤精确复用 KV Cache 的策略
- [[Action Chunking]]: VLA 推理时 KV Cache 复用的边界由动作块长度决定
