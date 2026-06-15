---
type: concept
aliases: [KV Cache, Key-Value Cache, KV Caching]
---

# KV 缓存

## 定义

KV 缓存（Key-Value Cache）是 [[Transformer]] 推理加速技术，通过缓存并复用注意力机制中已计算的 Key（K）和 Value（V）矩阵，避免在自回归生成或多步去噪过程中对历史 token 的重复计算，从而显著减少推理计算量。

## 数学形式

标准注意力计算：

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

使用 KV 缓存时，对于已处理的 token 集合 $\{1, \ldots, t-1\}$，其 $K_{1:t-1}$ 和 $V_{1:t-1}$ 被缓存复用，新查询 $Q_t$ 直接与缓存拼接计算，无需重新前向传播历史 token。

## 核心要点

1. **空间换时间**：以 GPU 内存存储历史 KV 为代价，将自回归推理从 $O(T^2)$ 降至 $O(T)$
2. **适用场景**：语言模型自回归生成、扩散模型的多步去噪（跨步骤缓存固定条件 token）
3. **扩散模型加速**：在扩散/流匹配推理中，条件 token（如记忆、历史帧）在每个去噪步中不变，KV 缓存可避免重复计算
4. **局限**：长序列时内存占用显著，需结合 sliding window 或分层压缩

## 代表工作

- [[WEAVER]]: 在多步流匹配推理中，对稀疏记忆和短期历史 token 跨去噪步缓存 KV，配合 [[SPRINT]] 实现最高 5-10× 速度提升
- GPT 系列模型：标准语言模型推理中的 KV 缓存

## 相关概念

- [[Transformer]]
- [[SPRINT]]
- [[流匹配]]
- [[Rectified Flow]]
