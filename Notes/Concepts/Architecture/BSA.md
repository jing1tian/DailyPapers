---
type: concept
aliases: [Block Sparse Attention, 块稀疏注意力]
---

# BSA (Block Sparse Attention)

## 定义
一种稀疏注意力机制，将注意力计算限制在预定义的块结构内，降低高分辨率/长序列下的计算复杂度。

## 数学形式
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V \quad \text{仅在稀疏块内计算}$$

## 核心要点
1. 将序列分块，仅在块内或跨预定义块对间计算注意力，避免全局 $O(n^2)$ 复杂度
2. 对视频生成尤其有效：时间维度和空间维度均可做块稀疏化
3. LongCat-Video 用于在高分辨率（720p）下加速推理

## 代表工作
- [[LongCat-Video]]: 在 DiT 视频生成中使用 BSA 加速高分辨率推理

## 相关概念
- [[DiT]]
- [[Cross-Attention]]
