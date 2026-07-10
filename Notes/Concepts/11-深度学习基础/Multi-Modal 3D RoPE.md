---
type: concept
aliases: [3D MM-RoPE, Multi-Modal 3D Rotary Position Embedding, 多模态3D旋转位置编码]
---

# Multi-Modal 3D RoPE（多模态 3D 旋转位置编码）

## 定义
将多模态 token（条件文本/图像 + 视觉 patch）置于同一非重叠 3D 坐标系中的位置编码方案，在单流 Transformer 中同时保持视觉 token 的空间局部性、时序顺序，以及条件 token 与视觉 token 的坐标分离，无需任务专属架构。

## 核心要点
1. **条件 token**：使用仅时序坐标 $(i, 0, 0)$（$i = 1,\ldots,L$），不占用空间维度
2. **视觉 patch token**：使用 $(L+1+f, h, w)$，时序维度偏置避免与条件 token 重叠
3. **Query/Key 维度分拆**：注意力头维度沿时序、垂直、水平三轴分割，独立应用对应旋转频率
4. **完全单流兼容**：无需分离注意力路径，所有 token 在同一 Transformer 序列中处理，简化分布式并行

## 数学形式

设 $L$ 个条件 token，$F \times H \times W$ 视觉 latent 格：

$$
\text{条件 token}_i: \text{coord} = (i, 0, 0), \quad i = 1,\ldots,L
$$

$$
\text{视觉 token}_{f,h,w}: \text{coord} = (L+1+f, h, w)
$$

每个头维度按 $d_t : d_h : d_w$ 分割，分别应用时序/垂直/水平旋转频率。

## 代表工作
- Wan 系列视频模型
- Z-Image (Cai et al., arXiv 2511.22699): 引入该设计
- [[LingBot-Video]]: 在单流 DiT 中使用 3D MM-RoPE 统一条件与视觉 token 坐标

## 相关概念
- [[RoPE]]
- [[3D RoPE]]
- [[DiT]]
- [[MM-DiT]]
