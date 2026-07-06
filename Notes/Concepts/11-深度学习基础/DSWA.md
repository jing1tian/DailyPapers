---
type: concept
aliases: [Dynamic Sliding Window Attention, DSWA]
---

# DSWA

## 定义
Dynamic Sliding Window Attention：Kairos 提出的混合线性注意力机制，动态调整滑动窗口大小以在长序列建模中平衡局部精度与全局效率。

## 数学形式
$$\text{DSWA}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}} \cdot M_w\right)V$$

其中 $M_w$ 是动态窗口掩码，窗口大小 $w$ 根据序列位置或内容自适应调整。

## 核心要点
1. 结合 GLA（门控线性注意力）使用，形成 Hybrid Linear Attention 架构
2. 长时序列下显著降低注意力计算复杂度（从 $O(n^2)$ 降低）
3. 在 Kairos 的多尺度时序记忆（Multi-Scale Temporal Memory）中提供局部精细感知
4. 相比固定窗口 SWA，动态调整更适应视频/动作序列的非均匀信息密度

## 代表工���
- [[Kairos]]: 在 World Model 架构中引入 DSWA + GLA 混合注意力

## 相关概念
- [[Sliding Window Attention]]: DSWA 的前身（固定窗口版本）
- [[GLA]]: Kairos 中与 DSWA 配合的线性注意力
- [[DiT]]: Kairos 整体架构基础
