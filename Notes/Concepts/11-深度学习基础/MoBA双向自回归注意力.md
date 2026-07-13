---
type: concept
aliases: [MoBA Attention Mask, Mixture of Bidirectional and Autoregressive Attention, 双向自回归混合注意力]
---

# MoBA 双向自回归注意力

## 定义

LingBot-World-Infinity 提出的混合注意力掩码机制，将双向注意力作为正则化项叠加在自回归 Teacher Forcing 掩码上，缓解因果视频生成中长序列[[teacher forcing|Teacher Forcing]]退化问题，同时保留自回归结构。

> **注意**: 此"MoBA"（Mixture of Bidirectional and Autoregressive）与 [[MoBA]]（Mixture of Block Attention，稀疏块注意力机制）是同名不同义的两个概念。

## 数学形式

**自注意力（Self-Attention）掩码**:

$$M_{self} = M_{tf} \oplus M_{bi}$$

其中 $M_{tf}$ 为标准下三角 Teacher Forcing 掩码，$M_{bi}$ 为额外追加的全帧双向块。

**交叉注意力（Cross-Attention）掩码**:

$$\text{Attention}(Q, K_c) = \underbrace{\text{Attn}_{tf}(Q,\, K_{bg},\, K_{chunk})}_{\text{下三角，防止未来泄露}} + \underbrace{\text{Attn}_{bi}(Q,\, K_{global})}_{\text{全局双向}}$$

## 核心要点

1. **Teacher Forcing 退化问题**: 纯因果模型在长序列中完全依赖历史上下文，预测质量随时序增加而退化
2. **双向正则化**: 追加双向块迫使模型学习不依赖特定上下文顺序的特征，提升泛化能力
3. **自注意力分离**: 带噪当前帧同时关注干净历史（因果）+ 全局上下文（双向）
4. **交叉注意力分离**: Chunk Prompt 使用下三角掩码避免未来信息泄露；Global Prompt 使用全局双向注意力

## 代表工作

- [[LingBot-World-Infinity]]: 提出此机制，用于无漂移长时域交互世界生成

## 相关概念

- [[MoBA]]: 同名不同义（Mixture of Block Attention，稀疏块选择）
- [[teacher forcing]]: 此机制要克服的退化问题
- [[Block Causal Attention]]: 相关的因果注意力变体
- [[Mixed Attention]]: 混合注意力机制的通用类别
