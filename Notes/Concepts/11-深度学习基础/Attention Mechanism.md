---
type: concept
aliases: [Attention, 注意力机制, Self-Attention, Scaled Dot-Product Attention]
---

# Attention Mechanism（注意力机制）

## 定义

注意力机制通过计算查询（Query）与键（Key）的相关性，加权聚合值（Value）的信息，使模型能够自适应地关注输入序列的不同部分。

## 数学形式

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

注意力输出（含残差）：

$$
H = U + A \cdot V \cdot W_O
$$

其中 $A = \text{softmax}(QK^\top / \sqrt{d_k})$ 为注意力权重。

## 核心要点

1. **Scaled Dot-Product**：$\sqrt{d_k}$ 缩放防止梯度消失
2. **Multi-Head**：并行多组注意力捕获不同子空间关系
3. **空间传播**：局部扰动（如对抗补丁）可通过注意力权重广播至全局位置
4. **VAE 瓶颈中的注意力**：视觉编码器中注意力机制是对抗补丁传播的核心机制

## 代表工作

- [[TAPDreamer]]: 通过注意力权重 $A$ 和值矩阵 $V$ 的联合变化（$\Delta A \Delta V$）实现局部补丁的全局传播

## 相关概念

- [[Transformer]]
- [[FlashAttention]]
- [[VAE]]
- [[Visual Encoder]]
