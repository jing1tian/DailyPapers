---
type: concept
aliases: [Root Mean Square Normalization, RMS归一化]
---

# RMSNorm

## 定义
RMSNorm 是 LayerNorm 的简化版本，仅通过均方根（RMS）对激活进行缩放，去除了均值中心化步骤，计算更高效且效果相当。

## 数学形式

$$
\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma, \quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}
$$

**符号说明**:
- $x \in \mathbb{R}^d$: 输入向量
- $\gamma$: 可学习缩放参数
- $\epsilon$: 数值稳定项

## 核心要点
1. 相比 LayerNorm 去除了均值减法，计算量减少约 7-8%
2. 在 LLaMA、Gemma、WEAVER 等现代大模型中广泛使用
3. 与 SwiGLU 配合，构成现代 Transformer 的标准归一化-前馈组合

## 代表工作
- [[WEAVER]]: 32 层动力学 Transformer 使用 RMSNorm
- LLaMA 系列: 将 RMSNorm 作为标准归一化层

## 相关概念
- [[Transformer]]
- [[SwiGLU]]
- [[QK Normalization]]
