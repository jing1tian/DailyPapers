---
type: concept
aliases: [Swish-Gated Linear Unit, SwiGLU FFN]
---

# SwiGLU

## 定义
SwiGLU 是一种门控前馈网络激活函数，将 Swish 激活与门控线性单元（GLU）结合，在保持参数量的前提下显著提升 Transformer 前馈层的表达能力。

## 数学形式

$$
\text{SwiGLU}(x, W, V, b, c) = \text{Swish}(xW + b) \odot (xV + c)
$$

其中 $\text{Swish}(x) = x \cdot \sigma(x)$，$\sigma$ 为 sigmoid 函数。

完整 FFN 层：

$$
\text{FFN}_{SwiGLU}(x) = \text{SwiGLU}(x, W_1, W_3) \cdot W_2
$$

## 核心要点
1. 相比 ReLU/GELU，门控机制允许网络动态控制信息流
2. 需要三个权重矩阵（$W_1, W_2, W_3$），为保持参数量通常将隐层维度乘以 $2/3$
3. 已成为现代大语言模型（LLaMA、PaLM 等）和视觉 Transformer 的标准前馈层

## 代表工作
- [[WEAVER]]: 32 层动力学 Transformer 使用 SwiGLU 前馈网络
- LLaMA 系列: 以 SwiGLU 替换标准 FFN

## 相关概念
- [[Transformer]]
- [[RMSNorm]]
