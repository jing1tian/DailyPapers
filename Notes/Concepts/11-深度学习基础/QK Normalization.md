---
type: concept
aliases: [QKNorm, QK Norm, Query-Key Normalization]
---

# QK Normalization

## 定义
QK Normalization 是一种在 Transformer 注意力机制中对 Query 和 Key 向量分别进行 L2 归一化的技术，用于稳定大模型训练时的注意力分布。

## 数学形式

标准缩放点积注意力：

$$
\text{Attn}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

引入 QK Normalization 后：

$$
\text{Attn}(Q,K,V) = \text{softmax}\!\left(\frac{\hat{Q}\hat{K}^\top}{\tau}\right)V
$$

其中 $\hat{Q} = Q / \|Q\|_2$，$\hat{K} = K / \|K\|_2$，$\tau$ 为可学习温度参数。

## 核心要点
1. 防止注意力 logit 随模型规模增大而爆炸，提升训练稳定性
2. 常与 RMSNorm 和 RoPE 配合使用（如 Stable Diffusion 3、WEAVER）
3. 取消了传统的 $1/\sqrt{d_k}$ 缩放，改用可学习温度

## 代表工作
- [[WEAVER]]: 32 层动力学 Transformer 使用 QK Normalization 稳定训练
- Stable Diffusion 3: 在 MMDiT 中引入 QKNorm

## 相关概念
- [[Transformer]]
- [[RMSNorm]]
- [[RoPE]]
