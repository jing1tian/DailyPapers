---
type: concept
aliases: [Perceiver IO, PerceiverIO, 跨注意力序列编码器]
---

# Perceiver-IO

## 定义

一种基于 cross-attention 的通用编码/解码架构，可将任意模态、任意长度的输入序列映射到固定尺寸的隐向量集合，再通过解码器重建任意输出序列，天然适合不定长时间序列的压缩表示学习。

## 数学形式

**编码器（输入 → 隐向量）**:

$$
Z = \text{CrossAttn}(Q_{\text{latent}},\ K=V=X_{\text{input}}) \in \mathbb{R}^{K \times d}
$$

随后经 $L$ 层自注意力细化：

$$
Z \leftarrow \text{SelfAttn}^{(L)}(Z)
$$

**解码器（隐向量 → 输出）**:

$$
Y = \text{CrossAttn}(Q_{\text{query}},\ K=V=Z) \in \mathbb{R}^{N_{\text{out}} \times d}
$$

其中 $Q_{\text{latent}}$ 是可学习的 $K$ 个查询向量，$X_{\text{input}}$ 是任意长度的输入序列。

## 核心要点

1. **解耦输入/输出长度**: 隐向量数 $K$ 固定，与输入长度 $T$ 无关，计算复杂度从 $O(T^2)$ 降至 $O(T \cdot K + K^2)$
2. **通用接口**: 支持点云、图像、音频、时间序列等多模态输入，无需专用架构
3. **对称编解码**: 同一结构用于编码（压缩）和解码（重建），适合 VAE 训练
4. **在 FM-VLA 中的配置**: 编码器 2 层 cross-attn + 10 层 self-attn，隐藏维度 384，输出 $K=8$ 个 96 维向量

## 代表工作

- Jaegle et al. (2021), "Perceiver IO: A General Architecture for Structured Inputs & Outputs"
- [[FM-VLA]]: 用 Perceiver-IO 编/解码 6 轴 wrench 历史序列，训练 Force-VAE

## 相关概念

- [[VAE]]
- [[Force Memory Token]]
- [[Masked ELBO]]
- [[Transformer]]
