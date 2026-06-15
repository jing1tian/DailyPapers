---
type: concept
aliases: [Transformer解码器, Transformer Decoder, Cross-Attention Decoder]
---

# Transformer 解码器

## 定义
Transformer 解码器是 Transformer 架构中负责生成输出序列的模块，包含自注意力层和交叉注意力层，通过交叉注意力从编码器输出中提取相关信息。

## 核心要点
1. **自注意力**: 对输出序列本身建模因果依赖
2. **交叉注意力**: 以查询（来自解码器）和键/值（来自编码器）进行跨模态信息对齐
3. **空间解码器变体**: 以空间位置嵌入为查询，将1D特征序列映射为2D空间分布

## 数学形式

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中 $Q$ 为解码器查询，$K, V$ 来自编码器输出。

## 代表工作
- [[AffordanceVLA]]: Where2Act 使用 Transformer 解码器，以空间位置嵌入为交叉注意力查询生成 2D 可供性热图

## 相关概念
- [[视觉语言模型]]
- [[MoT架构]]
- [[可供性预测]]
