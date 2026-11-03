---
type: concept
aliases: [Additive Quantization of Language Models]
---

# AQLM

## 定义
一种针对大语言模型的加性码书量化（Additive Quantization）方法，通过多个低位码书的加性组合来近似权重矩阵，实现极低比特（2-3 bit）量化。

## 数学形式
$$W \approx \sum_{k=1}^{M} C_k[I_k]$$

其中 $C_k$ 是第 $k$ 个码书，$I_k$ 是索引向量。

## 核心要点
1. 多码书加性组合比单码书更准确地逼近原始权重分布
2. 专为 LLM/VLA 的 post-training quantization (PTQ) 设计
3. 比 GPTQ 在极低比特（2-3 bit）下保持更好精度

## 代表工作
- [[SACQ]]: 在 VLA 量化中与 AQLM 对比，SACQ 针对 VLA 激活函数特性做了改进

## 相关概念
- [[GPTQ]]
