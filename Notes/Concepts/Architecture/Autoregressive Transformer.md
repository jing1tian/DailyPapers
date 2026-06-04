---
type: concept
aliases: [AR Transformer, 自回归变换器, Autoregressive LM]
---

# Autoregressive Transformer

## 定义

基于 Transformer 架构的自回归语言/序列模型，通过 next-token prediction 逐步生成序列，每个位置的输出只能依赖当前位置之前的 token（因果掩码注意力）。

## 数学形式

$$
P(x_1, x_2, \ldots, x_N) = \prod_{i=1}^{N} P_\theta(x_i \mid x_1, \ldots, x_{i-1})
$$

训练目标（负对数似然）：

$$
\mathcal{L}_{AR} = -\sum_{i=1}^{N} \log P_\theta(x_i \mid x_{<i})
$$

## 核心要点

1. **因果掩码（Causal Mask）**: 注意力矩阵下三角为 1，上三角为 0，保证位置 $i$ 不能看到未来 token
2. **KV Cache**: 推理时缓存历史 key-value，避免重复计算，实现 O(N) 推理
3. **扩展性**: Scaling Law 显示参数量与性能呈幂律关系，是现代大语言模型的核心范式

## 与扩散模型的对比

| 属性 | 自回归 Transformer | 扩散模型 |
|------|-------------------|---------|
| 生成方式 | 逐 token 顺序生成 | 迭代去噪 |
| 适合模态 | 离散序列（文本、离散视觉 token）| 连续信号（图像、视频、音频）|
| 推理速度 | 可并行（KV Cache）| 需多步去噪 |

## 代表工作

- [[Cosmos3]]: MoT 架构中 AR 子序列负责物理场景推理，条件信号注入扩散生成子序列
- [[pi0]]: 用预训练 VLM（自回归）作为语言理解骨干，接扩散策略头输出动作
- [[OpenVLA]]: 完全自回归 VLA，将动作 token 化后用 next-token prediction 预测

## 相关概念

- [[Diffusion Transformer]]
- [[Mixture-of-Transformers]]
- [[Vision Transformer]]
