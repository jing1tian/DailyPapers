---
type: concept
aliases: [解码对齐, Decoding Alignment]
---

# Decoding Alignment（解码对齐）

## 定义

在 VLA（Vision-Language-Action）模型中，要求不同模态的输出使用各自适合的解码机制：语言/推理文本采用自回归因果解码，连续动作序列采用并行双向解码；强行使用单一解码器处理两种模态会同时损害性能和推理效率。

## 数学形式

$$
P(A, R \mid V, L) = \underbrace{P(A \mid V, L, R)}_{\text{双向并行解码}} \cdot \underbrace{P(R \mid V, L)}_{\text{自回归因果解码}}
$$

- 推理文本 $R$：因果 attention mask，逐 token 自回归生成
- 动作 $A$：双向 attention mask，并行解码（匹配 flow matching）

## 核心要点

1. **模态不对称性**: 自然语言的连贯性依赖因果注意力（前 token 决定后 token）；连续动作的 flow matching 天然适合双向建模
2. **性能代价**: 将 CoT 插入纯 AR 动作解码器（CoT+AR Action），LIBERO 性能下降 4.2pp（85.5% → 81.3%），推理延迟增加 4×
3. **混合解码器**: 先因果自回归生成 CoT，再切换 bidirectional attention 并行解码动作，恢复性能至 96.8% 且延迟接近基线

## 代表工作

- [[DeepThinkVLA]]: 提出 Decoding Alignment 概念并设计 [[Hybrid-Attention Decoder]] 解决该问题

## 相关概念

- [[Causal Alignment]]: 配套的第二个必要条件——通过 RL 建立 CoT 与任务成功的因果联系
- [[Hybrid-Attention Decoder]]: 实现解码对齐的具体架构方案
- [[Chain-of-Thought Reasoning]]: 引入 CoT 后需要解决解码对齐问题
- [[Autoregressive Generation]]: 自回归解码范式
