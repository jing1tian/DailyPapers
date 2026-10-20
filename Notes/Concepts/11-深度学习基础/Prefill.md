---
type: concept
aliases: [预填充, Prompt Prefill, Context Encoding, Prefill Phase]
---

# Prefill

## 定义

Prefill（预填充）是 Transformer 推理的第一阶段：将输入 prompt（或观测序列）并行编码，计算并缓存所有输入 token 对应的 Key-Value 对（KV Cache），为后续的自回归解码做准备。

## 数学形式

$$\{(h_{t,l}, K_{t,l}, V_{t,l})\}_{l=1}^{L} = \Theta_{VLM}(o_t)$$

$$\mathcal{K}_t = \{(K_{t,l}, V_{t,l})\}_{l=1}^{L}$$

其中 $o_t$ 为输入观测/prompt，$L$ 为 Transformer 层数。Prefill 是**计算密集型**（compute-bound）操作，输入越长开销越大。

## 核心要点

1. **并行计算**：Prefill 阶段对所有输入 token 做一次并行 attention 计算，与解码的逐 token 串行不同
2. **KV 持久化**：Prefill 产生的 KV Cache 保存在 GPU 内存中，供后续解码复用
3. **重复开销问题**：VLA 多任务推理中，若动作专家和语言专家各做一次 Prefill，会重复计算完全相同的 KV，造成浪费
4. **跨任务共享**：[[OxyGen]] 通过一次 Prefill + KV Cache 共享，消除多任务推理中的重复 Prefill

## 代表工作

- [[OxyGen]]: 将一次 Prefill 的 KV Cache 跨任务共享，核心优化之一

## 相关概念

- [[KV Cache]]
- [[Autoregressive Generation]]
- [[Continuous Batching]]
- [[Mixture-of-Transformers]]
