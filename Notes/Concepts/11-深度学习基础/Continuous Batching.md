---
type: concept
aliases: [连续批处理, 动态批处理, Iteration-Level Batching]
---

# Continuous Batching

## 定义

连续批处理（Continuous Batching）是一种 LLM 推理调度策略：在每个解码迭代（iteration）结束后立即将已完成的请求移出、将等待的新请求加入当前 batch，而非等整个 batch 的所有请求都完成后才切换，从而最大化 GPU 利用率。

## 数学形式

设当前 batch 为 $\hat{\sigma} = (\hat{\mathcal{K}}, \hat{y}, \hat{\delta})$，每迭代执行 $k$ 步解码后：

$$\hat{\sigma}' = \{(\mathcal{K}_i, y_i', \delta_i') \mid i \in \mathcal{B}_{active}\}$$

其中 $\mathcal{B}_{active}$ 在每次迭代后动态更新：已终止（$\delta_i=1$）的请求移除，新到达的请求加入。

## 核心要点

1. **细粒度调度**：以单次 forward pass（iteration）为调度粒度，而非以整个 batch 完成为粒度
2. **消除等待气泡**：避免短请求因等待同批次长请求而空转
3. **与 PagedAttention 配合**：配合内存分页（PagedAttention）可支持动态内存分配，进一步提升吞吐
4. **跨帧扩展**：在 [[OxyGen]] 中扩展为"跨帧连续批处理"，将来自不同控制帧的语言请求合并批处理

## 代表工作

- [[OxyGen]]: 将 Continuous Batching 扩展至 VLA 多任务推理，提出跨帧连续批处理
- vLLM: 最早将 Continuous Batching + PagedAttention 结合用于 LLM serving

## 相关概念

- [[KV Cache]]
- [[Autoregressive Generation]]
- [[Prefill]]
