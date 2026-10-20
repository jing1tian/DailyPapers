---
type: concept
aliases: [多任务并行, 多任务推理并行, Heterogeneous Multi-Task Inference]
---

# Multi-Task Parallelism

## 定义

多任务并行（Multi-Task Parallelism）是指在同一推理系统中，从相同的输入观测出发，同时运行具有**异构输出格式**和**不同时间约束**的多个任务。在具身 AI 中，典型场景是操控动作生成（高频、严格实时）与语言对话/记忆构建（低频、跨帧延伸）的并行。

## 数学形式

**优化目标**（[[OxyGen]] 中的正式定义）：

$$\text{maximize} \quad \tau = \frac{Bk}{T} \quad \text{s.t.} \quad f = \frac{H}{T} \geq f_{\min}$$

- $\tau$：语言吞吐（tokens/s）
- $f$：动作频率（Hz），需满足最低要求 $f_{\min}$
- $k$：每帧解码步数（调节动作/语言资源分配的关键超参数）

## 核心要点

1. **异构截止期**：动作任务有严格 per-frame 截止（如 $T=14.3$ ms for 70 Hz），语言任务可跨帧完成
2. **资源竞争**：孤立执行时，多任务争夺同一 GPU 的计算核心和 KV Cache 内存
3. **共享观测**：多任务从同一视觉/语言观测出发，KV Cache 可跨任务共享
4. **Tradeoff 调节**：通过 $k$（每帧解码步数）在动作频率和语言吞吐之间做 Pareto 权衡

## 代表工作

- [[OxyGen]]: 首次正式建模 MoT VLA 下的多任务并行推理优化问题，提出统一 KV 管理

## 相关概念

- [[Mixture-of-Transformers]]
- [[KV Cache]]
- [[Continuous Batching]]
- [[Prefill]]
