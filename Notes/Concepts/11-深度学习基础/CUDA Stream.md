---
type: concept
aliases: [CUDA流, CUDA并行流, GPU Stream, CUDA Event]
---

# CUDA Stream

## 定义

CUDA Stream 是 NVIDIA CUDA 编程模型中的异步执行队列，同一 Stream 内的操作顺序执行，不同 Stream 之间可并行执行，通过 CUDA Event 实现精确的跨流依赖同步。

## 数学形式

无严格数学形式，主要是执行调度语义：

- **Stream 内**: 操作 A → 操作 B（顺序，A 完成才执行 B）
- **Stream 间**: Stream_1 的操作与 Stream_2 的操作并行（若无 Event 依赖）
- **Event 同步**: `Stream_1.record(e)` → `Stream_2.wait(e)` 确保 Stream_2 在 Event e 之后的操作等待 Stream_1 的记录点完成

## 核心要点

1. **默认 Stream**：CUDA 操作默认在 Stream 0（默认流）顺序执行，无并行性。
2. **多 Stream 并行**：将不相关的 GPU 操作分配到不同 Stream，可同时占用不同 SM 或 MEM BW，提升 GPU 利用率。
3. **CUDA Event**：轻量级同步原语，仅记录时间点，比完整的 Stream 同步（`cudaStreamSynchronize`）粒度更细。
4. **应用场景**：计算-通信重叠（AllReduce + 前向计算）、多头注意力并行、跨模块流水线（[[CEWP]]）。

## 代表工作

- [[RealtimeWAM]]：通过双 CUDA Stream + Event 实现 [[CEWP]] 跨专家波前流水线

## 相关概念

- [[CEWP]]
- [[CUDA Graph]]
- [[Mixture-of-Transformers]]
