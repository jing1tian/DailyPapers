---
type: concept
aliases: [Multi-Buffer Joint Fine-tuning, 多缓冲联合微调]
---

# MBFT

## 定义
Multi-Buffer Joint Fine-tuning：FlashVLA 提出的训练技术，同时训练多个 action buffer 的 action head，减少异步部署时因推理延迟导致的 action staleness 问题。

## 核心要点
1. 并行训练多个时间偏移的 action buffer，每个 buffer 对应不同的执行延迟
2. 解决 flow-matching VLA 的 TTFA（Time-to-First-Action）瓶颈
3. 与 Chunk-wise Autoregressive 解码结合实现流式 action 输出
4. Fine-tuning 代价需要在每个新任务上评估

## 代表工作
- [[FlashVLA]]: MBFT 原始提出

## 相关概念
- [[Action Chunking]]
- [[SmolVLA]]
- [[Flow Matching]]
