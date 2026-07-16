---
type: concept
aliases: [FlagScale框架, 智源分布式训练框架]
---

# FlagScale

## 定义

由北京智源人工智能研究院（BAAI）开发的大规模分布式训练框架，集成 FSDP2、激活重计算、通信调度等优化，为百亿参数以上模型提供高吞吐量训练支持。

## 核心要点

1. 基于 [[FSDP2]] 实现灵活的参数分片，提升内存利用率
2. 通过 Chunked Cross-Entropy Loss 降低大词表的内存峰值
3. 前向/后向传播预取机制使 FSDP all-gather 通信与计算重叠
4. 在 Orca 训练中实现 4.4× 吞吐量加速（2.91 vs 0.66 Samples/Sec/GPU）

## 代表工作

- [[Orca]]: 使用 FlagScale 框架在 32 节点 / 256 GPU 上训练

## 相关概念

- [[FSDP2]]
