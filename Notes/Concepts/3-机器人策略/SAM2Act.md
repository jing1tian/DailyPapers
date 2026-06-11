---
type: concept
aliases: [SAM2Act]
---

# SAM2Act

## 定义

基于 SAM2（Segment Anything Model 2）的记忆感知机器人操作方法，将 SAM2 的视频对象跟踪与记忆机制引入操作策略学习，以增强对长时域任务中目标状态的持续感知。

## 核心要点

1. 利用 SAM2 的实时视频分割与跟踪能力，为操作策略提供稳定的目标状态表示
2. 继承 SAM2 的记忆存储机制，在时序上保持对目标的长程跟踪
3. 与 [[MemoryVLA]] 等方法同为操作中引入显式记忆的代表性工作

## 代表工作

- SAM2Act: SAM2 驱动的记忆感知操作策略

## 相关概念

- [[MemoryVLA]]
- [[Memory Gating]]
- [[HiMem-WAM]]
