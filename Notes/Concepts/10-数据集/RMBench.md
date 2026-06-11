---
type: concept
aliases: [RMBench, 记忆操作基准]
---

# RMBench

## 定义
一个专门评测机器人操作策略记忆能力的基准测试，包含 9 个需要跨时间步骤记忆状态的操作任务，分为单次记忆 M(1) 和多次试验 M(n) 两类。

## 核心要点
1. **M(1) 任务**: 需要单次观察并记住状态，包括 Observe and Pick Up、Rearrange Blocks 等 5 个任务
2. **M(n) 任务**: 需要多次试验并积累记忆，包括 Battery Try、Blocks Ranking Try 等 4 个任务
3. M(n) 任务比 M(1) 难度更高，要求更强的长期规划能力
4. 专门测试策略的记忆跟踪和历史状态推理能力

## 代表工作
- [[HiMem-WAM]]: M(1) 任务达 31.6%，M(n) 任务达 19.8%，总平均 26.3%

## 相关概念
- [[Memory Gating]]
- [[LIBERO]]
- [[World Action Model]]
