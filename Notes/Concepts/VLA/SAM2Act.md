---
type: concept
aliases: [SAM2Act]
---

# SAM2Act

## 定义
一种将 SAM2（Segment Anything Model 2）的视频目标追踪能力引入机器人操作策略的方法，通过追踪目标物体的时序状态来提供隐式记忆，增强对长视野任务的历史感知能力。

## 核心要点
1. 利用 SAM2 的内置记忆机制追踪操作目标
2. 提供基于视觉目标追踪的隐式历史状态表示
3. 与 HiMem-WAM 不同，记忆由目标追踪驱动而非技能边界触发
4. 适合目标明确的操作场景，对目标遮挡敏感

## 代表工作
- [[HiMem-WAM]]: 作为记忆感知操作方法的对比基线之一

## 相关概念
- [[MemoryVLA]]
- [[Memory Gating]]
- [[World Action Model]]
