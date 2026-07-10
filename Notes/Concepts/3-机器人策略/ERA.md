---
type: concept
aliases: [Event-Retrieve-Action, Event-centric Retrieval Architecture, 事件中心检索架构]
---

# ERA

## 定义
一种以"事件"为基本单元的具身决策架构，通过检索历史相关事件作为上下文，提升长程决策的一致性和效率。

## 核心要点
1. 将连续观察流压缩为离散"事件"序列，每个事件封装一段语义完整的交互片段
2. 决策时用向量检索（VPF, Vector-based Past Framework）找到过去相关事件
3. 检索到的事件作为 memory 上下文条件化当前决策策略

## 代表工作
- [[ERA]]: Event-Centric World Modeling with Memory-Augmented Retrieval for Embodied Decision-Making (2604.07392)

## 相关概念
- [[WAM]]
- [[MemoryVLA]]
