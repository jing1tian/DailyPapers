---
type: concept
aliases: [Twin Bridge, 数字孪生驱动RL]
---

# TwinBridge

## 定义
数字孪生驱动的机器人在线强化学习框架：用 3DGS 重建数字孪生提供 policy 初始化和 [[PBRS]] 奖励信号，配合少量真实 RGB 做视觉迁移，解决实验室机器人 RL 中演示成本高、奖励难设计的问题。

## 核心要点
1. 数字孪生（3DGS）提供 policy 初始化，减少真实环境探索需求
2. 用有序事件序列设计 PBRS 奖励，[[SplatSim]] 做视觉迁移
3. 对比 [[SERL]], [[ConRFT]], [[ReWiND]] 等在线 RL 方案
4. 场景：实验室精细操作（生化实验室机器人）

## 相关概念
- [[PBRS]]
- [[SplatSim]]
- [[SERL]]
- [[ConRFT]]
