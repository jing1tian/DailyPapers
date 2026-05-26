---
type: concept
aliases: [Simple VLA-RL, SimpleVLA RL]
---

# SimpleVLA-RL

## 定义

SimpleVLA-RL 是一种将强化学习引入 VLA 模型在线适应的方法，提供在线 RL 基线，用于评估更复杂的 VLA RL 框架。

## 核心要点

1. 基于 VLA 基础模型进行在线 RL 微调
2. 在 LIBERO-Long 上达到 91.2% 平均成功率
3. 作为 Agentic-VLA 的效率基准：需 2,450 次迭代（78.4k rollouts）才能达到 90% 成功率

## 代表工作

- [[Agentic-VLA]]: 用于对比训练效率，Agentic-VLA 比 SimpleVLA-RL 快 2.4×

## 相关概念

- [[GRPO]]
- [[EVOLVE-VLA]]
- [[VLA-RL]]
