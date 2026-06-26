---
type: concept
aliases: [PA-RL, Policy Agnostic Reinforcement Learning]
---

# PA-RL

## 定义

PA-RL（Policy-Agnostic Reinforcement Learning）是一种离线-在线过渡强化学习方法，旨在与不同策略架构（包括 VLA）解耦，通过价值函数引导对任意预训练策略进行在线改进，无需修改策略本身的结构。

## 核心要点

1. **策略无关性**：价值函数/Critic 的训练与具体策略架构解耦，可适配不同 backbone
2. **离线预训练 + 在线微调**：与 [[Cal-QL]]、[[CQL]] 等方法类似的两阶段范式
3. **未显式处理分布偏移**：相比 [[FORCE]] 的分布式 Warm-up 机制，PA-RL 未显式解决离线到在线切换时的 Q 值分布错配问题

## 局限性

- 在 PullCubeTool、PlaceSphere 等高精度任务上成功率为 0，说明对困难探索任务的适应性不足
- 缺乏对探索样本质量的显式过滤机制

## 代表工作

- [[FORCE]]: 将 PA-RL 作为 baseline 对比，在 ManiSkill 任务上平均成功率落后 FORCE 30+ 个百分点

## 相关概念

- [[强化学习]]
- [[Cal-QL]]
- [[CQL]]
- [[Actor-Critic]]
