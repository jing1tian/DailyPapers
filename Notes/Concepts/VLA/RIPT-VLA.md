---
type: concept
aliases: [RIPT-VLA, Reinforced In-Process Training VLA]
---

# RIPT-VLA

## 定义

一种面向 [[VLA]] 模型的强化学习微调（RFT）方法，在 LIBERO-Long 任务上实现 93.8% 成功率，是 DyGRO-VLA 论文中重要的单任务 RL 基线之一。

## 核心要点

1. 属于 VLA 强化后训练（RFT）方法流派
2. 针对单任务场景优化，在 LIBERO-Long 上达到 93.8%
3. 不解决多任务灾难性遗忘问题，跨任务泛化性有限
4. 被 DyGRO-VLA（95.2%）在 LIBERO-Long 单任务设置下超越

## 代表工作

- [[DyGRO-VLA]]: 在多任务 RL 框架下超越 RIPT-VLA 的单任务性能

## 相关概念

- [[VLA]]
- [[强化学习]]
- [[LIBERO]]
