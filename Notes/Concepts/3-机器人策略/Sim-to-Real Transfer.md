---
type: concept
aliases: [Sim2Real, 仿真到现实迁移]
---

# Sim-to-Real Transfer

## 定义

将在仿真环境中学习的策略或知识迁移到真实机器人的技术，核心挑战是跨越视觉、物理、运动学等域差距（sim-to-real gap）。

## 核心要点

1. **域随机化（Domain Randomization）**：训练时随机化纹理/光照/物理参数以提升泛化
2. **系统辨识（System ID）**：精确建模真实机器人的物理参数
3. **语义迁移（Semantic Transfer, WICL）**：只迁移与视觉无关的语义知识，天然规避视觉 gap

## 代表工作

- [[RoboScript]]: 通过语义 Task Memory（WICL）实现 sim-to-real，仿真 ALOHA → 真实 PiPER 无需重排练

## 相关概念

- [[World In-Context Learning]]
- [[Domain Randomization]]
- [[Simulator]]
