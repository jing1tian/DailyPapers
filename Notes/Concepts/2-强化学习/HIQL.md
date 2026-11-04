---
type: concept
aliases: [Hierarchical Implicit Q-Learning, Hi-IQL]
---

# HIQL

## 定义

层次化目标条件强化学习方法，在 IQL（Implicit Q-Learning）框架上引入层次化子目标策略，高层策略生成中间子目标，低层策略执行到达子目标的动作，无需 MPC 推理即可实现长时域导航。

## 核心要点

1. **基于 IQL**: 继承 IQL 的期望回归值估计（expectile regression），避免 OOD 动作的过估计
2. **层次化子目标**: 高层策略以目标为条件生成子目标，低层策略以子目标为条件执行动作
3. **值函数共享**: 高低层策略共享同一个值函数，简化训练
4. **步数效率高**: 学习型策略无需在线搜索，执行速度远快于 MPC 方法

## 代表工作

- [[IDWM]]: HIQL 作为长时域导航基线（Medium Maze），步数效率优于 ID-WM，但成功率不如 ID-WM（特别在操作任务上）

## 相关概念

- [[GCBC]]
- [[MPC]]
- [[层次化规划]]
- [[行为克隆]]
