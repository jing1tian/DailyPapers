---
type: concept
aliases: [SufficientPlan]
---

# SufficientPlan

## 定义
一种减少 World Action Model 规划搜索成本的方法，通过"充分性"终止条件提前终止规划搜索，避免不必要的 rollout。

## 核心要点
1. 核心思想：只要当前候选方案已达到"充分好"的阈值，就停止进一步搜索
2. 与 PILOT 等 policy-initialized 方法可以组合使用
3. 是 pico-WAM 中用于降低规划成本的基线方法

## 代表工作
- [[pico-WAM]]: 以 SufficientPlan 作为规划搜索的对比方法

## 相关概念
- [[PILOT]]
- [[DINO-WM]]
- [[MCTS]]
