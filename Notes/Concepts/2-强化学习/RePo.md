---
type: concept
aliases: [RePo, Representation-based Policy Optimization]
---

# RePo

## 定义
基于 bisimulation metric 的 RL 表示学习方法，通过最小化任务无关特征来让策略学到对 visual distractor 鲁棒的紧凑表示。

## 核心要点
1. 用 latent bisimulation distance 度量两个状态在任务上的等价性：$d(z_1, z_2) = |r_1 - r_2| + \gamma \cdot \mathbb{E}[d(z_1', z_2')]$
2. 优化目标使等价状态在 latent 空间中靠近，不等价状态远离
3. 相比 TIA 的显式分解，RePo 通过度量学习隐式过滤 distractor

## 代表工作
- [[RePo]] (Yang et al., 2023): 在 DMC 和 CARLA 干扰环境中超越 TIA

## 相关概念
- [[TIA]]
- [[DreamerV3]]
- [[Dueling-WM]]
