---
type: concept
aliases: [Dichotomous Diffusion Policy Optimization]
---

# DIPOLE

## 定义
Dichotomous Diffusion Policy Optimization：将 diffusion denoising 过程拆为高噪声阶段（value function 引导）和低噪声阶段（行为克隆稳定）两段，以解决 offline RL 训练 diffusion policy 不稳定的问题。

## 核心要点
1. 高噪声阶段：Q-value 梯度引导生成方向探索
2. 低噪声阶段：行为克隆约束保持策略稳定性
3. 解决直接最大化 value function 导致的 denoising 崩溃

## 代表工作
- [[Dichotomous Diffusion Policy Optimization]]: 原始论文（arXiv 2601.00898）

## 相关概念
- [[DiffusionForcing]]
- [[FQL]]
- [[IQL]]
- [[ExORL]]
