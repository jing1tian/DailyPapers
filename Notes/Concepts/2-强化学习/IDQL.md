---
type: concept
aliases: [Implicit Diffusion Q-Learning, 隐式扩散Q学习]
---

# IDQL

## 定义
Implicit Diffusion Q-Learning，将扩散模型策略与隐式 Q-learning 结合的 offline RL 方法，通过保守估计避免值函数外推错误。

## 数学形式
$$Q(s, a) \leftarrow r + \gamma \mathbb{E}_{a' \sim \pi_\theta(\cdot|s')} [Q(s', a')]$$

策略通过扩散模型参数化：$a \sim p_\theta(a|s)$，Q 值作为 reweighting 权重。

## 核心要点
1. 扩散模型参数化策略，天然支持多模态动作分布
2. 隐式策略提取：不需要显式最大化 Q 值，避免 action optimization 困难
3. 与 [[FQL]] 相比，推理步数更多（扩散模型较慢）
4. 被 [[ForesightFlow]] 等工作用作基线对比

## 代表工作
- Hansen-Estruch et al. (2023): IDQL: Implicit Q-Learning as an Actor-Critic Method with Diffusion Policies

## 相关概念
- [[FQL]]
- [[AWR]]
- [[CFM]]
- [[Diffusion Policy]]
