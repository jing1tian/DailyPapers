---
type: concept
aliases: [REINFORCE Leave-One-Out, Leave-One-Out baseline]
---

# RLOO

## 定义
REINFORCE Leave-One-Out 是 policy gradient 的一种方差缩减变体，对每个样本使用同组其他样本的均值奖励作为 baseline，不需要额外训练 critic 网络。

## 数学形式
$$\nabla J(\theta) = \frac{1}{N} \sum_{i=1}^{N} \left( r_i - \frac{1}{N-1}\sum_{j \neq i} r_j \right) \nabla \log \pi_\theta(a_i|s_i)$$

## 核心要点
1. 不需要 critic 网络，计算成本低
2. 基线估计来自同 batch 的其他样本
3. 在 N 较大时方差缩减效果好
4. 常与 GRPO 对比，GRPO 是其在语言模型 RL 的变体

## 代表工作
- [[Prompt-Driven Exploration]]：与 GRPO 对比作为 RL 优化器

## 相关概念
- [[GRPO]]
- [[PPO]]
