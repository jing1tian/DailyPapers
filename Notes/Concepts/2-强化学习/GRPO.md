---
type: concept
aliases: [Group Relative Policy Optimization, 组相对策略优化]
---

# GRPO

## 定义

GRPO（Group Relative Policy Optimization）是一种无需价值网络的强化学习策略优化算法，通过对同一 prompt 生成多个候选输出并进行组内相对奖励归一化，来估计优势函数，从而降低训练复杂度。

## 数学形式

对同一输入生成 $G$ 个候选输出 $\{o_1, \ldots, o_G\}$，组内归一化优势：

$$
\hat{A}_i = \frac{R_i - \mu_R}{\sigma_R + \varepsilon}
$$

策略梯度目标（带 clip 的 PPO 风格）：

$$
\mathcal{L}_{GRPO} = \mathbb{E}\left[\min\left(\frac{\pi_\theta}{\pi_{ref}}\hat{A}_i, \text{clip}\left(\frac{\pi_\theta}{\pi_{ref}}, 1-\varepsilon, 1+\varepsilon\right)\hat{A}_i\right)\right] - \beta D_{KL}(\pi_\theta \| \pi_{ref})
$$

## 核心要点

1. **无价值网络**: 用组内相对奖励代替价值函数估计优势，节省训练资源
2. **组内归一化**: 同一 prompt 的 $G$ 个 rollout 共享归一化，减少奖励尺度偏差
3. **KL 惩罚**: 对比参考策略 $\pi_{ref}$ 的 KL 散度限制策略偏离
4. **扩展性**: 已被扩展至 Flow Matching（Flow-GRPO）用于视频生成模型对齐

## 代表工作

- [[GRPO]] 原始提出：DeepSeek-R1
- [[Shared-Prefix GRPO]]: 共享前缀的 GRPO 变体，减少生成冗余
- [[EVA]]: 将 GRPO 扩展至 Flow Matching 视频世界模型的 RL 对齐

## 相关概念

- [[PPO]]
- [[RLHF]]
- [[Flow Matching]]
- [[可执行性奖励]]
