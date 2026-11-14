---
type: concept
aliases: [Twin Delayed DDPG, Twin Delayed Deep Deterministic Policy Gradient, 双延迟深度确定性策略梯度]
---

# TD3（Twin Delayed Deep Deterministic Policy Gradient）

## 定义

TD3 是在 [[DDPG]] 基础上引入三项关键改进（双 critic、延迟 actor 更新、目标策略平滑）的离线 Actor-Critic 强化学习算法，专为连续动作空间设计，有效缓解 Q 值过高估计问题。

## 数学形式

**双 critic 目标（取最小以减少过估计）**：

$$
y = r + \gamma \min_{i=1,2} \bar{Q}_i(s', \bar{\pi}(s') + \epsilon), \quad \epsilon \sim \mathrm{clip}(\mathcal{N}(0,\sigma), -c, c)
$$

**Critic 损失**：

$$
\mathcal{L}_{Q} = \sum_{i=1}^{2} \mathbb{E} \left[ (Q_i(s,a) - y)^2 \right]
$$

**Actor 损失（延迟更新，每 $d$ 步一次）**：

$$
\mathcal{L}_\pi = -\mathbb{E}_{s} \left[ Q_1(s, \pi(s)) \right]
$$

## 核心要点

1. **双 Critic（Twin Critics）**: 维护两个独立 Q 网络，Bellman 目标取两者最小值，解决 DDPG 的 Q 值高估问题
2. **延迟 Actor 更新（Delayed Updates）**: Actor 每 $d$ 步（通常 $d=2$）更新一次，而 Critic 每步更新，减少 Actor 更新时使用不稳定 Q 值的风险
3. **目标策略平滑（Target Policy Smoothing）**: 在目标 Actor 的动作上加裁剪高斯噪声，防止 Q 函数在某些动作上过拟合

## 代表工作

- [[TEMPO]]: 对 VLA 的语义投影层和动作专家各使用一套 TD3 循环，以不同频率进行双时间尺度 RL 后训练
- [[FastTD3]]: 对 TD3 的高效实现，通过向量化环境和编译优化加速训练

## 相关概念

- [[DDPG]]
- [[SAC]]
- [[Temporal Difference]]
- [[Actor-Critic]]
- [[Two-Timescale Optimization]]
