---
type: concept
aliases: [Soft Actor-Critic, SAC, 软演员评论家]
---

# SAC（Soft Actor-Critic）

## 定义

SAC 是一种基于最大熵框架的离策略深度强化学习算法，通过同时最大化累积奖励和策略熵，在连续动作空间中实现稳健高效的学习。

## 数学形式

$$
\mathcal{L}_\pi = \mathbb{E}_{a \sim \pi}\!\left[\alpha \log \pi(a|s) - Q(s,a)\right]
$$

其中 $\alpha$ 为自适应温度系数，$Q(s,a)$ 为软 Q 函数（由双 Critic 取最小估计）。

## 核心要点

1. **最大熵目标**: 策略在最大化奖励的同时最大化熵，鼓励探索并避免过早收敛
2. **双 Critic 网络**: 取两个 Q 网络的最小值估计以减少高估偏差（Clipped Double Q-learning）
3. **自适应温度**: $\alpha$ 自动调节熵目标约束，无需手动调参
4. **离策略优化**: 支持 replay buffer，样本效率高，适合真实机器人场景

## 代表工作

- [[HAF]]: 在频谱潜变量空间中用 SAC 进行离线到在线 RL 微调，冻结 VLA 主干
- [[DSRL]]: 同类机器人潜变量 RL 基线

## 相关概念

- [[BC]]（行为克隆）
- [[Flow-Matching]]
- [[Spectral Latent Space]]
