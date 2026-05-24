---
type: concept
aliases: [Actor-Critic, AC算法, 演员-评论家]
---

# Actor-Critic

## 定义

结合策略梯度（Actor）与值函数估计（Critic）的强化学习框架：Actor 学习策略 $\pi_\theta(a|s)$，Critic 学习 Q 值/优势函数来指导 Actor 的更新。

## 数学形式

Actor 更新（策略梯度方向）：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot A^\pi(s,a) \right]
$$

Critic 更新（TD 误差）：

$$
\mathcal{L}_\text{critic} = \mathbb{E} \left[ \left( Q_\phi(s,a) - (r + \gamma Q_{\bar\phi}(s',a')) \right)^2 \right]
$$

## 核心要点

1. **Actor**：直接参数化策略，通过 Q 值梯度更新
2. **Critic**：估计状态-动作价值，提供低方差梯度信号
3. **集成 Critic**：使用多个 Q 网络取最小值（如 SAC），减少高估偏差
4. **离线-在线过渡**：Cal-QL 等方法解决 offline→online 的分布偏移问题

## 代表工作

- [[DyGRO-VLA]]: 使用集成 Critic（K=4）+ h 步 Bellman 回归的 Actor-Critic 框架进行在线 VLA-RL
- SAC (Soft Actor-Critic): 加熵正则化的 Actor-Critic
- TD3: 双 Critic 减少高估

## 相关概念

- [[强化学习]]
- [[Cal-QL]]
- [[PPO]]
