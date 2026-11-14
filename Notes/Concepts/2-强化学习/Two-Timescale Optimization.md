---
type: concept
aliases: [Two-Timescale Stochastic Approximation, 双时间尺度随机近似, 双时间尺度优化, 双速率优化]
---

# Two-Timescale Optimization（双时间尺度优化）

## 定义

双时间尺度优化是一种将参数分为"快参数"和"慢参数"两组并以不同学习率（或更新频率）分别更新的优化策略，理论上可保证在某些条件下两组参数均收敛到各自的最优解。

## 数学形式

设快参数 $\theta$（学习率 $\alpha_k$）与慢参数 $\omega$（学习率 $\beta_k$），满足：

$$
\alpha_k \to 0, \quad \beta_k \to 0, \quad \sum_k \alpha_k = \sum_k \beta_k = \infty, \quad \frac{\beta_k}{\alpha_k} \to 0
$$

则慢参数以足够慢的速率变化，从快参数的视角近似静止；快参数以足够快的速率变化，从慢参数的视角近似已收敛。

## 核心要点

1. **收敛保证**: 在标准随机近似条件下，两个参数组均收敛至各自对应的不动点
2. **Actor-Critic 对应**: 经典 Actor-Critic 中 Critic（快，频繁更新）和 Actor（慢，延迟更新）构成天然的双时间尺度结构
3. **模块解耦**: 两组参数可以使用完全不同的损失函数，只要两者的更新频率比 $\rho = \alpha/\beta$ 满足渐进条件即可

## 代表工作

- [[TEMPO]]: 将双时间尺度理论应用于 VLA RL 后训练，语义投影层（慢）与动作专家（快）以 $\rho=5:1$ 的频率比分别训练，抑制隐动作漂移
- [[TD3]]: Actor-Critic 中的延迟 actor 更新本质上是双时间尺度的一种实例化

## 相关概念

- [[TD3]]
- [[Actor-Critic]]
- [[Temporal Difference]]
- [[Latent-Action]]
