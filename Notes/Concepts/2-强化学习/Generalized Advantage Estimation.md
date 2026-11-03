---
type: concept
aliases: [GAE, 广义优势估计]
---

# Generalized Advantage Estimation

## 定义

Generalized Advantage Estimation（GAE）是一种用于策略梯度算法的优势函数估计方法，通过对多步 TD 误差加权求和，在偏差（bias）和方差（variance）之间取得平衡。

## 数学形式

$$
A^{\mathrm{GAE}}_t = \sum_{k \ge 0} (\gamma\lambda)^k \delta_{t+k}
$$

其中 TD 误差为：

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

- $\lambda \to 0$: 退化为 1-step TD（低方差，高偏差）
- $\lambda \to 1$: 退化为 Monte Carlo（无偏差，高方差）

## 核心要点

1. $\lambda$ 是偏差-方差权衡的关键超参，常用 $\lambda=0.95$ 或 $0.9$；
2. 配合价值网络（critic）使用，$V(s)$ 越准确 GAE 越有效；
3. 支持多轮/多步场景，能将后续奖励信用向前传播；
4. CaPtain 中 $\lambda=0.9$，无折扣（$\gamma=1$），用 TD 差分跨请求传播物理信用。

## 代表工作

- [[CaPtain]]: GAE λ=0.9 配合物理 critic TD 差分，将运动信用传播到早期感知请求
- PPO (Schulman et al. 2017): GAE 是 PPO 的标准组件
- 原始 GAE 论文: Schulman et al. 2015

## 相关概念

- [[Reinforcement Learning]]
- [[Proximal Policy Optimization]]
- [[Reward Model]]
- [[Temporal Difference]]
