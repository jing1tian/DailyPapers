---
type: concept
aliases: [PPO, Proximal Policy Optimization, 近端策略优化]
---

# PPO（Proximal Policy Optimization）

## 定义

PPO 是一种基于策略梯度的强化学习算法，通过裁剪概率比来约束每次更新步长，在保证训练稳定性的同时实现样本高效利用，是当前机器人控制和 RLHF 训练中最常用的算法之一。

## 数学形式

**裁剪目标函数**:

$$
\mathcal{L}^\text{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta) \hat{A}_t,\; \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_t\right)\right]
$$

**概率比**:

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)}
$$

**符号说明**:
- $\pi_\theta$: 当前策略；$\pi_{\theta_\text{old}}$: 旧策略（数据收集时）
- $\hat{A}_t$: 优势估计值（Generalized Advantage Estimation, GAE）
- $\varepsilon$: 裁剪阈值（通常 0.1 或 0.2）

## 核心要点

1. **裁剪约束**: 限制策略更新幅度，避免过大步长导致训练崩溃（比 TRPO 更简洁）
2. **On-policy**: 需要用当前策略收集数据，但可以多次复用同一批数据（多个 epoch）
3. **Actor-Critic 结构**: 通常配合价值函数（Critic）估计优势，联合训练
4. **适合稀疏奖励**: 结合 GAE 可有效处理长视距任务中的奖励延迟问题

## 代表工作

- [[SpikeVLA]]: 使用 PPO + [[代理梯度|代理梯度（Surrogate Gradient）STBP]] 训练 Spike-A 脉冲动作策略，实现四足运动控制

## 相关概念

- [[代理梯度|Surrogate Gradient]]
- [[脉冲神经网络|SNN]]
- [[Advantage Estimation]]
