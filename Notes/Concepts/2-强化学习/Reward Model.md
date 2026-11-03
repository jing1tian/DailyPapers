---
type: concept
aliases: [Critic, 奖励模型, Value Function, 价值函数]
---

# Reward Model

## 定义

Reward Model（奖励模型）是一个辅助网络，用于在强化学习中估计状态（或状态-动作对）的价值，为策略优化提供密集的训练信号，而非仅依赖稀疏的环境奖励。

## 数学形式

$$
V_\phi(s_t) \approx \mathbb{E}_{\tau \sim \pi}\left[\sum_{k \ge 0} \gamma^k r_{t+k} \mid s_t\right]
$$

在 CaPtain 中为特权成功率 critic：

$$
V_\phi(s_t, t) = \sigma\!\left(\frac{1}{5}\sum_{k=1}^{5} f_{\phi_k}([x(s_t);\, t;\, e_{\mathrm{task}}])\right)
$$

## 核心要点

1. **密集信号**: 将稀疏的 episode 级奖励转化为每步的价值估计；
2. **特权 critic**: 训练时读取仿真器特权状态，测试时不使用（Asymmetric Actor-Critic）；
3. **集成**: 多个 MLP 集成可降低方差，提升 AUC；
4. **在线共训练**: critic 和策略同时更新，适应策略改变。

## 代表工作

- [[CaPtain]]: 用 36 维特权状态特征训练集成 MLP critic（AUC 0.970），精准分配物理信用
- PPO: critic 作为价值网络预测状态价值
- SWEET-RL, Le Critique: LLM 场景的特权 critic 方法

## 相关概念

- [[Reinforcement Learning]]
- [[Generalized Advantage Estimation]]
- [[Temporal Difference]]
- [[Proximal Policy Optimization]]
