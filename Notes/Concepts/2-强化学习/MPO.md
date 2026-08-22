---
type: concept
aliases: [Maximum A Posteriori Policy Optimisation, MPO算法]
---

# MPO（Maximum A Posteriori Policy Optimisation）

## 定义

MPO 是一种基于期望最大化（EM）框架的 off-policy 强化学习算法，通过在策略更新中引入 KL 散度约束，实现在不偏离当前策略过远的前提下稳定提升回报。

## 数学形式

$$
\pi^{k+1} = \arg\max_\pi \mathbb{E}_{s \sim d^k}\!\left[\mathbb{E}_{a \sim q(\cdot|s)}\!\left[\frac{\pi(a|s)}{q(a|s)} Q^k(s, a)\right]\right]
$$

subject to $\mathbb{E}_s[\text{KL}(q(\cdot|s) \| \pi(\cdot|s))] \leq \varepsilon$

## 核心要点

1. **两步 EM 迭代**: E 步求解非参数化中间分布 $q$（通过软最大化 Q 值），M 步将策略 $\pi$ 拟合到 $q$
2. **Off-policy 兼容**: 利用重要性采样从回放缓冲区采样，支持高效的 off-policy 训练
3. **KL 约束稳定性**: 通过约束 $q$ 与 $\pi$ 之间的 KL 散度，避免策略突变带来的不稳定
4. **连续动作空间友好**: 对高维连续动作空间（如机器人关节控制）表现稳定

## 代表工作

- [[EXIMO]]: 用 MPO 训练残差策略做 VLA 微调的在线 RL 阶段

## 相关概念

- [[Residual Policy]]
- [[强化学习]]
