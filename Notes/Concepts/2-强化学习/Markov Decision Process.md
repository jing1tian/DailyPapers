---
type: concept
aliases: [MDP, 马尔可夫决策过程]
---

# Markov Decision Process

## 定义
强化学习的标准数学形式化，将序贯决策问题描述为五元组 $\mathcal{M}=(\mathcal{S},\mathcal{A},P,r,\rho_0,\gamma)$，智能体在状态 $\mathcal{S}$ 下根据策略选择动作 $\mathcal{A}$，环境按转移概率 $P$ 演化并给出奖励 $r$，目标是最大化折扣累计回报。

## 数学形式
$$
\mathcal{M}=(\mathcal{S},\mathcal{A},P,r,\rho_0,\gamma), \qquad J(\pi_\theta)=\mathbb{E}_{\pi_\theta}\Big[\sum_t \gamma^t r_t\Big]
$$

## 核心要点
1. $\mathcal{S}$ 为状态空间，$\mathcal{A}$ 为动作空间，$P(s'\mid s,a)$ 为状态转移核，$\rho_0$ 为初始状态分布，$\gamma\in[0,1)$ 为折扣因子
2. 马尔可夫性假设：下一状态只依赖当前状态和动作，与历史无关
3. 在含安全约束的场景中可扩展为 [[Constrained MDP]]，额外引入安全代价信号 $c$ 并要求满足预算约束

## 代表工作
- [[SafeDojo]]: 将标准 VLA 的 MDP 扩展为带安全代价的 Constrained MDP，作为安全约束 RL 优化的形式化基础

## 相关概念
- [[Constrained MDP]]
- [[GRPO]]
- [[强化学习]]
