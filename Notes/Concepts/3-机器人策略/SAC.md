---
type: concept
aliases: [Soft Actor-Critic, 软演员评论家]
---

# SAC (Soft Actor-Critic)

## 定义
基于最大熵框架的 off-policy 强化学习算法，通过最大化累积奖励和策略熵的加权和，在连续动作空间实现稳定高效的学习。

## 数学形式
目标函数：
$$J(\pi) = \sum_t \mathbb{E}_{(s_t,a_t)\sim\rho_\pi}\left[r(s_t,a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))\right]$$
其中 $\mathcal{H}$ 为策略熵，$\alpha$ 为温度参数（可自适应调整）。

## 核心要点
1. Off-policy：使用经验回放，数据效率高
2. 最大熵框架：鼓励探索，防止早熟收敛
3. 自动调整温度 $\alpha$，无需手动调参
4. 双 Q 网络（取较小值）缓解 Q 值高估
5. 广泛用于机器人操作的 online RL fine-tuning

## 代表工作
- [[RankQ]]：使用 SAC 采样动作来构造 ranking 对，规避 OOD Q 值高估

## 相关概念
- [[CQL]]
- [[DP]]
