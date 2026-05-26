---
type: concept
aliases: [Inverse Reinforcement Learning, 逆强化学习]
---

# IRL

## 定义
Inverse Reinforcement Learning，从专家演示中反推奖励函数的方法，假设专家行为是在某个未知奖励函数下的最优策略，目标是找到能解释演示行为的奖励函数。

## 数学形式
MaxEntIRL 目标（最大熵 IRL）：
$$\max_r \sum_{\tau \in \mathcal{D}} r(\tau) - \log Z(r)$$
$$Z(r) = \sum_\tau \exp(r(\tau))$$

## 核心要点
1. 与 BC（行为克隆）不同，IRL 不直接模仿动作，而是学习奖励
2. 学到的奖励可以迁移到新环境（泛化性更好）
3. 演示不完美时（低估某些特征），IRL 容易产生 reward misalignment
4. GAIL 用 GAN 框架隐式优化 IRL，避免显式 rollout 计算配分函数

## 代表工作
- [[ASQ]]: 用主动探索修复 IRL 的 reward misalignment 问题
- [[Reinforcement Learning]]: IRL 是 RL 的逆问题

## 相关概念
- [[奖励函数]]
- [[模仿学习]]
- [[行为克隆]]
- [[强化学习]]
