---
type: concept
aliases: [Constrained MDP, 约束马尔可夫决策过程, Constrained Markov Decision Process]
---

# CMDP

## 定义
Constrained Markov Decision Process：在标准 MDP 基础上引入约束代价（cost）函数，要求策略在最大化累积奖励的同时，将累积代价控制在阈值之内，是安全强化学习（Safe RL）的标准形式化框架。

## 数学形式
$$\max_\pi J_r(\pi) = \mathbb{E}\left[\sum_t \gamma^t r(s_t, a_t)\right]$$
$$\text{s.t.} \quad J_c(\pi) = \mathbb{E}\left[\sum_t \gamma^t c(s_t, a_t)\right] \leq d$$

其中 $r$ 是奖励函数，$c$ 是代价函数，$d$ 是代价预算。

## 核心要点
1. 标准 MDP 添加代价函数 $c(s,a) \geq 0$ 和约束阈值 $d$
2. 拉格朗日松弛法是最常见解法：将约束转化为惩罚项 $\lambda \cdot J_c$
3. 离线设置下的 CMDP 需要额外处理分布偏移问题（如配合 [[BCQ]] 等方法）
4. 实际应用中代价标注通常稀疏（轨迹级而非逐步），需要 credit assignment

## 代表工作
- Altman, "Constrained Markov Decision Processes", 1999（奠基性著作）
- CPO（Constrained Policy Optimization）
- [[RCI]]（稀疏代价下的 CMDP offline RL）

## 相关概念
- [[BCQ]]: 离线 RL 基础算法
- [[CBF]]: 控制层面的安全约束方法（与 CMDP 互补）
