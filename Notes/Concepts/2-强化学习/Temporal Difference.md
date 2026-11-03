---
type: concept
aliases: [TD学习, TD Error, TD误差, Temporal Difference Learning]
---

# Temporal Difference

## 定义

Temporal Difference（TD）学习是强化学习中一类不需要等到 episode 结束就能更新价值估计的方法，通过"自举（bootstrapping）"用当前估计值更新自身。

## 数学形式

**1-step TD 误差**：

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

**CaPtain 中的逐请求 TD 信用（时间索引固定）**：

$$
\delta_t = c_t + V_\phi(s_{t+1}, t) - V_\phi(s_t, t)
$$

固定时间索引 $t$（而非 $t+1$）消除了"任务自然进展"带来的虚假信用。

## 核心要点

1. TD 是 Monte Carlo（无偏但高方差）和动态规划（低方差但需模型）之间的折中；
2. 1-step TD 偏差最大，n-step TD 偏差随 $n$ 增大而减小；
3. [[Generalized Advantage Estimation|GAE]] 是对多步 TD 误差的加权平均；
4. 在多轮代码智能体中，固定时间索引的 TD 差分可避免感知类请求因任务进展获得虚假信用。

## 代表工作

- [[CaPtain]]: 用固定时间索引的 TD 差分为 22 类原语精准分配物理信用

## 相关概念

- [[Reinforcement Learning]]
- [[Generalized Advantage Estimation]]
- [[Reward Model]]
