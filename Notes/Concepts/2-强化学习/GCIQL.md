---
type: concept
aliases: [Goal-Conditioned IQL, Goal-Conditioned Implicit Q-Learning]
---

# GCIQL

## 定义
Goal-Conditioned Implicit Q-Learning：将 IQL 的隐式 Q 值学习扩展到 goal-conditioned 设置，避免 Q-value 外推错误。

## 数学形式
$$V(s,g) = \mathbb{E}_\tau\left[\sum_t \gamma^t r(s_t, g)\right]$$

通过 expectile regression 而非最大化 Q 值来学习价值函数，避免分布外动作的 Q 过估计。

## 核心要点
1. 继承 IQL 的 in-sample learning 优势，不需要 actor 在分布外采样
2. Goal 条件通过 goal-state concatenation 或 goal embedding 注入
3. 常用于 latent WM 规划中的 offline baseline

## 相关概念
- [[GCBC]]
- [[GCIVL]]
- [[LeFlow]]
