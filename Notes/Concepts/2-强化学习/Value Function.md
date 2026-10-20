---
type: concept
aliases: [价值函数, Q函数, Q-Function, 状态价值函数, V函数]
---

# Value Function（价值函数）

## 定义

在强化学习中，**价值函数**估计从某状态（或状态-动作对）出发，按当前策略执行后能获得的**期望累计折扣回报**，用于评估状态或动作的"好坏"。

## 数学形式

**状态价值函数**（V 函数）：

$$
V^\pi(s) = \mathbb{E}_\pi\!\left[\sum_{t=0}^{\infty} \gamma^t r_t \mid s_0 = s\right]
$$

**动作价值函数**（Q 函数）：

$$
Q^\pi(s, a) = \mathbb{E}_\pi\!\left[\sum_{t=0}^{\infty} \gamma^t r_t \mid s_0 = s, a_0 = a\right]
$$

**Bellman 方程**：

$$
Q^\pi(s, a) = r + \gamma \mathbb{E}_{s' \sim P, a' \sim \pi}[Q^\pi(s', a')]
$$

## 核心要点

1. **策略评估**：给定策略 π，价值函数反映其质量
2. **最优价值**：$Q^*(s,a) = \max_\pi Q^\pi(s,a)$，用于 Q-Learning / DQN 等方法
3. **离线 RL 应用**：从固定数据集估计 Q 值，不需要在线交互
4. **在机器人中的应用**：对多个候选动作评分，选择价值最高的执行（Best-of-N 采样）

## 代表工作

- [[LAW]]: 用 101-bin [[HL-Gauss]] 离散价值头估计子任务级 Q 值，对 WAM 候选动作评分

## 相关概念

- [[子任务局部价值学习]]
- [[HL-Gauss]]
- [[Action Chunking]]
- [[World Action Model]]
