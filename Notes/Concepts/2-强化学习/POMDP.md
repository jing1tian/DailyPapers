---
type: concept
aliases: [部分可观测马尔可夫决策过程, Partially Observable MDP]
---

# POMDP

## 定义

**部分可观测马尔可夫决策过程**（POMDP）是描述智能体在无法直接观测完整状态的环境中进行序列决策的形式化框架。

## 数学形式

POMDP 由七元组 $(\mathcal{S}, \mathcal{A}, \mathcal{O}, T, O, R, \gamma)$ 定义：

- $\mathcal{S}$: 状态空间
- $\mathcal{A}$: 动作空间
- $\mathcal{O}$: 观测空间
- $T(\mathbf{s}' \mid \mathbf{s}, \mathbf{a})$: 状态转移模型
- $O(\mathbf{o} \mid \mathbf{s})$: 观测模型
- $R(\mathbf{s}, \mathbf{a})$: 奖励函数
- $\gamma \in [0,1)$: 折扣因子

**信念更新（Belief Update）**:

$$
\mathbf{x}_{t}(\mathbf{s}_{t}) \propto O(\mathbf{o}_{t} \mid \mathbf{s}_{t}) \sum_{s_{t-1}} T(\mathbf{s}_{t} \mid \mathbf{s}_{t-1}, \mathbf{a}_{t-1}) \mathbf{x}_{t-1}(\mathbf{s}_{t-1})
$$

其中 $\mathbf{x}_t$ 是对真实状态的后验信念分布。

## 核心要点

1. 智能体只能观测到观测值 $\mathbf{o}_t$，而非真实状态 $\mathbf{s}_t$
2. 需要维护**信念状态**（belief state）$\mathbf{x}_t$ 来聚合历史信息
3. 最优策略作用于信念状态而非直接状态
4. 世界模型在 POMDP 框架下承担 Renderer（$O$）和 Simulator（$T$）的角色

## 代表工作

- [[WorldModelRoadmap]]: 以 POMDP 为形式化基础定义世界模型三类功能角色
- [[MBRL]]: 学习 $T$ 和 $O$ 以减少真实环境交互

## 相关概念

- [[World Model]]
- [[MBRL]]
- [[Chain-of-Imagination]]
- [[World Action Model]]
