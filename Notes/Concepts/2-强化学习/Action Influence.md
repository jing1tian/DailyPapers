---
type: concept
aliases: [动作影响力, Action Influence Score, 动作影响量]
---

# Action Influence

## 定义

动作影响力（Action Influence）衡量训练数据集中**动作与状态转移之间的统计相关程度**：在给定当前状态的条件下，采取不同动作是否会导致明显不同的下一状态。正式定义为数据集上的条件互信息 $I(A; S' \mid S)$。

## 数学形式

$$
I(A; S' \mid S) = \mathbb{E}_{\rho(s)\pi(a|s)}\Big[D_{\text{KL}}\big(p(\cdot|s,a) \| p_\pi(\cdot|s)\big)\Big]
$$

其中 $p_\pi(\cdot|s) = \int p(\cdot|s,a)\pi(a|s)\,da$ 是在策略 $\pi$ 下的边际转移分布。

## 核心要点

1. **上界作用**: 训练数据集的 $I(A;S'|S)$ 严格上界动作条件世界模型的可控性（Explained Action Information），无论模型容量多大
2. **专家数据的局限**: 专家演示中所有动作的结果都差不多（都是成功），导致 $I(A;S'|S)$ 极低，训练出的世界模型倾向于忽略动作输入
3. **可优化性**: Action Influence Maximization（AIM）通过设计内在奖励主动采集高 $I(A;S'|S)$ 的数据，而非依靠增加专家数据量

## 代表工作

- [[AIM]]: 提出动作影响力的正式定义，并通过主动数据采集最大化该指标

## 相关概念

- [[互信息]]
- [[可控性]]
- [[Action-Conditioned World Model]]
- [[Curiosity-driven Exploration]]
