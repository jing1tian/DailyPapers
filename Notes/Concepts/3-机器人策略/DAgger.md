---
type: concept
aliases: [Dataset Aggregation, 数据集聚合]
---

# DAgger

## 定义

DAgger（Dataset Aggregation）是一种在线[[模仿学习]]算法，通过迭代地让专家对学习策略的访问状态提供标注，逐步消除行为克隆中的分布偏移问题。

## 数学形式

$$
\pi_{t+1} = \arg\min_\pi \mathbb{E}_{s \sim d_{\pi_t}} \left[ \ell\left(\pi(s),\, \pi^*(s)\right) \right]
$$

其中 $d_{\pi_t}$ 是当前策略 $\pi_t$ 访问的状态分布，$\pi^*$ 是专家策略。

## 核心要点

1. **在线数据聚合**: 每轮用当前策略执行 rollout，专家对新访问状态标注，累积到数据集中重训。
2. **消除分布偏移**: 与[[行为克隆]]不同，DAgger 的训练分布与部署分布在迭代中逐渐对齐。
3. **理论保证**: 在强凸损失下，DAgger 可以得到与专家策略相当的期望错误界。
4. **人机交互变体**: [[HG-DAgger]]（Human-Gated）在人类判断策略不确定时才触发干预，降低标注成本。

## 代表工作

- [[FlowDAgger]]: 在潜噪声空间执行 DAgger 循环，适配生成策略
- [[HG-DAgger]]: Human-Gated DAgger，降低人工干预频率

## 相关概念

- [[行为克隆]]
- [[模仿学习]]
- [[HG-DAgger]]
- [[FlowDAgger]]
