---
type: concept
aliases: [SMODICE, Stationary-distribution Matching Offline DIversity via Constrained optimization]
---

# SMODICE

## 定义
离线多样性最大化方法，通过约束优化在 demonstration 分布约束下最大化策略行为多样性，用于 quality-diversity 离线 RL。

## 核心要点
1. 在离线数据集上最大化 state-action 覆盖度
2. 用 stationary distribution matching 代替在线采样
3. 已知缺陷：分布偏移下容易崩溃，Dual-Force 提出改进

## 代表工作
- [[Dual-Force]]: 指出 SMODICE 的偏移问题并提出改进

## 相关概念
- [[Dual-Force]]
- [[QD]]（Quality-Diversity）
- [[DICE]]
