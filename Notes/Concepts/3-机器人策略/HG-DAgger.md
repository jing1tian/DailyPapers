---
type: concept
aliases: [Human-Gated DAgger, 人类门控数据聚合]
---

# HG-DAgger

## 定义

HG-DAgger（Human-Gated Dataset Aggregation）是 DAgger 算法的变体，通过人类操作员主动触发干预（而非持续监督）来收集纠正示范数据，实现更高效的模仿学习。

## 数学形式

与标准 DAgger 相同，迭代构建数据集：

$$D_{i+1} = D_i \cup \{(s, \pi^*(s)) : s \in \text{human-gated states}\}$$

## 核心要点

1. **人类门控**: 仅在人类判断策略将失败时才介入，减少人工标注负担
2. **纯模仿学习**: 没有奖励函数，不做 RL 优化，受限于示范分布上限
3. **适用于 VLA**: 可用于预训练 VLA 的持续改进
4. **局限**: 无法超越人类示范的质量上限，无法通过奖励探索新行为

## 代表工作

- [[EXPO-FT]]: 作为对比基线，EXPO-FT 通过 RL 奖励优化超越了 HG-DAgger 的纯模仿学习上限
- [[FlowDAgger]]: 在潜噪声空间执行 DAgger 循环的方法，与 HG-DAgger 共享在线干预框架

## 相关概念

- [[模仿学习]]
- [[行为克隆]]
- [[强化学习]]
