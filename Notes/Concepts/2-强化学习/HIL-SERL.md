---
type: concept
aliases: [Human-in-the-Loop SERL, 人类在环样本高效RL]
---

# HIL-SERL

## 定义

HIL-SERL（Human-in-the-Loop Sample-Efficient Reinforcement Learning）是一种结合人类干预的样本高效机器人 RL 框架，通过人类在线纠正加速真实机器人操作策略的训练。

## 核心要点

1. 基于 [[SERL]] 框架，在 RL 训练过程中允许人类操作员对机器人动作进行实时干预纠正
2. 使用小型高斯策略（非预训练大模型）从零开始训练
3. 结合人类示范数据和在线 RL 经验，实现高样本效率
4. 局限：不利用预训练 VLA 先验，无法直接从大模型能力受益

## 代表工作

- [[EXPO-FT]]: 作为对比基线，EXPO-FT 通过利用预训练 VLA 先验超越了 HIL-SERL

## 相关概念

- [[SERL]]
- [[强化学习]]
- [[模仿学习]]
- [[经验回放]]
