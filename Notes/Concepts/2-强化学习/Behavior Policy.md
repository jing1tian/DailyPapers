---
type: concept
aliases: [行为策略, Old Policy, Frozen Policy, 冻结策略]
---

# Behavior Policy

## 定义

行为策略（Behavior Policy）：在 on-policy 强化学习和蒸馏框架中，用于生成训练数据（轨迹/样本）的参考策略，通常为当前可训练策略的冻结副本或其指数移动平均。

## 数学形式

在 DiffusionOPSD 中，行为策略 $v_{old} \equiv v_{\theta_{old}}$ 是可训练策略 $v_\theta$ 的冻结副本，通过 [[EMA]] 更新：

$$
\theta_{old} \leftarrow \eta_{beh}\,\theta_{old} + (1 - \eta_{beh})\,\theta
$$

行为策略在每次外层迭代后更新，在迭代内（轨迹采集、目标构建、有限拟合期间）保持冻结。

## 核心要点

1. **On-policy 保证**：训练数据来自行为策略自身的轨迹，保证学习分布与生成分布一致
2. **冻结期稳定性**：拟合期间行为策略冻结，避免 target 随模型更新而漂移
3. **EMA 渐进更新**：避免行为策略突变，提供稳定的 anchor 基础
4. **与 Off-policy 的区别**：Off-policy 方法复用历史轨迹，行为策略则始终生成最新轨迹

## 代表工作

- [[DiffusionOPSD]] (2608.24646)：行为策略作为 query 采集和 anchor 构建的核心机制

## 相关概念

- [[EMA]]
- [[On-Policy Distillation]]
- [[RLHF]]
- [[Trust Region]]
