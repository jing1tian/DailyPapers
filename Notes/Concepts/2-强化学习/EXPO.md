---
type: concept
aliases: [EXPO Algorithm, Edit Policy RL, Expressive Policy Optimization]
---

# EXPO

## 定义

EXPO（Expressive Policy Optimization）是一种用于对扩散/流策略进行离策略强化学习微调的算法，通过维护"编辑策略"（edit policy）来预测对基础策略输出的残差修正量，避免对大型预训练模型进行直接反向传播。

## 数学形式

编辑策略损失：

$$
\mathcal{L}(\pi_{\text{edit}}) = -\mathbb{E}\left[Q_\phi(s_t, a_t + \hat{a}_t) - \alpha \log \pi_{\text{edit}}(\hat{a}_t | s_t, a_t)\right]
$$

在线策略选择：

$$
\tilde{a}^* = \arg\max_{a \in \bigcup_{i=1}^N \{a_i, \tilde{a}_i\}} Q_\phi(s, a)
$$

## 核心要点

1. **双策略设计**: 冻结的预训练基础策略 $\pi_{\text{base}}$ + 轻量级编辑策略 $\pi_{\text{edit}}$
2. **有界编辑**: 编辑量受限于范围 $[-\beta, \beta]$，防止偏离预训练先验过远
3. **离策略训练**: 使用经验回放缓冲区，支持高效的 off-policy 更新
4. **熵正则化**: 基于 SAC 的熵正则化，维持策略探索能力

## 代表工作

- [[EXPO-FT]]: 将 EXPO 扩展至 VLA 模型 RL 微调，支持动作块和人类干预

## 相关概念

- [[SAC]]
- [[Actor-Critic]]
- [[经验回放]]
- [[VLA]]
