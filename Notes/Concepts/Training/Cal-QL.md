---
type: concept
aliases: [Cal-QL, Calibrated Q-Learning, 校准Q学习]
---

# Cal-QL

## 定义

Cal-QL（Calibrated Q-Learning）是一种离线-在线过渡 RL 方法，通过校准正则化项解决 offline RL 训练的 Q 值保守性与 online fine-tuning 的分布偏移之间的冲突。

## 数学形式

Critic 校准损失：

$$
\mathcal{L}_Q(\theta) = \mathcal{L}_\text{TD}(\theta) + \lambda_\text{Cal} \mathcal{L}_\text{CalReg}(\theta)
$$

其中校准正则化项约束 Q 值不对 offline 数据上的动作过于保守，允许 online 数据平滑地扩展分布。

## 核心要点

1. **问题**：纯 offline RL（如 CQL）的保守 Q 值在 online 阶段限制了探索
2. **解法**：校准正则化项让 Q 值在 offline 数据上保持合理估计，online 数据上可以自由更新
3. **h 步 Bellman 回归**：结合多步累积回报，适配动作块（Action Chunk）预测结构
4. **与 RLPD 的关系**：Cal-QL 是 RLPD 框架的扩展，加入了校准机制

## 代表工作

- [[DyGRO-VLA]]: 在 VLA 多任务 RL 场景中使用 Cal-QL 平滑离线到在线过渡
- Nakamoto et al. (2023): Cal-QL 原始论文

## 相关概念

- [[Actor-Critic]]
- [[强化学习]]
- [[Action Chunking]]
