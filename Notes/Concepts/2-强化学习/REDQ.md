---
type: concept
aliases: [RedQ, Randomized Ensembled Double Q-learning, REDQ Ensemble]
---

# REDQ

## 定义

REDQ（Randomized Ensembled Double Q-learning）是一种通过维护大规模 Q 网络集成来降低价值估计偏差、提升样本效率的强化学习算法。通过在每次梯度更新时随机子采样 2 个 Q 网络计算目标值，有效减少 Q 值过估计问题。

## 数学形式

TD 目标值计算（子采样 M 个网络中的最小值）：

$$
y = r + \gamma \min_{i \in \mathcal{M}} Q_{\phi'_i}(s', a')
$$

其中 $\mathcal{M}$ 是从 N 个网络中随机采样的大小为 M 的子集。

## 核心要点

1. **大集成**: 维护 N 个（通常 10 个）Q 网络，远多于标准 Double Q-learning 的 2 个
2. **随机子采样**: 每次更新随机选取 M 个网络计算目标，引入随机性降低偏差
3. **高 UTD 比**: 支持高 update-to-data (UTD) 比，每步环境交互进行多次梯度更新
4. **层归一化**: 常与层归一化配合使用，稳定高 UTD 比下的训练

## 代表工作

- [[EXPO-FT]]: 使用 REDQ 风格的 10 个 Q 网络集成进行 VLA RL 微调
- [[RLPD]]: 结合 REDQ 和离线数据的高效 RL 方法

## 相关概念

- [[SAC]]
- [[Actor-Critic]]
- [[RLPD]]
- [[经验回放]]
